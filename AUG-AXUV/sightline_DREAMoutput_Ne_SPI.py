import argparse
import bisect
import csv
import pickle
import matplotlib
import h5py
import numpy as np
from scipy.interpolate import CubicSpline

from scipy.constants import electron_mass, atomic_mass

from raysect.core import Vector3D, translate, rotate_basis
from raysect.optical import World
from raysect.primitive import Cylinder
from raysect.optical.observer import SightLine, SpectralPowerPipeline0D, SpectralRadiancePipeline0D
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator

from cherab.core import Species, Maxwellian, Plasma, Line
from cherab.core.atomic import deuterium, neon
from cherab.core.model import ExcitationLine, GaussianLine, RecombinationLine, Bremsstrahlung
from cherab.openadas import OpenADAS

from axuv.io import DATADIR, PROJECT_ROOT


# must be set before any pyplot import; keeps observe() from blocking on headless machines
matplotlib.use('Agg')

XAXIS = Vector3D(1, 0, 0)
ZAXIS = Vector3D(0, 0, 1)

MAJOR_RADIUS = 1.7365249  # from DREAM
CENTRE_Z = 0
CYLINDER_HEIGHT = 2


class CustomFunction:
    """3D radial interpolator for temperature and density profiles."""

    def __init__(self, radialgrid, data):
        self.radialgrid = radialgrid
        self.dr = abs(radialgrid[-1] - radialgrid[-2])
        self.interpolated = CubicSpline(radialgrid, data)

    def __call__(self, x, y, z):
        if abs(z) >= CYLINDER_HEIGHT / 2:
            return 0.
        radius = np.sqrt((x - MAJOR_RADIUS) ** 2 + y ** 2)
        if radius <= self.radialgrid[-1] + self.dr / 2:
            return self.interpolated(radius)
        return 0.


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Observe plasma spectrum from DREAM output along a sightline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("time", type=float, help="Observation time [s].")
    parser.add_argument(
        "--pixel-samples", type=float, default=3e4, metavar="N",
        help="Number of rays per pixel (los.pixel_samples).",
    )
    parser.add_argument(
        "--pickle", action="store_true",
        help="Also export the spectral radiance object as a pickle file.",
    )
    parser.add_argument(
        "--download-adas", action="store_true",
        help="Run populate() from cherab.openadas.repository before simulation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.download_adas:
        from cherab.openadas.repository import populate
        populate()

    # Load DREAM output
    with h5py.File(DATADIR + "dream_output.h5") as do:
        do_ions = do["eqsys/n_i"][()]
        do_n_cold = do["eqsys/n_cold"][()]
        do_n_hot = do["eqsys/n_hot"][()]
        do_n_re = do["eqsys/n_re"][()]
        do_n_free = do_n_cold + do_n_hot + do_n_re
        do_time = 2.3 + do["grid/t"][()]
        do_minor_radius = do["grid/r"][()]
        do_minor_radius_edges = do["grid/r_f"][()]
        do_temperature = do["eqsys/T_cold"][()]
        do_radiated_power = do["other/fluid/Tcold_radiation"][()]
        d0_density = do_ions[:, 0, :]
        d0i_density = do_ions[:, 2, :]
        d1_density = do_ions[:, 1, :]
        d1i_density = do_ions[:, 3, :]
        ne_densities = do_ions[:, 4:, :]

    ti = bisect.bisect_left(do_time, args.time)
    if ti >= len(do_time):
        raise SystemExit(f"Time {args.time} s is beyond DREAM grid end {do_time[-1]:.4f} s")

    MINOR_RADIUS = do_minor_radius_edges[-1]

    # Scene
    world = World()

    # SightLine looks in the direction of XAXIS with ZAXIS up
    los = SightLine(parent=world, transform=rotate_basis(XAXIS, ZAXIS) * translate(0, 0, 0))
    sppipeline = SpectralPowerPipeline0D()
    sradpipeline = SpectralRadiancePipeline0D()
    los.pipelines = [sppipeline, sradpipeline]
    los.spectral_rays = 1
    los.spectral_bins = 2500
    los.ray_extinction_prob = 0.01
    los.min_wavelength = 0.25
    los.max_wavelength = 1240
    los.pixel_samples = args.pixel_samples

    # Plasma
    print("Building the cylindrical plasma...")
    adas = OpenADAS(permit_extrapolation=True)

    sigma = 0.25
    plasma = Plasma(parent=world)
    plasma.atomic_data = adas
    plasma.geometry = Cylinder(
        radius=MAJOR_RADIUS + MINOR_RADIUS, height=CYLINDER_HEIGHT,
        transform=translate(0, 0, CENTRE_Z - CYLINDER_HEIGHT / 2),
    )
    plasma.geometry_transform = None
    plasma.integrator = NumericalIntegrator(step=sigma / 5.0)

    bulk_velocity = Vector3D(0, 0, 0)
    deuterium_mass = deuterium.atomic_weight * atomic_mass
    neon_mass = neon.atomic_weight * atomic_mass

    temperature = CustomFunction(do_minor_radius, do_temperature[ti, :])
    electron_density = CustomFunction(do_minor_radius, do_n_free[ti, :])

    fd0_density = CustomFunction(do_minor_radius, d0_density[ti, :] + d0i_density[ti, :])
    fd1_density = CustomFunction(do_minor_radius, d1_density[ti, :] + d1i_density[ti, :])

    e_distribution = Maxwellian(electron_density, temperature, bulk_velocity, electron_mass)
    d0_distribution = Maxwellian(fd0_density, temperature, bulk_velocity, deuterium_mass)
    d1_distribution = Maxwellian(fd1_density, temperature, bulk_velocity, deuterium_mass)

    ne_density_funcs = [
        CustomFunction(do_minor_radius, ne_densities[ti, i, :]) for i in range(11)
    ]
    ne_distributions = [
        Maxwellian(f, temperature, bulk_velocity, neon_mass) for f in ne_density_funcs
    ]
    ne_species = [Species(neon, i, ne_distributions[i]) for i in range(11)]

    plasma.b_field = Vector3D(1.0, 1.0, 1.0)
    plasma.electron_distribution = e_distribution
    plasma.composition = [
        Species(deuterium, 0, d0_distribution),
        Species(deuterium, 1, d1_distribution),
        *ne_species,
    ]

    # Deuterium lines: full Lyman (n→1, 91–122 nm) and Balmer (n→2, 375–656 nm) series
    deuterium_lines = []
    for lower_n, upper_range in [(1, range(2, 13)), (2, range(3, 13))]:
        for upper_n in upper_range:
            line = Line(deuterium, 0, (upper_n, lower_n))
            deuterium_lines.append(ExcitationLine(line, lineshape=GaussianLine))
            deuterium_lines.append(RecombinationLine(line, lineshape=GaussianLine))

    # Neon lines from ne.csv
    neon_lines = []
    with open(DATADIR + "ne.csv") as f:
        for i, part1, part2 in csv.reader(f):
            line = Line(neon, int(i), (part1, part2))
            neon_lines.append(ExcitationLine(line, lineshape=GaussianLine))
            neon_lines.append(RecombinationLine(line, lineshape=GaussianLine))

    plasma.models = [*deuterium_lines, *neon_lines, Bremsstrahlung()]

    # Observe
    print("Observing the radiation spectrum along the LOS...")
    los.observe()

    # Save HDF5
    folder_name = PROJECT_ROOT / "output" / "1D_data"
    folder_name.mkdir(parents=True, exist_ok=True)
    hdf5_path = folder_name / "highres_{:.4f}s.h5".format(do_time[ti])

    with h5py.File(hdf5_path, "w") as h5f:
        h5f.create_dataset("spectral_power_mean",     data=sppipeline.samples.mean)
        h5f.create_dataset("spectral_power_variance", data=sppipeline.samples.variance)
        h5f.create_dataset("wavelengths_nm",          data=sppipeline.wavelengths)
        h5f.create_dataset("photon_energies_eV",      data=1239.8 / sppipeline.wavelengths)
        h5f.attrs["observation_time_s"] = float(do_time[ti])
        h5f.attrs["pixel_samples"]      = args.pixel_samples
        h5f.attrs["spectral_bins"]      = los.spectral_bins
        h5f.attrs["min_wavelength_nm"]  = los.min_wavelength
        h5f.attrs["max_wavelength_nm"]  = los.max_wavelength

    print(f"Saved HDF5 to {hdf5_path}")

    if args.pickle:
        spectrum = sradpipeline.to_spectrum()
        pickle_path = folder_name / "highres_{:.4f}s_spectrum.pkl".format(do_time[ti])
        with open(pickle_path, "wb") as fh:
            pickle.dump(spectrum, fh)
        print(f"Saved spectrum pickle to {pickle_path}")
