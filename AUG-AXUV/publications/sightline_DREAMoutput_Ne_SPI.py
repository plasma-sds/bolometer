"""Spectrum of a DREAM SPI plasma along one sightline through a cylinder model.

Two inputs:
- the bundled DREAM output (data file dream/dream_output.h5), with the
  observation time as the positional argument, as in section 3 of the paper;
- a DREAM SPI case directory (init_0.h5, init_1.h5, ..., main.h5) with --case
  and --t-rel, the time relative to the thermal quench onset.

Species are read from the file's ionmeta (names and Z), see axuv.dream. For
cases with W and B, the spectrum is also observed separately for the W lines,
the B lines and the rest (all other lines and Bremsstrahlung), so their shares
of the power can be computed. Emission is additive, so the parts sum to the
total.
"""

import argparse
import bisect
import pickle
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from cherab.core import Maxwellian, Plasma, Species
from cherab.core.model import Bremsstrahlung
from cherab.openadas import OpenADAS
from raysect.core import SerialEngine, Vector3D, rotate_basis, translate
from raysect.optical import World
from raysect.optical.material.emitter.inhomogeneous import NumericalIntegrator
from raysect.optical.observer import SightLine, SpectralPowerPipeline0D, SpectralRadiancePipeline0D
from raysect.primitive import Cylinder
from scipy.constants import atomic_mass, electron_mass
from scipy.interpolate import CubicSpline

from axuv.atomic import add_variant_argument, install_atomic_data, line_models
from axuv.datafiles import data_path
from axuv.dream import load_case, load_file, species_densities
from axuv.io import OUTPUT_DIR

# must be set before any pyplot import; keeps observe() from blocking on headless machines
matplotlib.use('Agg')

XAXIS = Vector3D(1, 0, 0)
ZAXIS = Vector3D(0, 0, 1)

MAJOR_RADIUS = 1.7365249  # from DREAM
CENTRE_Z = 0
CYLINDER_HEIGHT = 2

SPECTRAL_BINS = 2500
MIN_WAVELENGTH = 0.25  # nm, about 5 keV
MAX_WAVELENGTH = 1240  # nm, 1 eV

# elements whose line emission is observed as a separate part
SEPARATE_ELEMENTS = ("W", "B", "C")


class CustomFunction:
    """3D radial interpolator for temperature and density profiles.

    Cubic spline in minor radius, zero outside the plasma. Negative values from
    spline overshoot between grid points are set to zero.
    """

    def __init__(self, radialgrid, data):
        self.radialgrid = radialgrid
        self.dr = abs(radialgrid[-1] - radialgrid[-2])
        self.interpolated = CubicSpline(radialgrid, data)

    def __call__(self, x, y, z):
        if abs(z) >= CYLINDER_HEIGHT / 2:
            return 0.
        radius = np.sqrt((x - MAJOR_RADIUS) ** 2 + y ** 2)
        if radius <= self.radialgrid[-1] + self.dr / 2:
            return max(float(self.interpolated(radius)), 0.)
        return 0.


def build_scene(case, index, pixel_samples, elements=None):
    """World, sightline and plasma for one step of a DREAM case.

    :return: (los, power pipeline, radiance pipeline, plasma)
    """
    world = World()
    # SightLine looks in the direction of XAXIS with ZAXIS up
    los = SightLine(parent=world, transform=rotate_basis(XAXIS, ZAXIS) * translate(0, 0, 0))
    # no accumulation: each observe() of a model part starts from zero
    sppipeline = SpectralPowerPipeline0D(accumulate=False)
    sradpipeline = SpectralRadiancePipeline0D(accumulate=False)
    los.pipelines = [sppipeline, sradpipeline]
    los.spectral_rays = 1
    los.spectral_bins = SPECTRAL_BINS
    los.ray_extinction_prob = 0.01
    los.min_wavelength = MIN_WAVELENGTH
    los.max_wavelength = MAX_WAVELENGTH
    los.pixel_samples = pixel_samples
    if pixel_samples == 1:
        los.render_engine = SerialEngine()

    minor_radius = case.r_f[-1]
    sigma = 0.25
    plasma = Plasma(parent=world)
    plasma.atomic_data = OpenADAS(permit_extrapolation=True)
    plasma.geometry = Cylinder(
        radius=MAJOR_RADIUS + minor_radius, height=CYLINDER_HEIGHT,
        transform=translate(0, 0, CENTRE_Z - CYLINDER_HEIGHT / 2),
    )
    plasma.geometry_transform = None
    plasma.integrator = NumericalIntegrator(step=sigma / 5.0)

    bulk_velocity = Vector3D(0, 0, 0)
    temperature = CustomFunction(case.r, case.T_cold[index])
    electron_density = CustomFunction(case.r, case.n_free[index])
    plasma.b_field = Vector3D(1.0, 1.0, 1.0)
    plasma.electron_distribution = Maxwellian(
        electron_density, temperature, bulk_velocity, electron_mass
    )
    composition = []
    for (element, charge), density in species_densities(case, index, elements).items():
        mass = element.atomic_weight * atomic_mass
        distribution = Maxwellian(CustomFunction(case.r, density), temperature, bulk_velocity, mass)
        composition.append(Species(element, charge, distribution))
    plasma.composition = composition
    return los, sppipeline, sradpipeline, plasma


def model_parts(plasma, variants):
    """Line and continuum models, split into the parts observed separately.

    :return: Dict from part name ("W", "B", ..., "rest") to a model list; the
        rest holds all other lines and Bremsstrahlung.
    """
    parts = {}
    for model in line_models(plasma, variants=variants):
        symbol = model.line.element.symbol
        name = symbol if symbol in SEPARATE_ELEMENTS else "rest"
        parts.setdefault(name, []).append(model)
    parts.setdefault("rest", []).append(Bremsstrahlung())
    return parts


def observe(los, pipeline, plasma, models):
    """Spectral power (W/nm) for one set of plasma models."""
    plasma.models = models
    los.observe()
    return np.array(pipeline.samples.mean), np.array(pipeline.samples.variance)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Observe plasma spectrum from DREAM output along a sightline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("time", type=float, nargs="?",
                        help="Observation time [s] in the bundled DREAM output (2.3 s + DREAM time).")
    parser.add_argument("--case", type=Path, metavar="DIR",
                        help="DREAM SPI case directory (init_*.h5, main.h5); use with --t-rel.")
    parser.add_argument("--t-rel", type=float, metavar="S",
                        help="Time relative to the TQ onset [s], for --case.")
    parser.add_argument("--output", type=Path, metavar="FILE",
                        help="Output HDF5 file; default under $AXUV_OUTPUT_DIR/1D_data/.")
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
        help="Install the Open-ADAS data (axuv.atomic.install_atomic_data) before simulation.",
    )
    add_variant_argument(parser)
    args = parser.parse_args()
    if (args.case is None) == (args.time is None):
        parser.error("give either a time (bundled DREAM output) or --case with --t-rel")
    if args.case is not None and args.t_rel is None:
        parser.error("--case needs --t-rel")
    return args


if __name__ == "__main__":
    args = _parse_args()

    if args.download_adas:
        install_atomic_data()

    if args.case is None:
        case = load_file(data_path("dream/dream_output.h5"))
        do_time = 2.3 + case.t
        ti = bisect.bisect_left(do_time, args.time)
        if ti >= len(do_time):
            raise SystemExit(f"Time {args.time} s is beyond DREAM grid end {do_time[-1]:.4f} s")
        observation_time = float(do_time[ti])
        default_name = "highres_{:.4f}s".format(observation_time)
    else:
        case = load_case(args.case)
        ti = case.index_at(args.t_rel)
        observation_time = float(case.t_rel[ti])
        default_name = f"{args.case.name}_trel{observation_time * 1e3:+.3f}ms"

    print("Building the cylindrical plasma...")
    los, sppipeline, sradpipeline, plasma = build_scene(case, ti, args.pixel_samples)
    parts = model_parts(plasma, args.pec_variant)

    print("Observing the radiation spectrum along the LOS...")
    total, variance = observe(los, sppipeline, plasma, [m for ms in parts.values() for m in ms])
    part_spectra = {}
    if len(parts) > 1:
        for name, models in parts.items():
            part_spectra[name] = observe(los, sppipeline, plasma, models)[0]
        # the radiance pipeline keeps the last observe(); restore the total for --pickle
        if args.pickle:
            observe(los, sppipeline, plasma, [m for ms in parts.values() for m in ms])

    hdf5_path = args.output or OUTPUT_DIR / "1D_data" / f"{default_name}.h5"
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as h5f:
        h5f.create_dataset("spectral_power_mean",     data=total)
        h5f.create_dataset("spectral_power_variance", data=variance)
        h5f.create_dataset("wavelengths_nm",          data=sppipeline.wavelengths)
        h5f.create_dataset("photon_energies_eV",      data=1239.8 / sppipeline.wavelengths)
        for name, spectrum in part_spectra.items():
            h5f.create_dataset(f"parts/{name}", data=spectrum)
        h5f.attrs["observation_time_s"] = observation_time
        h5f.attrs["pixel_samples"]      = args.pixel_samples
        h5f.attrs["spectral_bins"]      = los.spectral_bins
        h5f.attrs["min_wavelength_nm"]  = los.min_wavelength
        h5f.attrs["max_wavelength_nm"]  = los.max_wavelength
        h5f.attrs["flags"]              = "\n".join(case.flags)

    print(f"Saved HDF5 to {hdf5_path}")

    if args.pickle:
        spectrum = sradpipeline.to_spectrum()
        pickle_path = hdf5_path.with_name(hdf5_path.stem + "_spectrum.pkl")
        with open(pickle_path, "wb") as fh:
            pickle.dump(spectrum, fh)
        print(f"Saved spectrum pickle to {pickle_path}")
