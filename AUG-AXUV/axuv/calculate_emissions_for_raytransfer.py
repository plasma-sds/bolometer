# This script is used for calculating emissions for sector 16 of AUG

import argparse
import csv
from pathlib import Path

import cherab.core.atomic.elements as elements
import h5py
import numpy as np
from cherab.core import Line, Maxwellian, Plasma, Species
from cherab.core.math import AxisymmetricMapper
from cherab.core.model import (
    Bremsstrahlung,
    ExcitationLine,
    GaussianLine,
    RecombinationLine,
)
from cherab.openadas import OpenADAS
from cherab.tools.primitives import axisymmetric_mesh_from_polygon
from raysect.core import Vector3D
from raysect.core.math.function.float import Interpolator2DArray
from scipy.constants import atomic_mass, electron_mass
from scipy.spatial import ConvexHull

from axuv.cameras import SECTOR_CAMERAS, create_observable_world
from axuv.interpolation import (
    POLOIDAL_RMAX,
    POLOIDAL_RMIN,
    POLOIDAL_ZMAX,
    POLOIDAL_ZMIN,
    interpolate_parameters,
)
from axuv.io import DATADIR, RAYTRANSFER_PATH, SAVEDIR, load_axuv_df
from axuv.plasma import emission_function_3d, get_spectrum_part


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate AXUV emissions from a JOREK HDF5 file."
    )
    parser.add_argument("input_file", help="Path to the JOREK output HDF5 file")
    parser.add_argument(
        "--sectors",
        "-s",
        nargs="+",
        default=["S16"],
        choices=list(SECTOR_CAMERAS),
        metavar="SECTOR",
        help=f"Sectors to simulate. Choices: {list(SECTOR_CAMERAS)}. Default: S16.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    INPUT_FILENAME = args.input_file
    SECTORS = args.sectors  # e.g. ["S5", "S16"]

    # ... load data ...

    mask_negative = True
    test_uniform = False
    print(INPUT_FILENAME)
    with h5py.File(INPUT_FILENAME, "r") as f:
        majorR = np.asarray(f["R"])[:, np.newaxis]
        zaxis = np.asarray(f["Z"])[:, np.newaxis]

        # np.column_stack handles the newaxis automatically for 1D datasets
        neon = np.column_stack([np.asarray(f[f"Ne{i}"]) for i in range(11)])

        eTemp = np.asarray(f["Te"])[:, np.newaxis]
        eDens = np.asarray(f["ne"])[:, np.newaxis]
        if "time" in f:
            SI_time = f["time"][()]
        elif "t_now" in f:
            SI_time = f["t_now"][()]
        else:
            # Assumes filename like: .../input/output_step<value>_out.h5
            SI_time = (
                Path(INPUT_FILENAME)
                .stem.removeprefix(  # "output_step<value>_out"
                    "output_step"
                )  # "<value>_out"
                .removesuffix("_out")  # "<value>"
            )

    if mask_negative:
        eTemp = np.maximum(eTemp, 1.0)
        eDens = np.maximum(eDens, 0)
        neon = np.maximum(neon, 0)

    # Uniform test values per charge state (Ne0..Ne10)
    UNIFORM_NEON_VALUES = [
        7e9,
        1e15,
        1e18,
        3e18,
        2e18,
        1e18,
        6e17,
        5e17,
        2e17,
        5e14,
        1e12,
    ]

    if test_uniform:
        eTemp.fill(10.0)
        eDens.fill(1e20)
        for i, val in enumerate(UNIFORM_NEON_VALUES):
            neon[:, i] = val

    neonlist = [neon[:, i] for i in range(11)]

    # Setting up interpolation of JOREK data
    # In this case the vertical and horizontal distances between the gridpoints will be the same
    # Later the voxel grid will have the same dimensions, but will be masked where there is no plasma
    # this results in ~4 GB memory allocation for the creation of the ~4000 element voxel grid
    # NOTE Doubling the total number of grid points results in a 2^2=4 times increase in the memory needed!
    # NOTE Doubling the resolution in both directions results in a (2*2)^2=16 times increase!
    resolution_R = 60
    resolution_z = 110

    # The points at which the JOREK data is defined
    points = np.hstack([majorR, zaxis])

    interpolated_neon, interpolated_eTemp, interpolated_eDens = interpolate_parameters(
        points, resolution_R, resolution_z, neonlist, eTemp, eDens, method="linear"
    )

    try:
        with h5py.File(RAYTRANSFER_PATH, "r") as h5f:
            sensitivity_matrix = h5f["sensitivity_matrix"][()]
            grid_centres = h5f["grid_centres"][()]
            voxel_map = h5f["voxel_map"][()]
            inverse_voxel_map = h5f["inverse_voxel_map"][()]
            laplacian = h5f["laplacian"][()]
            mask = h5f["mask"][()]
            completed_bins = h5f["completed_bins"][()]
        print("Sensitivity matrix loaded")
    except Exception as e:
        print(f"Could not load sensitivity matrix from {RAYTRANSFER_PATH}: {e}")

    # A convex hull is created around the JOREK datapoints to be used as boundary for the voxel grid later
    hull = ConvexHull(points)
    convex_hull = points[hull.vertices]

    # Load the AXUV diode geometry data
    axuv_df = load_axuv_df()

    world, cameras = create_observable_world(
        sectors=SECTORS,
        axuv_df=axuv_df,
        cad_mesh=False,
        show_plots=False,
    )

    print("Creating plasma...")
    plasma = Plasma(parent=world)
    plasma.atomic_data = OpenADAS(permit_extrapolation=True)
    plasma_mesh = axisymmetric_mesh_from_polygon(convex_hull)
    plasma.geometry = plasma_mesh

    linspace_R = np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, resolution_R)
    linspace_z = np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, resolution_z)

    # No net velocity for any species
    zero_velocity = Vector3D(0, 0, 0)

    deuterium_mass = elements.deuterium.atomic_weight * atomic_mass
    neon_mass = elements.neon.atomic_weight * atomic_mass

    extrap_x = 0.1
    extrap_y = 0.1

    # D1 density from quasi-neutrality: ne = sum(Z_i * n_i)
    # Ne0 contributes 0 electrons, Ne1 → 1, ...
    charges = np.arange(1, 11)  # shape (10,)
    neon_electron_contributions = np.sum(
        charges[:, None, None] * interpolated_neon[1:, :, :], axis=0
    )
    calculated_d1 = interpolated_eDens - neon_electron_contributions

    # create 2D interpolators for the densities and temperature
    e_density_interp = Interpolator2DArray(
        linspace_R,
        linspace_z,
        interpolated_eDens,
        interpolation_type="linear",
        extrapolation_type="nearest",
        extrapolation_range_x=extrap_x,
        extrapolation_range_y=extrap_y,
    )
    e_temperature_interp = Interpolator2DArray(
        linspace_R,
        linspace_z,
        interpolated_eTemp,
        "linear",
        "nearest",
        extrap_x,
        extrap_y,
    )

    # map the 2D interpolators into 3D functions using the axisymmetry operator
    e_density = AxisymmetricMapper(e_density_interp)
    e_temperature = AxisymmetricMapper(e_temperature_interp)

    de1_density_interp = Interpolator2DArray(
        linspace_R, linspace_z, calculated_d1, "linear", "nearest", extrap_x, extrap_y
    )

    de1_density = AxisymmetricMapper(de1_density_interp)

    # Set up the distributions to be Maxwellians
    e_distribution = Maxwellian(e_density, e_temperature, zero_velocity, electron_mass)

    de1_distribution = Maxwellian(
        de1_density, e_temperature, zero_velocity, deuterium_mass
    )

    # Define the different plasma species
    de1_species = Species(elements.deuterium, 1, de1_distribution)

    # Neon: build interpolators -> 3D mappers -> Maxwellian distributions -> Species in one pass
    neon_species = []
    for i in range(11):
        interp = Interpolator2DArray(
            linspace_R,
            linspace_z,
            interpolated_neon[i, :, :],
            "linear",
            "nearest",
            extrap_x,
            extrap_y,
        )
        density = AxisymmetricMapper(interp)
        dist = Maxwellian(density, e_temperature, zero_velocity, neon_mass)
        neon_species.append(Species(elements.neon, i, dist))

    ##############################################################
    # Get Neon lines from Photon Emissivity Coefficients datafiles
    ##############################################################
    neon_lines = []
    with open(DATADIR + "ne.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            i, part1, part2 = row

            neon_lines.append(
                ExcitationLine(
                    Line(elements.neon, int(i), (part1, part2)), lineshape=GaussianLine
                )
            )
            neon_lines.append(
                RecombinationLine(
                    Line(elements.neon, int(i), (part1, part2)), lineshape=GaussianLine
                )
            )

    # add all neon lines to the plasma + Bremsstrahlung
    plasma.models = [*neon_lines, Bremsstrahlung()]

    # define species, field and composition
    plasma.b_field = Vector3D(0, 0, 0)
    plasma.electron_distribution = e_distribution
    plasma.composition = [*neon_species, de1_species]

    # Define spectral measurements array - has to be size: num of diodes by spectral bins
    NUM_OF_DIODES = sensitivity_matrix.shape[0]
    
    # ── Spectral configuration ───────────────────────────────────────────────────
    SPECTRAL_BINS = 100  # number of spectral bins in each spectrum part
    MIN_WAVELENGTHS = [1, 12.4, 124.0]  # nm  (photon energies: 1240, 100, 10 eV)
    MAX_WAVELENGTHS = [12.4, 124.0, 1240.0]  # nm  (photon energies:  100,  10,  1 eV)

    wavelengths = np.unique(
        np.array(
            [
                *get_spectrum_part(0, MIN_WAVELENGTHS, MAX_WAVELENGTHS, SPECTRAL_BINS),
                *get_spectrum_part(1, MIN_WAVELENGTHS, MAX_WAVELENGTHS, SPECTRAL_BINS),
                *get_spectrum_part(2, MIN_WAVELENGTHS, MAX_WAVELENGTHS, SPECTRAL_BINS),
            ]
        )
    )
    energies_eV = 1239.8 / wavelengths
    total_wavelength_bins = len(wavelengths) - 1

    emissions = np.zeros([inverse_voxel_map.shape[0], total_wavelength_bins])

    for i in range(inverse_voxel_map.shape[0]):
        # Get the indices of the i-th voxel in the grid_centres array
        aa = inverse_voxel_map[i, 0, 0]
        cc = inverse_voxel_map[i, 2, 0]

        # Get the real world coordinates corresponding to the i-th voxel
        xi = grid_centres[aa, cc, 0]  # To get R coordinate
        yi = 0  # Assume y coordinate is 0
        zi = grid_centres[aa, cc, 1]  # To get z coordinate

        emission_in_point = np.zeros(total_wavelength_bins)
        for part in range(3):
            emission_in_point[part * 99 : (part + 1) * 99] = emission_function_3d(
                xi,
                yi,
                zi,
                part,
                plasma,
                MIN_WAVELENGTHS,
                MAX_WAVELENGTHS,
                SPECTRAL_BINS,
            )

        emissions[i, :] = emission_in_point
        print(str(i) + "/" + str(inverse_voxel_map.shape[0]), end="\r")

    measured_spectra = np.zeros([NUM_OF_DIODES, total_wavelength_bins])
    for i in range(NUM_OF_DIODES):
        for j in range(total_wavelength_bins):
            measured_spectra[i, j] = np.sum(
                sensitivity_matrix[i, :, j] * emissions[:, j]
            )

    # Saving the emission data as HDF5
    if type(SI_time) is float:
        SI_time = np.round(SI_time, 6)
    savename: str = SAVEDIR + "raytransfer_emissions_lowres_1eV_" + str(SI_time) + ".h5"
    print(savename)
    with h5py.File(savename, "w") as file:
        file.create_dataset("emissions", data=emissions)
        file.create_dataset("wavelengths", data=wavelengths)
        file.create_dataset("energies", data=energies_eV)
        file.create_dataset("diode_measurements", data=measured_spectra)

    print("\nSaved emission data.")
