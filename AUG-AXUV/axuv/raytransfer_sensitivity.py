"""
raytransfer_sensitivity.py
--------------------------
Calculates the AXUV bolometer sensitivity (geometry) matrix for one or more
AUG sectors using RayTransferCylinder.

The sensitivity matrix encodes, for every diode and every wavelength bin, how
much of the emission from each spatial voxel reaches that diode.  It is a
purely geometric quantity and does NOT depend on plasma data.

An optional JOREK HDF5 file can be supplied via --jorek-file to trim the voxel
grid to the convex hull of the plasma data.  If omitted, the full rectangular
grid is used.

Usage
-----
    # Sector 5, no reflections (simple absorbing wall):
    python raytransfer_sensitivity.py --sectors S5

    # Sector 16, with full CAD mesh (reflections enabled):
    python raytransfer_sensitivity.py --sectors S16 --reflections

    # Both sectors together, 500 k pixel samples:
    python raytransfer_sensitivity.py --sectors S5 S16 --pixel-samples 500000

    # Sector 5 masked to the JOREK plasma extent:
    python raytransfer_sensitivity.py --sectors S5 --jorek-file data/step02410_out.h5

Output
------
An HDF5 file whose name encodes the sectors and reflection mode:
    raytransfer_<sectors>_[no]refl.h5
e.g.  raytransfer_S5_norefl.h5
      raytransfer_S16_refl.h5
      raytransfer_S5_S16_norefl.h5

The file supports resuming: completed wavelength bins are skipped on re-run.
"""

import argparse
import os
import time

import h5py
import numpy as np
import shapely
from cherab.tools.raytransfer import RayTransferCylinder, RayTransferPipeline0D
from raysect.core import MulticoreEngine, translate
from raysect.optical.observer import PowerPipeline0D
from scipy.spatial import ConvexHull

from axuv.cameras import (
    MAX_WAVELENGTHS,
    WAVELENGTH_BIN_EDGES,
    SECTOR_CAMERAS,
    SPECTRAL_BINS,
    create_observable_world,
)
from axuv.interpolation import (
    POLOIDAL_RMAX,
    POLOIDAL_RMIN,
    POLOIDAL_ZMAX,
    POLOIDAL_ZMIN,
)
from axuv.io import load_axuv_df

MAX_BIN_WIDTH = MAX_WAVELENGTHS[-1] / SPECTRAL_BINS


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the AXUV raytransfer sensitivity matrix for one or more AUG sectors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sectors",
        "-s",
        nargs="+",
        default=["S16"],
        choices=list(SECTOR_CAMERAS),
        metavar="SECTOR",
        help=f"Sectors to simulate. Choices: {list(SECTOR_CAMERAS)}.",
    )
    parser.add_argument(
        "--reflections",
        action="store_true",
        default=False,
        help=(
            "Use the full AUG CAD mesh so that reflections are modelled. "
            "Requires access to the AUG CAD files via cad_files.py. "
            "When omitted, a simple toroidal absorbing wall is used."
        ),
    )
    parser.add_argument(
        "--jorek-file",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Optional JOREK HDF5 file.  When supplied, the voxel grid is "
            "trimmed to the convex hull of the JOREK plasma points.  "
            "Expected datasets: R, Z (1-D arrays of radial and vertical positions)."
        ),
    )
    parser.add_argument(
        "--resolution-r",
        type=int,
        default=60,
        metavar="N",
        help="Number of voxel columns in the radial direction.",
    )
    parser.add_argument(
        "--resolution-z",
        type=int,
        default=110,
        metavar="N",
        help="Number of voxel rows in the vertical direction.",
    )
    parser.add_argument(
        "--pixel-samples",
        type=int,
        default=1_000_000,
        metavar="N",
        help="Number of pixel samples per foil per wavelength bin.",
    )
    parser.add_argument(
        "--ray-max-depth",
        type=int,
        default=10,
        metavar="N",
        help="Maximum ray recursion depth (relevant when --reflections is set).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path for the output HDF5 file. "
            "Defaults to raytransfer_<sectors>_[no]refl.h5 in the current directory."
        ),
    )
    parser.add_argument(
        "--observe-processes",
        type=int,
        default=10,
        metavar="proc",
        help="Spawned processes during ray transfer simulation. "
        "Recommended to set to available CPU threads.",
    )
    parser.add_argument(
        "--etendue-mode",
        action="store_true",
        help="Only calculate the etendue of the diodes and save to a separate file.",
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_output_path(sectors: list, reflections: bool, output_arg) -> str:
    """Returns the output HDF5 path, auto-generating a name if none was given."""
    if output_arg is not None:
        return output_arg
    sector_tag = "_".join(sectors)
    refl_tag = "refl" if reflections else "norefl"
    return f"raytransfer_{sector_tag}_{refl_tag}.h5"

def _build_etendue_output_path(sectors: list, output_arg) -> str:
    """Returns the output HDF5 path, auto-generating a name if none was given."""
    if output_arg is not None:
        return output_arg
    sector_tag = "_".join(sectors)
    return f"etendue_{sector_tag}.h5"


def _load_jorek_hull(jorek_file: str) -> np.ndarray:
    """
    Loads R and Z from a JOREK HDF5 file and returns the vertices of their
    convex hull as an (n_vertices, 2) array.
    """
    with h5py.File(jorek_file, "r") as f:
        R = np.asarray(f["R"]).ravel()
        Z = np.asarray(f["Z"]).ravel()
    points = np.column_stack([R, Z])
    hull = ConvexHull(points)
    return points[hull.vertices]


def _full_rectangular_hull() -> np.ndarray:
    """Returns the four corners of the full poloidal grid as the hull polygon."""
    return np.array(
        [
            [POLOIDAL_RMIN, POLOIDAL_ZMIN],
            [POLOIDAL_RMAX, POLOIDAL_ZMIN],
            [POLOIDAL_RMAX, POLOIDAL_ZMAX],
            [POLOIDAL_RMIN, POLOIDAL_ZMAX],
        ]
    )


def _build_voxel_grid(resolution_R: int, resolution_Z: int, hull_points: np.ndarray):
    """
    Builds a RayTransferCylinder voxel grid and a Laplacian smoothing operator.

    The grid is trimmed to the convex-hull polygon of `hull_points`, which is
    expanded by 10 cm so boundary voxels are always included.

    Returns
    -------
    ray_transfer_grid : RayTransferCylinder
    cell_centres      : ndarray of shape (nx, ny, 2)  – (R, z) of each voxel centre
    grid_laplacian    : ndarray of shape (num_cells, num_cells)
    num_cells         : int
    """
    nx = resolution_R
    ny = resolution_Z

    cell_r, cell_dx = np.linspace(POLOIDAL_RMIN, POLOIDAL_RMAX, nx, retstep=True)
    cell_z, cell_dz = np.linspace(POLOIDAL_ZMIN, POLOIDAL_ZMAX, ny, retstep=True)
    cell_r_grid, cell_z_grid = np.broadcast_arrays(cell_r[:, None], cell_z[None, :])
    cell_centres = np.stack((cell_r_grid, cell_z_grid), axis=-1)  # (nx, ny, 2)

    cell_vertices_r = np.linspace(
        cell_r[0] - 0.5 * cell_dx, cell_r[-1] + 0.5 * cell_dx, nx + 1
    )
    cell_vertices_z = np.linspace(
        cell_z[0] - 0.5 * cell_dz, cell_z[-1] + 0.5 * cell_dz, ny + 1
    )

    # Buffer the hull polygon by 10 cm so edge voxels are not excluded
    polygon = shapely.geometry.Polygon(hull_points).buffer(0.1, join_style="mitre")

    grid_mask = np.zeros((nx, ny), dtype=bool)
    for ix in range(nx):
        for iy in range(ny):
            corners = [
                shapely.geometry.Point(cell_vertices_r[ix], cell_vertices_z[iy]),
                shapely.geometry.Point(cell_vertices_r[ix + 1], cell_vertices_z[iy]),
                shapely.geometry.Point(cell_vertices_r[ix], cell_vertices_z[iy + 1]),
                shapely.geometry.Point(
                    cell_vertices_r[ix + 1], cell_vertices_z[iy + 1]
                ),
            ]
            if any(polygon.contains(c) for c in corners):
                grid_mask[ix, iy] = True

    # Add toroidal axis dimension (axisymmetric → n_polar=1)
    grid_mask = grid_mask[:, None, :]
    num_cells = int(grid_mask.sum())

    ray_transfer_grid = RayTransferCylinder(
        radius_outer=cell_vertices_r[-1],
        radius_inner=cell_vertices_r[0],
        height=cell_vertices_z[-1] - cell_vertices_z[0],
        n_radius=nx,
        n_height=ny,
        mask=grid_mask,
        n_polar=1,
        transform=translate(0, 0, cell_vertices_z[0]),
    )

    # ── Laplacian regularisation operator (isotropic smoothing) ──────────────
    voxel_map_with_borders = -np.ones((nx + 2, ny + 2), dtype=int)
    voxel_map_with_borders[1:-1, 1:-1] = ray_transfer_grid.voxel_map[:, 0, :]
    inverted_voxel_map = ray_transfer_grid.invert_voxel_map()
    grid_laplacian = np.zeros((num_cells, num_cells))

    for ith_cell in range(num_cells):
        ix, _, iy = inverted_voxel_map[ith_cell]
        ix, iy = ix[0], iy[0]
        neighbours_2d = (
            [ix, ix, ix, ix + 1, ix + 1, ix + 2, ix + 2, ix + 2],
            [iy, iy + 1, iy + 2, iy, iy + 2, iy, iy + 1, iy + 2],
        )
        neighbours_1d = voxel_map_with_borders[neighbours_2d]
        neighbours_1d = neighbours_1d[neighbours_1d > -1]
        grid_laplacian[ith_cell, neighbours_1d] = -1
        grid_laplacian[ith_cell, ith_cell] = neighbours_1d.size

    return ray_transfer_grid, cell_centres, grid_laplacian, num_cells


def _init_output_file(
    hdf5_path: str,
    num_diodes: int,
    num_cells: int,
    total_wavelength_bins: int,
    cell_centres: np.ndarray,
    ray_transfer_grid: RayTransferCylinder,
    grid_laplacian: np.ndarray,
    wavelengths: np.ndarray,
    energies_eV: np.ndarray,
    diode_names: list,
) -> None:
    """
    Creates the output HDF5 file with all metadata datasets pre-allocated.
    Does nothing if the file already exists (supports resuming).

    :param diode_names: list of detector_id strings, one per diode, in the
                        same order as the sensitivity_matrix row axis.
    """
    if os.path.exists(hdf5_path):
        print(f"Resuming existing output file: {hdf5_path}")
        return

    with h5py.File(hdf5_path, "w") as h5f:
        # Sensitivity matrix is written column-by-column (one bin at a time)
        h5f.create_dataset(
            "sensitivity_matrix",
            shape=(num_diodes, num_cells, total_wavelength_bins),
            dtype="f8",
        )
        h5f.create_dataset("grid_centres", data=cell_centres)
        h5f.create_dataset("voxel_map", data=ray_transfer_grid.voxel_map)
        h5f.create_dataset(
            "inverse_voxel_map", data=ray_transfer_grid.invert_voxel_map()
        )
        h5f.create_dataset("laplacian", data=grid_laplacian)
        h5f.create_dataset("mask", data=ray_transfer_grid.mask)
        # completed_bins tracks progress for resuming: 0 = pending, 1 = done
        h5f.create_dataset(
            "completed_bins",
            shape=(total_wavelength_bins,),
            dtype="i1",
        )
        h5f.create_dataset("wavelength_bin_edges", data=wavelengths)
        h5f.create_dataset("energy_bin_edges_eV", data=energies_eV)
        h5f.create_dataset(
            "diode_names",
            data=np.array([n.encode("utf-8") for n in diode_names]),
        )

    print(f"Created output file: {hdf5_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()

    SECTORS = args.sectors
    USE_CAD_MESH = args.reflections
    PIXEL_SAMPLES = args.pixel_samples
    RAY_MAX_DEPTH = args.ray_max_depth
    RESOLUTION_R = args.resolution_r
    RESOLUTION_Z = args.resolution_z
    OBSERVE_PROCESSES = args.observe_processes
    ETENDUE_MODE = args.etendue_mode
    if not ETENDUE_MODE:
        HDF5_PATH = _build_output_path(SECTORS, USE_CAD_MESH, args.output)
    else:
        HDF5_PATH = _build_etendue_output_path(SECTORS, args.output)

    # Load the AXUV diode geometry data
    axuv_df = load_axuv_df()

    if not ETENDUE_MODE:
        # ── Build world with cameras ─────────────────────────────────────────────
        world, cameras = create_observable_world(
            sectors=SECTORS,
            axuv_df=axuv_df,
            cad_mesh=USE_CAD_MESH,
            show_plots=False,
        )

        # Count diodes dynamically so the matrix size is always correct
        NUM_OF_DIODES = sum(len(cam.foil_detectors) for cam in cameras)
        print(
            f"Sectors: {SECTORS}  |  Total diodes: {NUM_OF_DIODES}  |  "
            f"Reflections: {USE_CAD_MESH}"
        )
        for camera in cameras:
            print(f"  {camera.name}: {len(camera.foil_detectors)} diodes")

        diode_names = [foil.name for camera in cameras for foil in camera.foil_detectors]

        # ── Build voxel grid ─────────────────────────────────────────────────────
        print("Producing the voxel grid...")
        if args.jorek_file is not None:
            print(f"  Masking grid to JOREK plasma hull from: {args.jorek_file}")
            hull_points = _load_jorek_hull(args.jorek_file)
        else:
            print("  No JOREK file supplied — using full rectangular grid.")
            hull_points = _full_rectangular_hull()

        ray_transfer_grid, cell_centres, grid_laplacian, num_cells = _build_voxel_grid(
            RESOLUTION_R, RESOLUTION_Z, hull_points
        )
        print(f"  Active voxels: {num_cells} / {RESOLUTION_R * RESOLUTION_Z}")

        # ── Wavelength grid ──────────────────────────────────────────────────────
        wavelengths = WAVELENGTH_BIN_EDGES
        energies_eV = 1239.8 / wavelengths
        total_wavelength_bins = len(wavelengths) - 1
        print(
            f"  Wavelength bins: {total_wavelength_bins}  "
            f"({wavelengths[0]:.3f}–{wavelengths[-1]:.3f} nm)"
        )

        # ── Initialise output file (no-op if it already exists) ──────────────────
        _init_output_file(
            HDF5_PATH,
            NUM_OF_DIODES,
            num_cells,
            total_wavelength_bins,
            cell_centres,
            ray_transfer_grid,
            grid_laplacian,
            wavelengths,  # these are the wavelength bin edges
            energies_eV,  # these are the energy values corresponding to the bin edges
            diode_names,
        )

        # ── Attach voxel grid to the scene ───────────────────────────────────────
        ray_transfer_grid.parent = world

        sensitivity_matrix = np.zeros([NUM_OF_DIODES, num_cells, total_wavelength_bins])

        # ── Main computation loop ─────────────────────────────────────────────────
        # If no reflections are calculated, there is no need to calculate the sensitivity matrix
        # for every wavelength bin independently, since it will be the same for all bins

        if (
            not USE_CAD_MESH
        ):  # No reflections, calculate sensitivity matrix for all bins at once
            print(
                "\nNo reflections: calculating sensitivity matrix for all wavelength bins at once..."
            )
            diode_index = 0
            for camera in cameras:
                for foil in camera.foil_detectors:
                    print(
                        f"  [{diode_index + 1}/{NUM_OF_DIODES}] {foil.name}",
                        end="\r",
                    )
                    foil.pipelines = [RayTransferPipeline0D(kind=foil.units)]
                    foil.min_wavelength = 400  # as there are no reflections, we can use the visible range, for example
                    foil.max_wavelength = 700
                    foil.spectral_bins = ray_transfer_grid.bins
                    foil.spectral_rays = 1
                    foil.pixel_samples = PIXEL_SAMPLES
                    foil.ray_max_depth = RAY_MAX_DEPTH
                    foil.render_engine = MulticoreEngine(processes=OBSERVE_PROCESSES)
                    foil.observe()
                    # Instead of indexing into the sensitivity matrix, assign to all wavelength bins at once
                    sensitivity_matrix[diode_index, :, :] = foil.pipelines[0].matrix[
                        :, np.newaxis
                    ]
                    diode_index += 1

            # Save the sensitivity matrix for all wavelength bins
            with h5py.File(HDF5_PATH, "r+") as h5f:
                h5f["sensitivity_matrix"][:, :, :] = sensitivity_matrix
                h5f["completed_bins"][:] = 1
                h5f.flush()

        elif USE_CAD_MESH:  # Reflections ON, calculate sensitivity matrix for each wavelength bin independently
            for j in range(total_wavelength_bins):
                with h5py.File(HDF5_PATH, "r") as h5f:
                    if h5f["completed_bins"][j]:
                        print(
                            f"Skipping bin {j + 1}/{total_wavelength_bins} (already done)"
                        )
                        continue

                wl_lo, wl_hi = wavelengths[j], wavelengths[j + 1]
                print(f"\nBin {j + 1}/{total_wavelength_bins}: {wl_lo:.4f}–{wl_hi:.4f} nm")

                # ── Compute sensitivity matrix for this wavelength bin and measure time
                start_time = time.time()

                diode_index = 0
                for camera in cameras:
                    for foil in camera.foil_detectors:
                        print(
                            f"  [{diode_index + 1}/{NUM_OF_DIODES}] {foil.name}",
                            end="\r",
                        )
                        foil.pipelines = [RayTransferPipeline0D(kind=foil.units)]
                        foil.min_wavelength = wl_lo
                        foil.max_wavelength = wl_hi
                        foil.spectral_bins = ray_transfer_grid.bins
                        foil.spectral_rays = 1
                        foil.pixel_samples = PIXEL_SAMPLES
                        foil.ray_max_depth = RAY_MAX_DEPTH
                        foil.render_engine = MulticoreEngine(processes=OBSERVE_PROCESSES)
                        foil.observe()
                        sensitivity_matrix[diode_index, :, j] = foil.pipelines[0].matrix
                        diode_index += 1
                        # This will be overrwritten, but is needed so that Raysect doesn't fail for the next diode with
                        # "ValueError: The minimum wavelength must be less than the maximum wavelength."
                        foil.max_wavelength += MAX_BIN_WIDTH * 2

                # Flush this bin to disk immediately so a restart can resume from here
                with h5py.File(HDF5_PATH, "r+") as h5f:
                    h5f["sensitivity_matrix"][:, :, j] = sensitivity_matrix[:, :, j]
                    h5f["completed_bins"][j] = 1
                    h5f.flush()

                # ── Compute time taken for this bin and expected time remaining
                time_taken_one_bin = time.time() - start_time
                expected_time_remaining = time_taken_one_bin * (
                    total_wavelength_bins - j - 1
                )
                print(f"Time taken for one bin: {time_taken_one_bin:.2f} s")
                print(f"Expected time to complete: {expected_time_remaining / 60:.2f} min")

        print(f"\nDone. Results saved to {HDF5_PATH}")

    elif ETENDUE_MODE:
        print("NOTE: There is no reason to use CAD mesh reflections in ETENDUE mode.") if USE_CAD_MESH else None
        # ── Build world with cameras ─────────────────────────────────────────────
        world, cameras = create_observable_world(
            sectors=SECTORS,
            axuv_df=axuv_df,
            cad_mesh=False,
            show_plots=False,
            etendue_mode=True,
        )

        # Count diodes dynamically so the matrix size is always correct
        NUM_OF_DIODES = sum(len(cam.foil_detectors) for cam in cameras)
        print(
            f"Sectors: {SECTORS}  |  Total diodes: {NUM_OF_DIODES}  |  "
        )
        for camera in cameras:
            print(f"  {camera.name}: {len(camera.foil_detectors)} diodes")

        diode_names = [foil.name for camera in cameras for foil in camera.foil_detectors]

        raytraced_etendue = np.zeros(NUM_OF_DIODES)
        raytraced_error = np.zeros(NUM_OF_DIODES)

        i = 0
        for camera in cameras:
            for foil in camera.foil_detectors:
                foil.render_engine = MulticoreEngine(processes=OBSERVE_PROCESSES)
                foil.pipelines = [PowerPipeline0D(accumulate=False)]
                etendue, error = foil.calculate_etendue(ray_count=100000)
                print(
                    f"  [{i + 1}/{NUM_OF_DIODES}] {foil.name} (etendue={etendue:.2e}, error={error:.2e})",
                    end="\r",
                )
                raytraced_etendue[i] = etendue
                raytraced_error[i] = error
                i += 1

        # Save the etendue and error data with the diode names
        with h5py.File(HDF5_PATH, "w") as h5f:
            h5f["raytraced_etendue"] = raytraced_etendue
            h5f["raytraced_error"] = raytraced_error
            h5f["diode_names"] = diode_names

        print(f"\nEtendue results saved to {HDF5_PATH}")
