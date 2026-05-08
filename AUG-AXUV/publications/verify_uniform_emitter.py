"""
Uniform-emitter consistency verification for the norefl sensitivity matrices.

For each sector (S5, S16) this script:
  1. Recreates the exact raytransfer scene: cameras + absorbing rectangular vessel
     + ToroidalVoxelGrid matching the masked voxel grid from the HDF5 file.
  2. Assigns a spatially and spectrally uniform emissivity to every active voxel.
  3. Observes each diode with a PowerPipeline0D (instead of RayTransferPipeline0D).
  4. Compares the measured geometry factor G_obs = P / (ε₀ × bandwidth) against
     G_matrix = sensitivity_matrix[:, :, 0].sum(axis=1) (row sums).

Both quantities are restricted to the same voxel-grid region (R ≤ 2.2 m), so the
comparison is an internal consistency check of the sensitivity matrix.  Both G_obs and
G_matrix will be smaller than an etendue-based prediction because the diodes' view cones
extend to the absorbing wall at R = 2.5 m — this is expected and not an error.

Runtime note: increase PIXEL_SAMPLES for higher accuracy at proportional cost.
"""

import argparse
import matplotlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from raysect.core import MulticoreEngine
from raysect.optical.material.emitter.homogeneous import HomogeneousVolumeEmitter
from raysect.optical.observer import PowerPipeline0D
from raysect.core.math import Point2D
from cherab.tools.inversions import ToroidalVoxelGrid

from axuv.cameras import create_observable_world
from axuv.io import load_axuv_df, PROJECT_ROOT
from axuv.plotting import set_plt_rcparams

matplotlib.use("Agg")
set_plt_rcparams()

# ── Configuration ─────────────────────────────────────────────────────────────
EPSILON0      = 1.0      # W/m³/sr/nm — spectral emissivity of the uniform emitter
MIN_WL        = 1.0     # nm
MAX_WL        = 1240.0   # nm
BANDWIDTH     = MAX_WL - MIN_WL        # 1239.75 nm
PIXEL_SAMPLES = 100_000
N_PROCESSES   = 10
DATA_DIR      = PROJECT_ROOT / "axuv" / "data"
OUTPUT_DIR    = PROJECT_ROOT / "output"


# ── Uniform emitter material ──────────────────────────────────────────────────
class UniformVoxelEmitter(HomogeneousVolumeEmitter):
    """Spatially and spectrally flat volume emitter: ε₀ [W/m³/sr/nm] everywhere."""

    def __init__(self, epsilon0):
        super().__init__()
        self.epsilon0 = epsilon0

    def emission_function(self, direction, spectrum, world, ray,
                          primitive, world_to_primitive, primitive_to_world):
        spectrum.samples[:] += self.epsilon0
        return spectrum


# ── Helpers ───────────────────────────────────────────────────────────────────
def _decode(name):
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def _build_voxel_coords(grid_centres, voxel_map_3d):
    """
    Reconstruct ToroidalVoxelGrid coordinate list from HDF5 metadata.

    Returns a list of length n_cells where element j is a list of four
    Point2D vertices (r-z plane) for the voxel with sensitivity-matrix index j.
    """
    voxel_map = voxel_map_3d[:, 0, :]          # (nx, ny)
    nx, ny    = voxel_map.shape
    n_cells   = int((voxel_map >= 0).sum())
    dr = abs(float(grid_centres[1, 0, 0]) - float(grid_centres[0, 0, 0]))
    dz = abs(float(grid_centres[0, 1, 1]) - float(grid_centres[0, 0, 1]))

    voxel_coords = [None] * n_cells
    for ix in range(nx):
        for iy in range(ny):
            j = int(voxel_map[ix, iy])
            if j >= 0:
                R = float(grid_centres[ix, iy, 0])
                Z = float(grid_centres[ix, iy, 1])
                voxel_coords[j] = [
                    Point2D(R - dr / 2, Z - dz / 2),
                    Point2D(R + dr / 2, Z - dz / 2),
                    Point2D(R + dr / 2, Z + dz / 2),
                    Point2D(R - dr / 2, Z + dz / 2),
                ]
    return voxel_coords


def _verify_sector(sector, axuv_df, uniform_mat):
    """Run the uniform-emitter observation for one sector and return results."""
    rt_path = DATA_DIR / f"raytransfer_{sector}_norefl.h5"
    print(f"\n{'='*60}")
    print(f"Sector {sector}  —  loading {rt_path.name}")

    with h5py.File(rt_path, "r") as h5f:
        grid_centres  = h5f["grid_centres"][()]
        voxel_map_3d  = h5f["voxel_map"][()]
        sm_bin0       = h5f["sensitivity_matrix"][:, :, 0]   # (n_diodes, n_cells)
        rt_names      = [_decode(n) for n in h5f["diode_names"][()]]

    voxel_coords = _build_voxel_coords(grid_centres, voxel_map_3d)
    n_cells = len(voxel_coords)
    print(f"  Voxel grid: {n_cells} active cells")

    # Build Raysect scene
    world, cameras = create_observable_world(
        sectors=[sector], axuv_df=axuv_df,
        cad_mesh=False, etendue_mode=False,   # adds absorbing rectangular vessel wall
    )

    # Attach ToroidalVoxelGrid with uniform emitter on every voxel
    tvg = ToroidalVoxelGrid(
        voxel_coords, parent=world, active="all", primitive_type="csg"
    )
    for i in range(tvg.count):
        tvg[i].material = uniform_mat
    print(f"  TVG attached: {tvg.count} voxels")

    # Observe each foil
    obs_names, powers = [], []
    n_diodes = sum(len(cam.foil_detectors) for cam in cameras)
    diode_idx = 0
    for camera in cameras:
        for foil in camera.foil_detectors:
            diode_idx += 1
            print(f"  [{diode_idx}/{n_diodes}] {foil.name}", end="\r")
            pipeline = PowerPipeline0D(accumulate=False)
            foil.pipelines    = [pipeline]
            foil.min_wavelength = MIN_WL
            foil.max_wavelength = MAX_WL
            foil.spectral_bins  = 1
            foil.spectral_rays  = 1
            foil.pixel_samples  = PIXEL_SAMPLES
            foil.render_engine  = MulticoreEngine(processes=N_PROCESSES)
            foil.observe()
            obs_names.append(foil.name)
            powers.append(pipeline.value.mean)

    # Reorder to match HDF5 diode order
    obs_lookup = dict(zip(obs_names, powers))
    obs_power  = np.array([obs_lookup[n] for n in rt_names])

    G_obs    = obs_power / (EPSILON0 * BANDWIDTH)   # [m³·sr]
    G_matrix = sm_bin0.sum(axis=1)                  # [m³·sr]
    valid    = G_matrix > 0
    rel_err  = np.where(valid, (G_obs - G_matrix) / np.where(valid, G_matrix, 1.0), np.nan)

    n_valid = valid.sum()
    print(f"\n  Active diodes (G_matrix > 0): {n_valid}/{len(G_matrix)}")
    print(f"  Mean |rel_err| = {np.nanmean(np.abs(rel_err))*100:.2f} %  "
          f"  max |rel_err| = {np.nanmax(np.abs(rel_err))*100:.2f} %")

    return {
        "diode_names": rt_names,
        "G_obs":       G_obs,
        "G_matrix":    G_matrix,
        "rel_err":     rel_err,
    }


def _plot_and_save(results, sector):
    """Save parity plot and relative-error bar chart for one sector."""
    G_obs    = results["G_obs"]
    G_matrix = results["G_matrix"]
    rel_err  = results["rel_err"]
    n        = len(G_obs)

    # Parity plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(G_matrix, G_obs, s=15, color="k")
    lim = max(G_matrix.max(), G_obs.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y = x")
    ax.set_xlabel(r"$G_\mathrm{matrix}$ (m$^3$ sr)")
    ax.set_ylabel(r"$G_\mathrm{obs}$ (m$^3$ sr)")
    ax.set_title(f"Sector {sector} — parity")
    ax.legend(fontsize=12)
    for ext in ("png", "eps"):
        fig.savefig(OUTPUT_DIR / f"verify_uniform_{sector}_parity.{ext}",
                    format=ext if ext == "eps" else None, bbox_inches="tight")
    plt.close(fig)

    # Relative-error bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(n), rel_err * 100, color="steelblue", width=0.8)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Diode index")
    ax.set_ylabel(r"$(G_\mathrm{obs} - G_\mathrm{matrix})\,/\,G_\mathrm{matrix}$ (\%)")
    ax.set_title(f"Sector {sector} — relative error")
    ax.set_xlim(-1, n)
    for ext in ("png", "eps"):
        fig.savefig(OUTPUT_DIR / f"verify_uniform_{sector}_rel_err.{ext}",
                    format=ext if ext == "eps" else None, bbox_inches="tight")
    plt.close(fig)


# ── Plot from saved HDF5 ──────────────────────────────────────────────────────
def plot_from_hdf5(sector):
    """Load results from a previously saved HDF5 and regenerate the figures."""
    h5_path = OUTPUT_DIR / f"verify_uniform_{sector}.h5"
    with h5py.File(h5_path, "r") as h5f:
        results = {
            "diode_names": [_decode(n) for n in h5f["diode_names"][()]],
            "G_obs":       h5f["G_obs"][()],
            "G_matrix":    h5f["G_matrix"][()],
            "rel_err":     h5f["rel_err"][()],
        }
    _plot_and_save(results, sector)
    print(f"Saved figures for sector {sector}")


# ── Main ──────────────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Uniform-emitter consistency check for norefl sensitivity matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sectors", nargs="+", default=["S5", "S16"],
        choices=["S5", "S16"], metavar="SECTOR",
    )
    parser.add_argument("--pixel-samples", type=int, default=PIXEL_SAMPLES)
    parser.add_argument("--processes", type=int, default=N_PROCESSES)
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip ray tracing; regenerate figures from existing HDF5 output files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        for sector in args.sectors:
            plot_from_hdf5(sector)
    else:
        axuv_df     = load_axuv_df()
        uniform_mat = UniformVoxelEmitter(EPSILON0)

        for sector in args.sectors:
            results = _verify_sector(sector, axuv_df, uniform_mat)

            out_path = OUTPUT_DIR / f"verify_uniform_{sector}.h5"
            with h5py.File(out_path, "w") as h5f:
                h5f.create_dataset("diode_names",
                                   data=np.array([n.encode() for n in results["diode_names"]]))
                h5f.create_dataset("G_obs",    data=results["G_obs"])
                h5f.create_dataset("G_matrix", data=results["G_matrix"])
                h5f.create_dataset("rel_err",  data=results["rel_err"])
                h5f.attrs["epsilon0_W_m3_sr_nm"] = EPSILON0
                h5f.attrs["bandwidth_nm"]        = BANDWIDTH
                h5f.attrs["pixel_samples"]       = args.pixel_samples
            print(f"  Saved results to {out_path}")

            _plot_and_save(results, sector)
            print(f"  Saved figures for sector {sector}")
