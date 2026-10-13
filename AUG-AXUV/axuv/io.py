import os
import pickle
from functools import cache
from pathlib import Path

import h5py
import pandas as pd
import numpy as np

from axuv.datafiles import data_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Single source of truth for the W_th fractions at which vertical lines are drawn.
# Consumed by publications/plot_time_traces.py, axuv/emission_to_measurement.py and
# axuv_measurement/plotting_from_shotfiles.py.
WTH_THRESHOLDS = [0.9, 0.1]

# Root of everything the workflow writes (Stage 1-3 results, figures, CSVs).
# Kept apart from the data directory, which only holds inputs.
OUTPUT_DIR = Path(os.environ.get("AXUV_OUTPUT_DIR") or PROJECT_ROOT / "output").resolve()


def emissions_output_dir(dataset: str) -> Path:
    """Return the Stage 2 output directory of *dataset* (e.g. '40673/P1')."""
    return OUTPUT_DIR / dataset / "emissions"


# Diode and aperture geometry, one row per diode (data key, see axuv.datafiles)
AXUV_GEOMETRY_KEY = "geometry/AXUV_LOS_geom.txt"


def load_axuv_df() -> pd.DataFrame:
    """Load and return the AXUV diode geometry DataFrame."""
    path = data_path(AXUV_GEOMETRY_KEY)
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python").drop(
            columns=["act", "con", "F", "Foil_ID", "R_Kabel", "U_Gen.", "Faktor"]
        )
        print(f"Loaded AXUV geometry datafile '{path}' with shape {df.shape}")
        return df

    except Exception as e:
        raise RuntimeError(
            f"Could not load AXUV geometry datafile '{path}': {e}"
        ) from e


@cache
def load_gc_d_lines() -> list:
    """Return the plasma-facing component contours for poloidal-plane plots.

    Loaded on first call and cached; each item is an (R, z) pair of arrays.
    """
    with open(data_path("geometry/gc_d_lines.obj"), "rb") as fp:
        return pickle.load(fp)


# ── HDF5 data-loading helpers ─────────────────────────────────────────────────
def load_etendue(sector: str):
    etendue_path = data_path(f"geometry/etendue_{sector}.h5")
    with h5py.File(etendue_path, "r") as h5f:
        raytraced_etendue = np.asarray(h5f["raytraced_etendue"])
        raytraced_error = np.asarray(h5f["raytraced_error"])
        diode_names = np.asarray(h5f["diode_names"], dtype=str)
    print(f"Loaded raytraced etendue from {etendue_path}")
    return raytraced_etendue, raytraced_error, diode_names


def load_raytransfer_data(hdf5_path: str) -> dict:
    """
    Loads the sensitivity matrix and grid metadata from the HDF5 file
    produced by raytransfer_sensitivity.py.

    Returns a dict with keys:
        sensitivity_matrix   – ndarray (num_diodes, num_cells, num_wl_bins)
        grid_centres         – ndarray (nx, ny, 2) of (R, z) voxel centres
        laplacian            – ndarray (num_cells, num_cells)
        voxel_map            – voxel index map array
        inverse_voxel_map    – inverse mapping from voxel index to grid coords
        wavelength_bin_edges – 1-D array of wavelength bin edges [nm]
        energy_bin_edges_eV  – 1-D array of photon energy bin edges [eV]
        diode_names          – list of str, one detector ID per diode row
    """
    with h5py.File(hdf5_path, "r") as h5f:
        data = {
            "sensitivity_matrix": h5f["sensitivity_matrix"][()],
            "grid_centres": h5f["grid_centres"][()],
            "laplacian": h5f["laplacian"][()],
            "voxel_map": h5f["voxel_map"][()],
            "inverse_voxel_map": h5f["inverse_voxel_map"][()],
            "wavelength_bin_edges": h5f["wavelength_bin_edges"][()],
            "energy_bin_edges_eV": h5f["energy_bin_edges_eV"][()],
        }
        if "diode_names" in h5f:
            raw = h5f["diode_names"][()]
            data["diode_names"] = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in raw
            ]
        else:
            data["diode_names"] = []

    print(
        f"Loaded sensitivity matrix {data['sensitivity_matrix'].shape} from {hdf5_path}"
    )
    return data


def open_emission_data(filepath: str) -> dict:
    """
    Loads spectral emission data produced by
    calculate_emissions_for_raytransfer.py.

    Returns a dict with keys:
        emissions           – ndarray (num_voxels, num_wl_bins)
        wavelengths         – 1-D array of wavelength bin centres [nm]
        energies            – 1-D array of photon energy bin centres [eV]
        diode_measurements  – ndarray (num_diodes, num_wl_bins)
    """
    with h5py.File(filepath, "r") as f:
        data = {
            "emissions": f["emissions"][()],
            "wavelengths": f["wavelengths"][()],
            "energies": f["energies"][()],
            "diode_measurements": f["diode_measurements"][()],
        }
    return data
