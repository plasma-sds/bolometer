import os
import pickle
from pathlib import Path

import h5py
import pandas as pd

DATADIR_ENV = os.environ.get("datadir")
if DATADIR_ENV is None:
    DATADIR_ENV = str(Path(__file__).parent) + "/data"
DATADIR = str(DATADIR_ENV) + "/"

CURRENTDIR = DATADIR + "/" + str(os.environ.get("currentdir")) + "/"
SAVEDIR = CURRENTDIR + "output/"

RAYTRANSFER_PATH = DATADIR + "raytransfer_S05_lowres.h5"

# Loading diode geometry data into a pandas DataFrame for easier filtering
AXUV_DATAFILE = DATADIR + "AXUV_LOS_geom.txt"


def load_axuv_df() -> pd.DataFrame:
    """Load and return the AXUV diode geometry DataFrame."""
    try:
        df = pd.read_csv(AXUV_DATAFILE, sep=r"\s+", engine="python").drop(
            columns=["act", "con", "F", "Foil_ID", "R_Kabel", "U_Gen.", "Faktor"]
        )
        print(f"Loaded AXUV geometry datafile '{AXUV_DATAFILE}' with shape {df.shape}")
        return df

    except Exception as e:
        raise RuntimeError(
            f"Could not load AXUV geometry datafile '{AXUV_DATAFILE}': {e}"
        ) from e


# Plasma-facing component contours for poloidal-plane plots
# Using Path(__file__).parent ensures this works regardless of CWD
_GC_LINES_PATH = Path(__file__).parent / "data" / "gc_d_lines.obj"
with open(_GC_LINES_PATH, "rb") as fp:
    gc_d_lines = pickle.load(fp)


# ── HDF5 data-loading helpers ─────────────────────────────────────────────────


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
