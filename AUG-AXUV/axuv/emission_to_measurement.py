"""
emission_to_measurement.py
--------------------------
Stage 3 of the AXUV synthetic signal workflow.

After the geometry matrix (raytransfer_sensitivity.py) and the per-voxel
spectral emissions (calculate_emissions_for_raytransfer.py) have been
computed, this script folds the diode spectral power through the AXUV
spectral responsivity curve to produce a single scalar signal per diode.

Functions in this module:
  time_evolution   — compute and plot diode signals across multiple timesteps

All responsivity curve helpers live in axuv.responsivity.
All visualisation helpers live in axuv.plotting.
All data-loading helpers live in axuv.io.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from axuv.io import SAVEDIR, load_raytransfer_data, open_emission_data
from axuv.responsivity import get_weighted_power

plt.rcParams.update({"font.size": 16, "figure.dpi": 150,
                     "figure.constrained_layout.use": True})


def time_evolution(diode_first, diode_last, times, measured_times,
                   emissions_dir=None, noise=False, degraded=False):
    """
    Computes and plots the time evolution of integrated diode signals across
    multiple JOREK timesteps.

    Parameters
    ----------
    diode_first : int
        Index of the first diode to include (0-based, inclusive).
    diode_last : int
        Index of the last diode to include (0-based, exclusive).
    times : array-like
        Physical times in ms corresponding to each entry in measured_times.
    measured_times : list of str
        Timestep identifiers (e.g. ["02410", "04090"]).
    emissions_dir : str, optional
        Directory containing the emission HDF5 files.  Defaults to SAVEDIR.
    noise : bool
        If True, multiply each diode's signal by a Gaussian noise factor
        (mean=1, sigma=0.1) to simulate measurement uncertainty.
    degraded : bool
        If True, apply the degraded-diode responsivity curve.

    Returns
    -------
    fig, ax, pcm, diode_data_evolution
    """
    if emissions_dir is None:
        emissions_dir = SAVEDIR

    numofdiodes          = diode_last - diode_first
    diode_data_evolution = np.zeros([numofdiodes, len(measured_times)])

    noise_array = np.random.normal(1.0, 0.1, numofdiodes) if noise else np.ones(numofdiodes)

    for i, timestep in enumerate(measured_times):
        fpath = os.path.join(
            emissions_dir, f"raytransfer_emissions_lowres_1eV_{timestep}.h5"
        )
        data               = open_emission_data(fpath)
        energies           = data["energies"]
        diode_measurements = data["diode_measurements"]
        diode_data         = diode_measurements[diode_first:diode_last]

        diode_data_evolution[:, i] = get_weighted_power(
            diode_data, energies, degraded=degraded
        )
        diode_data_evolution[:, i] *= noise_array

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pcm = ax.pcolormesh(
        times, range(numofdiodes), diode_data_evolution,
        norm="log", vmin=1e-5, cmap="inferno",
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Diode index")
    ax.set_title("Diode signal time evolution")
    plt.colorbar(pcm, ax=ax, label="Weighted power [W]")

    return fig, ax, pcm, diode_data_evolution


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply AXUV responsivity to compute synthetic diode signals."
    )
    parser.add_argument(
        "sensitivity_file",
        help="Path to the raytransfer HDF5 file (from raytransfer_sensitivity.py)",
    )
    parser.add_argument(
        "emissions_file",
        help="Path to the emissions HDF5 file (from calculate_emissions_for_raytransfer.py)",
    )
    parser.add_argument(
        "--degraded", action="store_true", default=False,
        help="Use the degraded-diode responsivity curve.",
    )
    args = parser.parse_args()

    # ── Load geometry matrix ─────────────────────────────────────────────────
    rt_data            = load_raytransfer_data(args.sensitivity_file)
    sensitivity_matrix = rt_data["sensitivity_matrix"]
    grid_centres       = rt_data["grid_centres"]
    inverse_voxel_map  = rt_data["inverse_voxel_map"]
    diode_names        = rt_data["diode_names"]
    wavelength_edges   = rt_data["wavelength_bin_edges"]
    energy_edges       = rt_data["energy_bin_edges_eV"]

    # ── Load emissions ────────────────────────────────────────────────────────
    em_data            = open_emission_data(args.emissions_file)
    emissions          = em_data["emissions"]
    energies           = em_data["energies"]
    diode_measurements = em_data["diode_measurements"]

    print(f"Sensitivity matrix : {sensitivity_matrix.shape}")
    print(f"Emissions          : {emissions.shape}")
    print(f"Diode measurements : {diode_measurements.shape}")
    print(f"Diode names        : {diode_names[:5]} ...")

    # ── Weighted power per diode ──────────────────────────────────────────────
    weighted = get_weighted_power(diode_measurements, energies, degraded=args.degraded)
    print(f"\nWeighted power range: {weighted.min():.3e} – {weighted.max():.3e} W")