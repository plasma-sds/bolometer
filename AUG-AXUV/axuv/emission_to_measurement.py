"""
emission_to_measurement.py
--------------------------
Stage 3 of the AXUV synthetic signal workflow.

Discovers all emission HDF5 files in a user-supplied directory, groups them
by resolution/masking variant, and produces time-evolution plots and CSV data
tables for all four combinations of noise and degraded responsivity:

  · nominal_no_noise   — nominal responsivity, no noise
  · nominal_noise      — nominal responsivity, Gaussian noise (sigma=10%, seed=0)
  · degraded_no_noise  — degraded responsivity, no noise
  · degraded_noise     — degraded responsivity, Gaussian noise (sigma=10%, seed=0)

The noise seed is re-applied before every noisy call so that nominal_noise and
degraded_noise share the same noise realisation, enabling direct comparison.

Output tree
-----------
<project_root>/output/<parent_dir>/<emissions_dir>/<variant>/<combination>/
    diode_evolution.csv   — 2-D array (diodes × timesteps)
    times_ms.csv          — 1-D array of physical times [ms]
    diode_evolution.png   — log-normalised pcolormesh time-evolution plot

Expected filename formats (defined in calculate_emissions_for_raytransfer.py)
------------------------------------------------------------------------------
    emissions_lowres_<time>.h5
    emissions_lowres_masked_<time>.h5
    emissions_highres_<time>.h5
    emissions_highres_masked_<time>.h5

All visualisation helpers live in axuv.plotting.
All responsivity curve helpers live in axuv.responsivity.
All data-loading helpers live in axuv.io.
"""

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from axuv.responsivity import get_weighted_power, DETECTOR_CALIBRATION_AMPER_PER_WATT
from axuv.io import open_emission_data, load_etendue, WTH_THRESHOLDS
from axuv.plotting import set_plt_rcparams


# ── Filename pattern helpers ───────────────────────────────────────────────────
# Mirror the four save-names defined in calculate_emissions_for_raytransfer.py.

# Captures the resolution/masking variant (e.g. "refl" or "norefl", "lowres", "highres_masked").
_VARIANT_RE = re.compile(r"^emissions_((?:refl|norefl)(?:_low|_high)res(?:_masked)?)_")

# Matches any of the four emission filename prefixes defined in
# calculate_emissions_for_raytransfer.py:
#   emissions_lowres_<time>.h5          emissions_lowres_masked_<time>.h5
#   emissions_highres_<time>.h5         emissions_highres_masked_<time>.h5
# Strips the full prefix to leave only the time value string.
_TIME_RE = re.compile(r"^emissions_(?:refl|norefl)(?:_low|_high)res(?:_masked)?_")

# ── Processing constants ───────────────────────────────────────────────────────
_NOISE_SEED: int = 0

_COMBINATIONS: list[tuple[str, bool, bool]] = [
    # (subfolder_name,      noise,  degraded)
    ("nominal_no_noise", False, False),
    ("nominal_noise", True, False),
    ("degraded_no_noise", False, True),
    ("degraded_noise", True, True),
]

def calculate_time_evolution(
    diode_first,
    diode_last,
    filenames,
    etendue,
    noise=False,
    degraded=False
):
    """
    Calculates the time evolution of the synthetic AXUV diode signals.
    The result is returned as a 2D array of shape (num_diodes, num_times), and
    the time array is returned separately.
    The units of the diode signals are W/m^2 line integrated brightness.
    This is the same unit as in the AUG AXUV shotfiles.

    The user may add Gaussian noise with sigma=10% to simulate measurement
    uncertainty.  Either the manufacturer-specified or the degraded spectral
    response function can be applied.

    Parameters
    ----------
    diode_first : int
        Index of the first diode to include (0-based, inclusive).
    diode_last : int
        Index of the last diode to include (0-based, exclusive).
    filenames : list of str
        Full paths to the emission HDF5 files produced by
        calculate_emissions_for_raytransfer.py.  The physical time [s] is
        parsed from each basename after stripping the known prefix
        (emissions_lowres_, emissions_lowres_masked_,
         emissions_highres_, emissions_highres_masked_) and the .h5 suffix.
    noise : bool
        If True, multiply each diode's signal by a fixed per-diode Gaussian
        factor (mean=1, sigma=0.1) to simulate calibration uncertainty.
    degraded : bool
        If True, apply the degraded-diode responsivity curve.

    Returns
    -------
    diode_data_evolution, times
    """
    times = np.array(
        [
            float(_TIME_RE.sub("", os.path.basename(s)).removesuffix(".h5"))
            for s in filenames
        ]
    )
    numofdiodes = diode_last - diode_first
    diode_data_evolution = np.zeros([numofdiodes, len(filenames)])
    noise_array = (
        np.random.normal(1.0, 0.1, numofdiodes) if noise else np.ones(numofdiodes)
    )
    for i, fname in enumerate(filenames):
        emission_dict = open_emission_data(fname)
        diode_measurements = emission_dict["diode_measurements"]
        diode_data = diode_measurements[diode_first:diode_last]
        # First multiply by noise, then divide by detector calibration to get back the detected power value,
        # then divide by the etendue (4 pi is also needed) to arrive at the line integrated brightness
        # that is contained in the AUG AXUV shotfiles
        diode_data_evolution[:, i] = (
            get_weighted_power(diode_data, degraded=degraded) * noise_array * 4 * np.pi / (DETECTOR_CALIBRATION_AMPER_PER_WATT * etendue[diode_first:diode_last])
        )

    return diode_data_evolution, times

# --- Synthetic measurement data plotting ---
def plot_time_evolution(
    diode_first,
    diode_last,
    filenames,
    etendue,
    noise=False,
    degraded=False,
    vmin=None,
    vmax=None,
    remove_offset=False,
    shot=None,
):
    """
    Plots the time evolution of the synthetic measurement AXUV diode signals
    with log normalisation.  Returns the figure, axis, pcolormesh object, and
    the data as a 2-D array (diodes × time).

    The user may add Gaussian noise with sigma=10% to simulate measurement
    uncertainty.  Either the manufacturer-specified or the degraded spectral
    response function can be applied.

    Optionally, the time offset is removed from the time array to start from 0.

    Parameters
    ----------
    diode_first : int
        Index of the first diode to include (0-based, inclusive).
    diode_last : int
        Index of the last diode to include (0-based, exclusive).
    filenames : list of str
        Full paths to the emission HDF5 files produced by
        calculate_emissions_for_raytransfer.py.  The physical time [s] is
        parsed from each basename after stripping the known prefix
        (emissions_lowres_, emissions_lowres_masked_,
         emissions_highres_, emissions_highres_masked_) and the .h5 suffix.
    noise : bool
        If True, multiply each diode's signal by a fixed per-diode Gaussian
        factor (mean=1, sigma=0.1) to simulate calibration uncertainty.
    degraded : bool
        If True, apply the degraded-diode responsivity curve.
    plot : bool
        If True (default) create and return the figure; otherwise return only
        the data array.
    vmin : float
        Lower colour-scale limit for the log-normalised pcolormesh.
    vmax : float
        Upper colour-scale limit for the log-normalised pcolormesh.
    remove_offset : bool
        If True, remove the time offset from the time array to start from 0.
        If False, try to match the experimental time to the simulation time for `shot`
    shot : str
        AUG shotnumber for setting the starting time.

    Returns
    -------
    When plot=True  : fig, ax, pcm, diode_data_evolution
    When plot=False : diode_data_evolution, times

    NOTE: diode_first / diode_last follow the ordering in which cameras and
    their foil_detectors were appended when building the emission file.
    Use HDFView or h5py to inspect an unfamiliar file.
    """
    diode_data_evolution, times = calculate_time_evolution(diode_first, diode_last, filenames, etendue, noise=noise, degraded=degraded)

    if remove_offset:
        times -= times[0]
    elif shot == "40673":
        times -= times[0]
        times += 2.3276
    elif shot == "41007":
        times -= times[0]
        times += 2.3417
    else:
        pass

    vmax = diode_data_evolution.max() if vmax is None else vmax
    vmin = diode_data_evolution.min() if vmin is None else (vmax / 1e4)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pcm = ax.pcolormesh(
        times,
        range(1, diode_last - diode_first + 1),
        diode_data_evolution,
        norm="log",
        vmin=vmin,
        vmax=vmax,
        cmap="inferno",
        shading="nearest",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Diode number")
    plt.colorbar(pcm, ax=ax, label=r"$I$ [W/m²]")
    return fig, ax, pcm, diode_data_evolution, times


if __name__ == "__main__":
    import argparse

    # Set the standardized plotting parameters
    set_plt_rcparams()

    parser = argparse.ArgumentParser(
        description=(
            "Batch-process AXUV emission HDF5 files: produce time-evolution "
            "plots and CSV data for all four noise × degraded combinations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "emissions_dir",
        help=(
            "Directory containing emission HDF5 files produced by "
            "calculate_emissions_for_raytransfer.py "
            "(filenames: emissions_{lowres|highres}[_masked]_<time>.h5)."
        ),
    )
    parser.add_argument(
        "--diode-first",
        type=int,
        default=0,
        metavar="N",
        help="First diode index to include (0-based, inclusive).",
    )
    parser.add_argument(
        "--diode-last",
        type=int,
        default=96,
        metavar="N",
        help="Last diode index to include (0-based, exclusive). Defaults to all diodes.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=1e4,
        metavar="V",
        help="Colour-scale lower limit for the log-normalised plot of the line integrated brightness.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1e8,
        metavar="V",
        help="Colour-scale upper limit for the log-normalised plot of the line integrated brightness.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=2,
        default=WTH_THRESHOLDS,
        metavar=("T1", "T2"),
        help="W_th threshold fractions for vertical-line overlays (e.g. --thresholds 0.9 0.1). "
        "Defaults to WTH_THRESHOLDS from axuv.io. Times are read from output/threshold_times.json.",
    )
    parser.add_argument(
        "--no-thresholds",
        action="store_true",
        help="Disable W_th vertical-line overlays; only the no-line figures are saved.",
    )
    args = parser.parse_args()

    # ── Validate input directory ───────────────────────────────────────────────
    emissions_dir = Path(args.emissions_dir).resolve()
    if not emissions_dir.is_dir():
        raise SystemExit(f"Error: '{emissions_dir}' is not a directory.")

    # ── Discover and group files by resolution/masking variant ────────────────
    all_files = list(emissions_dir.glob("emissions_*.h5"))
    if not all_files:
        raise SystemExit(f"No emission HDF5 files found in '{emissions_dir}'.")

    groups: dict[str, list[Path]] = {}
    for f in all_files:
        m = _VARIANT_RE.match(f.name)
        if m:
            groups.setdefault(m.group(1), []).append(f)

    if not groups:
        raise SystemExit(
            "Files found but none matched the expected naming pattern "
            "(emissions_{refl|norefl}_{lowres|highres}[_masked]_<time>.h5)."
        )

    # Sort each group by ascending physical time
    for files in groups.values():
        files.sort(key=lambda f: float(_TIME_RE.sub("", f.name).removesuffix(".h5")))

    # ── Resolve diode_first and diode_last ─────────────
    diode_first: int = args.diode_first
    diode_last: int = args.diode_last

    # Create two plots if the diode selection is the default (0-96)
    if diode_first == 0 and diode_last == 96:
        two_plots = True
    else:
        two_plots = False


    # ── Build output root ──────────────────────────────────────────────────────
    # <project_root>/output/<parent_dir_name>/<emissions_dir_name>/
    project_root = Path(__file__).parent.parent
    out_root = (
        project_root
        / "output"
        / emissions_dir.parent.parent.name
        / emissions_dir.parent.name
    )

    sector = "S5" if emissions_dir.parent.name == "P45" else "S16"
    shot = str(emissions_dir.parent.parent.name)
    title_prefix = shot + " " + sector
    fname_prefix = shot + "_" + sector + "_"

    thresholds = None if args.no_thresholds else args.thresholds
    _pct_str = None
    t_thresholds = None
    if thresholds is not None:
        _pct_str = "_".join(str(int(thr * 100)) for thr in thresholds) + "pct"
        _thr_json = project_root / "output" / "threshold_times.json"
        if not _thr_json.exists():
            raise SystemExit(
                f"Error: {_thr_json} not found. Run publications/plot_time_traces.py first."
            )
        with open(_thr_json) as _f:
            _threshold_times = json.load(_f)
        shot_entry = _threshold_times.get(shot, {})
        t_thresholds = [shot_entry.get(str(thr), {}).get("sim") for thr in thresholds]
        if any(t is None for t in t_thresholds):
            print(f"Warning: not all threshold times found for shot {shot}; skipping vertical lines.")
            t_thresholds = None

    raytraced_etendue, _, diode_names = load_etendue(sector)
    total_files = sum(len(v) for v in groups.values())
    print(
        f"\nProcessing {total_files} file(s) across {len(groups)} variant group(s).\n"
    )

    # ── Process each variant group ─────────────────────────────────────────────
    for variant, files in groups.items():
        filenames = [str(f) for f in files]

        # Extract times [s] independently so they can be saved.
        # This mirrors the logic inside plot_time_evolution.
        times = np.array(
            [float(_TIME_RE.sub("", f.name).removesuffix(".h5")) for f in files]
        )

        out_base = out_root / variant
        print(f"Variant '{variant}' — {len(files)} timestep(s)")
        print(f"  Output: {out_base}")

        for subfolder, noise, degraded in _COMBINATIONS:
            out_dir = out_base / subfolder
            out_dir.mkdir(parents=True, exist_ok=True)

            # Re-seed before every noisy call so that nominal_noise and
            # degraded_noise use the same noise realisation.
            if noise:
                np.random.seed(_NOISE_SEED)

            if not two_plots:
                fig, ax, pcm, data, times = plot_time_evolution(
                    diode_first,
                    diode_last,
                    filenames,
                    raytraced_etendue,
                    noise=noise,
                    degraded=degraded,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    shot=shot,
                )

                fig.savefig(out_dir / "diode_evolution.png", bbox_inches="tight")
                plt.close(fig)

            else:

                fig1, ax1, pcm1, data1, times1 = plot_time_evolution(
                    0,
                    48,
                    filenames,
                    raytraced_etendue,
                    noise=noise,
                    degraded=degraded,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    shot=shot,
                )
                fig2, ax2, pcm2, data2, times2 = plot_time_evolution(
                    48,
                    96,
                    filenames,
                    raytraced_etendue,
                    noise=noise,
                    degraded=degraded,
                    vmin=args.vmin,
                    vmax=args.vmax,
                    shot=shot,
                )
                ax1.set_facecolor("k")
                ax2.set_facecolor("k")

                fig1.savefig(out_dir / str(fname_prefix + "horiz_notitle.png"), dpi=300, bbox_inches="tight")
                fig2.savefig(out_dir / str(fname_prefix + "vert_notitle.png"), dpi=300, bbox_inches="tight")

                if t_thresholds is not None and _pct_str is not None:
                    firstcolor = "cyan"
                    secondcolor = "lime"
                    labelsize = 16
                    for cur_ax in (ax1, ax2):
                        cur_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
                        for t, color in zip(t_thresholds, [firstcolor, secondcolor]):
                            cur_ax.axvline(t, color=color)
                        ax_top = cur_ax.twiny()
                        ax_top.set_xlim(cur_ax.get_xlim())
                        ax_top.set_xticks(t_thresholds)
                        ax_top.set_xticklabels([rf"$t_{{{thr}W}}$" for thr in thresholds])
                        ax_top.tick_params(direction='out', length=5, colors='black', labelsize=labelsize)
                        ax_top.spines['top'].set_visible(False)

                    fig1.savefig(out_dir / str(fname_prefix + f"horiz_{_pct_str}.png"), dpi=300, bbox_inches="tight")
                    fig2.savefig(out_dir / str(fname_prefix + f"vert_{_pct_str}.png"), dpi=300, bbox_inches="tight")

                plt.close(fig1)
                plt.close(fig2)

                data, times = calculate_time_evolution(
                    0,
                    96,
                    filenames,
                    raytraced_etendue,
                    noise=noise,
                    degraded=degraded
                )

            # Save the line integrated brightness evolution and times in one csv. First column is time in ms,
            # each subsequent column is the line integrated brightness of a diode at that time point
            # Time data is concatenated with diode evolution data to form a single csv
            np.savetxt(out_dir / "brightness_evolution.csv", np.column_stack((times, data.T)), delimiter=",", header="Time [s]," + ",".join(diode_names))


            print(f"    ✓ {subfolder}")

        print()

    print("Done.")
