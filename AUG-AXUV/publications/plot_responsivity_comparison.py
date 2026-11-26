"""Measured vs. synthetic AXUV brightness, nominal and degraded responsivity side by side.

One figure per shot and camera, with three panels in a row: the measured
signal, the synthetic signal with the nominal responsivity and the synthetic
signal with the degraded responsivity. All panels of all figures share one log
colour scale, its upper limit is the maximum over every panel of every figure
and the lower limit is four decades below.

Inputs:
  - measured: AUG shotfiles XVU / XVR read with aug_sfutils, smoothed as in
    axuv_measurement/plotting_from_shotfiles.py. Set SF_BASEPATH to read local
    copies of the shotfiles.
  - synthetic: brightness_evolution.csv written by axuv/emission_to_measurement.py,
    <synthetic-dir>/<shot>/<P1|P45>/<reflection>/<nominal|degraded>_no_noise/.
  - W_th threshold times: <synthetic-dir>/threshold_times.json.

Run with:
    uv run python publications/plot_responsivity_comparison.py \
        --synthetic-dir ~/Data/AUG-AXUV/output-responsivity-fix
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

from axuv.io import OUTPUT_DIR, WTH_THRESHOLDS
from axuv.plotting import FONT_SCALE, add_direction_arrow, save_png_pdf, set_plt_rcparams
from axuv_measurement.config import D16, DHC, DHT, DVC
from axuv_measurement.io import get_AXUV_signals

# Start and length of the plotted window [s], same as plotting_from_shotfiles.py.
# The synthetic times are shifted so that the first time step falls on the start.
SHOT_WINDOW = {
    40673: (2.3276, 1.9e-3),
    41007: (2.3417, 5.2e-3),
}
# sector -> (injection position folder, vertical camera, horizontal camera)
SECTORS = {16: ("P1", D16, DHT), 5: ("P45", DVC, DHC)}
# In brightness_evolution.csv, diodes 1-48 are the horizontal camera, 49-96 the vertical one.
CSV_COLUMNS = {"horizontal": slice(0, 48), "vertical": slice(48, 96)}
PANEL_TITLES = ("Measured", "Synthetic, nominal", "Synthetic, degraded")
LOG_DECADES = 4
THRESHOLD_COLORS = ("cyan", "lime")


def read_synthetic(path, orientation, start):
    """Returns (times [s], data [diodes x times]) of one camera from brightness_evolution.csv."""
    table = np.loadtxt(path, delimiter=",", comments="#")
    times = table[:, 0] - table[0, 0] + start
    data = table[:, 1:][:, CSV_COLUMNS[orientation]].T
    return times, data


def draw_panel(ax, times, data, norm, title, threshold_times):
    ax.set_facecolor("k")
    pcm = ax.pcolormesh(times, np.arange(1, data.shape[0] + 1), data, norm=norm, cmap="inferno", shading="nearest")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if threshold_times is not None:
        for t, color in zip(threshold_times, THRESHOLD_COLORS):
            ax.axvline(t, color=color)
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(threshold_times)
        ax_top.set_xticklabels([rf"$t_{{{thr}W}}$" for thr in WTH_THRESHOLDS])
        ax_top.tick_params(direction="out", length=5)
        ax_top.spines["top"].set_visible(False)
    return pcm


def load_panels(shot, sector, camera, orientation, synthetic_dir, reflection):
    """Returns [(times, data)] of the measured, nominal and degraded panels of one camera."""
    start, duration = SHOT_WINDOW[shot]
    run_dir = synthetic_dir / str(shot) / SECTORS[sector][0] / reflection

    measured, measured_times = get_AXUV_signals(
        shot=shot, camera=camera, tbeg=start, tend=start + duration, gaussian_sigma=10
    )
    panels = [(measured_times, measured)]
    for variant in ("nominal_no_noise", "degraded_no_noise"):
        panels.append(read_synthetic(run_dir / variant / "brightness_evolution.csv", orientation, start))
    return panels


def plot_comparison(shot, panels, orientation, norm, thresholds):
    shot_thresholds = thresholds.get(str(shot), {})
    exp_times = [shot_thresholds.get(str(thr), {}).get("exp") for thr in WTH_THRESHOLDS]
    sim_times = [shot_thresholds.get(str(thr), {}).get("sim") for thr in WTH_THRESHOLDS]
    exp_times = None if None in exp_times else exp_times
    sim_times = None if None in sim_times else sim_times

    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5), sharey=True)
    for ax, (times, data), title, t_thr in zip(axes, panels, PANEL_TITLES, (exp_times, sim_times, sim_times)):
        pcm = draw_panel(ax, times, data, norm, title, t_thr)
    axes[0].set_ylabel("Diode number")
    fig.colorbar(pcm, ax=axes, label=r"$I$ [W/m²]", extend="min", pad=0.01)
    add_direction_arrow(axes[0], orientation)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic-dir", type=Path, default=OUTPUT_DIR,
                        help="Root of the synthetic outputs (default: $AXUV_OUTPUT_DIR).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output folder (default: <synthetic-dir>/comparison).")
    parser.add_argument("--reflection", default="refl_lowres_masked",
                        help="Synthetic run folder, e.g. refl_lowres_masked or norefl_lowres_masked.")
    parser.add_argument("--shots", type=int, nargs="+", default=list(SHOT_WINDOW))
    args = parser.parse_args()

    synthetic_dir = args.synthetic_dir.expanduser().resolve()
    out_dir = (args.out_dir or synthetic_dir / "comparison").expanduser()
    with open(synthetic_dir / "threshold_times.json") as f:
        thresholds = json.load(f)

    cases = []
    for shot in args.shots:
        for sector, (_, vert_camera, horiz_camera) in SECTORS.items():
            for camera, orientation in ((vert_camera, "vertical"), (horiz_camera, "horizontal")):
                panels = load_panels(shot, sector, camera, orientation, synthetic_dir, args.reflection)
                cases.append((shot, camera, orientation, panels))
                peaks = ", ".join(f"{np.nanmax(d):.2g}" for _, d in panels)
                print(f"{shot} {camera}: peaks [W/m²] measured, nominal, degraded: {peaks}")

    # One colour scale for every figure, so all panels of all figures can be compared directly
    vmax = max(float(np.nanmax(data)) for *_, panels in cases for _, data in panels)
    norm = LogNorm(vmin=vmax / 10**LOG_DECADES, vmax=vmax)
    print(f"Common colour scale: {norm.vmin:.2g} to {norm.vmax:.2g} W/m²")

    set_plt_rcparams(scale=FONT_SCALE)
    for shot, camera, orientation, panels in cases:
        fig = plot_comparison(shot, panels, orientation, norm, thresholds)
        save_png_pdf(fig, out_dir / f"{shot}_{camera}_{args.reflection}_measured_nominal_degraded")
        plt.close(fig)


if __name__ == "__main__":
    main()
