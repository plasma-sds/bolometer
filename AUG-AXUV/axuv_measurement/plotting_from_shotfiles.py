"""
IPython Notebook for plotting AXUV signals from shotfiles
"""
# %%
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from axuv_measurement.config import (
    D16, DHT, DVC, DHC
)
from axuv.io import OUTPUT_DIR, WTH_THRESHOLDS as THRESHOLDS
from axuv.plotting import (
    DEFAULT_LIMITS,
    FONT_SCALE,
    SCALES,
    add_direction_arrow,
    plot_diode_evolution,
    save_png_pdf,
    scale_suffix,
)
from axuv_measurement.io import get_AXUV_signals
from axuv_measurement.plotting import set_plt_rcparams


set_plt_rcparams(scale=FONT_SCALE)

# THRESHOLDS is defined centrally in axuv.io (WTH_THRESHOLDS).
# The corresponding crossing times are read from $AXUV_OUTPUT_DIR/threshold_times.json.
# Set PLOT_THRESHOLD_LINES = False to skip the W_th vertical-line figures
# (the no-line figures are always saved regardless).
PLOT_THRESHOLD_LINES = True
# Colour scales to save, "log" and/or "linear". Linear figures get a "_lin"
# filename suffix, the default colour limits per scale are DEFAULT_LIMITS.
PLOT_SCALES = SCALES
_pct_str = "_".join(str(int(thr * 100)) for thr in THRESHOLDS) + "pct"

SHOT_CONFIG = {
    40673: {"starttime": 2.3276, "duration": 1.9e-3},
    41007: {"starttime": 2.3417, "duration": 5.2e-3},
}

_threshold_times_path = OUTPUT_DIR / "threshold_times.json"
if not _threshold_times_path.exists():
    raise FileNotFoundError(
        f"Threshold times JSON not found at {_threshold_times_path}. "
        "Run publications/plot_time_traces.py first."
    )
with open(_threshold_times_path) as _f:
    _threshold_times = json.load(_f)


def plot_one_camera(shotno, data, time, camera, output_dir, vmin=None, vmax=None, save=False, remove_offset=False,
                    scale="log"):
    """
    Plots the AXUV signal data as pcolormesh for one camera in one shot,
    on a log or linear colour scale.
    Optionally removes the starting time offset from the time array to start from 0.

    Parameters
    ----------
    shotno : int
        Shot number.
    data : np.ndarray
        AXUV signal data.
    time : np.ndarray
        Time array.
    camera : str
        Camera name.
    output_dir : pathlib.Path
        Root folder of the saved figures; they go to output_dir/<shotno>/.
    vmin : float, optional
        Minimum value for color scale. Defaults to DEFAULT_LIMITS[scale].
    vmax : float, optional
        Maximum value for color scale. Defaults to DEFAULT_LIMITS[scale];
        for the linear scale, that is the maximum of the data.
    save : bool, optional
        Whether to save the figure.
    remove_offset : bool, optional
        Whether to remove the starting time offset from the time array.
    scale : str, optional
        Colour scale, "log" or "linear". Linear figures are saved with a "_lin" suffix.
    """
    if remove_offset:
        # New array, so that the caller's time array is left unchanged
        time = time - time[0]

    default_vmin, default_vmax = DEFAULT_LIMITS[scale]
    vmin = default_vmin if vmin is None else vmin
    vmax = default_vmax if vmax is None else vmax
    suffix = scale_suffix(scale)

    fig, ax, pcm = plot_diode_evolution(time, data, scale=scale, vmin=vmin, vmax=vmax)
    ax.set_facecolor('k')

    # D16 and DVC look vertically and are marked with R, the horizontal ones with Z.
    add_direction_arrow(ax, "vertical" if "vert" in camera else "horizontal")

    if save:
        savename = str(shotno) + "_" + camera + "_notitle" + suffix
        save_png_pdf(fig, output_dir / str(shotno) / savename)

    firstcolor = "cyan"
    secondcolor = "lime"
    labelsize = 16 * FONT_SCALE

    shot_entry = _threshold_times.get(str(shotno), {})
    t_thr = [shot_entry.get(str(thr), {}).get("exp") for thr in THRESHOLDS]
    if PLOT_THRESHOLD_LINES and all(t is not None for t in t_thr):
        for t, color in zip(t_thr, [firstcolor, secondcolor]):
            ax.axvline(t, ls="--", color=color, linewidth=2)

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(t_thr)
        ax_top.set_xticklabels([rf"$t_{{{thr}W}}$" for thr in THRESHOLDS])
        ax_top.tick_params(direction='out', length=5, colors='black', labelsize=labelsize)
        ax_top.spines['top'].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

        savename = str(shotno) + "_" + camera + f"_{_pct_str}" + suffix
        save_png_pdf(fig, output_dir / str(shotno) / savename)

    title_suffix = camera.replace("_", " ")
    if "vert" in title_suffix:
        title_suffix = title_suffix.replace("vert", "vertical")
    elif "horiz" in title_suffix:
        title_suffix = title_suffix.replace("horiz", "horizontal")

    title = str(shotno) + " " + title_suffix
    plt.title(title)
    if save:
        savename = str(shotno) + "_" + camera + suffix
        save_png_pdf(fig, output_dir / str(shotno) / savename)

    plt.show()
    plt.close(fig)

def save_two_camera_plots(shotno, vert_data, vert_time, horiz_data, horiz_time, output_dir, save=False,
                          scales=PLOT_SCALES, limits=None):
    """
    Plots both cameras of one shot in every colour scale of *scales*.

    *limits* maps a scale to its (vmin, vmax) colour limits and overrides
    DEFAULT_LIMITS for that scale, e.g. {"linear": (0, 5e7)}. None in a
    limit keeps the default of plot_one_camera.
    """
    limits = limits or {}
    for scale in scales:
        vmin, vmax = limits.get(scale, (None, None))
        for data, time, camera in ((vert_data, vert_time, vert_camera), (horiz_data, horiz_time, horiz_camera)):
            plot_one_camera(shotno, data, time, camera, output_dir, vmin=vmin, vmax=vmax, save=save, scale=scale)


save_figures = True
# %%
shotno = 40673  # S16
starttime = SHOT_CONFIG[shotno]["starttime"]
endtime = starttime + SHOT_CONFIG[shotno]["duration"]
vert_camera = D16
horiz_camera = DHT

vert_data, vert_time = get_AXUV_signals(shot=shotno, camera=vert_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
horiz_data, horiz_time = get_AXUV_signals(shot=shotno, camera=horiz_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)

save_two_camera_plots(shotno, vert_data, vert_time, horiz_data, horiz_time, OUTPUT_DIR, save=save_figures)

# %%
shotno = 40673  # S05
starttime = SHOT_CONFIG[shotno]["starttime"]
endtime = starttime + SHOT_CONFIG[shotno]["duration"]
vert_camera = DVC
horiz_camera = DHC

vert_data, vert_time = get_AXUV_signals(shot=shotno, camera=vert_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
horiz_data, horiz_time = get_AXUV_signals(shot=shotno, camera=horiz_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
save_two_camera_plots(shotno, vert_data, vert_time, horiz_data, horiz_time, OUTPUT_DIR, save=save_figures)

# %%
shotno = 41007  # S16
starttime = SHOT_CONFIG[shotno]["starttime"]
endtime = starttime + SHOT_CONFIG[shotno]["duration"]
vert_camera = D16
horiz_camera = DHT

vert_data, vert_time = get_AXUV_signals(shot=shotno, camera=vert_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
horiz_data, horiz_time = get_AXUV_signals(shot=shotno, camera=horiz_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
save_two_camera_plots(shotno, vert_data, vert_time, horiz_data, horiz_time, OUTPUT_DIR, save=save_figures)

# %%
shotno = 41007  # S05
starttime = SHOT_CONFIG[shotno]["starttime"]
endtime = starttime + SHOT_CONFIG[shotno]["duration"]
vert_camera = DVC
horiz_camera = DHC

vert_data, vert_time = get_AXUV_signals(shot=shotno, camera=vert_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
horiz_data, horiz_time = get_AXUV_signals(shot=shotno, camera=horiz_camera, tbeg=starttime, tend=endtime, gaussian_sigma=10)
save_two_camera_plots(shotno, vert_data, vert_time, horiz_data, horiz_time, OUTPUT_DIR, save=save_figures)
