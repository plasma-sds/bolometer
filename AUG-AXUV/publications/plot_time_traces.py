# %%
import json
import matplotlib.pyplot as plt
import numpy as np

from axuv.io import PROJECT_ROOT, WTH_THRESHOLDS as THRESHOLDS
from axuv.plotting import set_plt_rcparams

set_plt_rcparams()

project_dir = PROJECT_ROOT
data_dir = project_dir / "axuv" / "data"
output_dir = project_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

SHOTS = [40673, 41007]
START_TIMES = [2.3276, 2.3417]
include_frad = False
if include_frad:
    VARIABLES = ["frad", "ip", "wth"]
else:
    VARIABLES = ["ip", "wth"]
COLORS = {"frad": "blue", "ip": "black", "wth": "red"}
LABELS = {
    "frad": r"$f_\mathrm{rad}$",
    "ip": r"$I_\mathrm{p} / I_\mathrm{p,0}$",
    "wth": r"$W_\mathrm{th} / W_\mathrm{th,0}$",
}

# %% Load and normalize
data = {}
for shot in SHOTS:
    shot_dir = data_dir / str(shot)
    data[shot] = {}
    for var in VARIABLES:
        sim = np.loadtxt(shot_dir / f"{var}-jorek-{shot}.txt")
        exp = np.loadtxt(shot_dir / f"{var}-exp-{shot}.txt")
        if var != "frad":
            sim[:, 1] = sim[:, 1] / sim[0, 1]
            exp[:, 1] = exp[:, 1] / exp[0, 1]
        data[shot][var] = {"sim": sim, "exp": exp}


def crossing_time(t, y, threshold):
    """Return the interpolated time when y first drops below threshold."""
    below = np.where(y < threshold)[0]
    if len(below) == 0:
        return None
    idx = below[0]
    if idx == 0:
        return t[0]
    t0, t1 = t[idx - 1], t[idx]
    y0, y1 = y[idx - 1], y[idx]
    return t0 + (threshold - y0) * (t1 - t0) / (y1 - y0)


# %% Compute and cache threshold crossing times
# THRESHOLDS is defined centrally in axuv.io (WTH_THRESHOLDS).
# Already-computed entries in threshold_times.json are reused without recomputing.
THRESHOLD_COLORS = ["deepskyblue", "limegreen"]

_threshold_times_path = output_dir / "threshold_times.json"
threshold_times = {}
if _threshold_times_path.exists():
    with open(_threshold_times_path) as f:
        threshold_times = json.load(f)

for shot in SHOTS:
    shot_key = str(shot)
    threshold_times.setdefault(shot_key, {})
    wth_sim = data[shot]["wth"]["sim"]
    wth_exp = data[shot]["wth"]["exp"]
    for thr in THRESHOLDS:
        thr_key = str(thr)
        if thr_key not in threshold_times[shot_key]:
            t_sim = crossing_time(wth_sim[:, 0], wth_sim[:, 1], thr)
            t_exp = crossing_time(wth_exp[:, 0], wth_exp[:, 1], thr)
            threshold_times[shot_key][thr_key] = {"sim": t_sim, "exp": t_exp}

with open(_threshold_times_path, "w") as f:
    json.dump(threshold_times, f, indent=2)

_pct_str = "_".join(str(int(thr * 100)) for thr in THRESHOLDS) + "pct"

# %% Plot
for shot in SHOTS:
    fig, ax = plt.subplots(figsize=(4.95, 5))

    t_start = START_TIMES[SHOTS.index(shot)]
    t_end = max(data[shot][var]["sim"][-1, 0] for var in VARIABLES)

    for var in VARIABLES:
        sim = data[shot][var]["sim"]
        exp = data[shot][var]["exp"]
        color = COLORS[var]
        label = LABELS[var]

        ax.plot(
            sim[:, 0], sim[:, 1], color=color, linestyle="-", label=f"{label} JOREK"
        )

        if var == "frad":
            ax.scatter(
                exp[:, 0],
                exp[:, 1],
                color=color,
                marker="x",
                s=20,
                zorder=5,
                label=f"{label} exp",
            )
        else:
            ax.plot(
                exp[:, 0], exp[:, 1], color=color, linestyle=":", label=f"{label} exp"
            )

    # Vertical lines at W_th thresholds
    for i, thr in enumerate(THRESHOLDS):
        entry = threshold_times[str(shot)][str(thr)]
        t_sim = entry["sim"]
        t_exp = entry["exp"]
        label_str = rf"$t_{{{thr}W}}$"
        if t_sim is not None:
            ax.axvline(
                t_sim,
                color=THRESHOLD_COLORS[i],
                linestyle="-",
                label=f"{label_str} JOREK",
            )
        if t_exp is not None:
            ax.axvline(
                t_exp,
                color=THRESHOLD_COLORS[i],
                linestyle=":",
                label=f"{label_str} exp",
            )

    ax.set_xlim(t_start, t_end)
    ax.set_ylim(0, 1.05)
    ax.grid()
    ax.set_xlabel("Time [s]")
    # ax.set_title(f"\\#{shot}")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.1, 1.02), borderaxespad=0, ncols=2, fontsize="small")

    plt.savefig(output_dir / f"time_traces_{shot}_{_pct_str}.png", dpi=300, bbox_inches="tight")
    plt.savefig(
        output_dir / f"time_traces_{shot}_{_pct_str}.eps", format="eps", bbox_inches="tight"
    )
    plt.show()
