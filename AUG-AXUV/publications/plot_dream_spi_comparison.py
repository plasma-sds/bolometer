"""Compare the DREAM SPI sightline runs with and without the W + B background.

Reads the step files of publications/run_dream_spi_sightlines.py and writes,
under <output root>/plots/:
- summary.csv: one row per run and step (time, total power, eta_eff for the
  three responsivities, W and B shares);
- eta_eff_<shot>.png/.pdf: eta_eff against t_rel per run, as the paper's
  eta_eff figure;
- power_<shot>.png/.pdf: total power against t_rel per run, with the W and B
  shares of the runs with impurities.

    python publications/plot_dream_spi_comparison.py [--output-root DIR]
"""

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from axuv.io import OUTPUT_DIR  # noqa: E402
from axuv.plotting import set_plt_rcparams  # noqa: E402

RUN_STYLE = {
    "noImpurities": ("k", "-", "without W + B"),
    "withImpurities_Wcl": ("r", "--", "with W + B (W cl)"),
    "withImpurities_Wca": ("b", "-.", "with W + B (W ca)"),
}
ETA = (("nominal", "Nominal"), ("degraded", "Degraded"), ("worst", "Worst estimated"))
COLUMNS = ("run", "t_rel_grid_ms", "t_rel_ms", "source_file", "source_index", "total_power_W",
           "eta_eff_nominal_A_per_W", "eta_eff_degraded_A_per_W", "eta_eff_worst_A_per_W",
           "share_W", "share_B")


def read_runs(root):
    """Rows per run directory, sorted by time."""
    runs = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name[:5].isdigit()):
        rows = []
        for path in sorted(run_dir.glob("trel*.h5")):
            with h5py.File(path, "r") as f:
                a = f.attrs
                rows.append({
                    "run": run_dir.name,
                    "t_rel_grid_ms": a["t_rel_grid_s"] * 1e3,
                    "t_rel_ms": a["t_rel_s"] * 1e3,
                    "source_file": a["source_file"],
                    "source_index": int(a["source_index"]),
                    "total_power_W": a["total_power_W"],
                    **{f"eta_eff_{k}_A_per_W": a[f"eta_eff_{k}_A_per_W"] for k, _ in ETA},
                    "share_W": a.get("share_W", 0.0),
                    "share_B": a.get("share_B", 0.0),
                    "flags": a.get("flags", ""),
                })
        rows.sort(key=lambda row: row["t_rel_ms"])
        if rows:
            runs[run_dir.name] = rows
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR / "1D_DREAM_SPI")
    args = parser.parse_args()
    root = args.output_root
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    runs = read_runs(root)

    with open(plots / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rows in runs.values():
            writer.writerows(rows)
    flags = sorted({row["flags"] for rows in runs.values() for row in rows if row["flags"]})
    (plots / "flags.txt").write_text("\n".join(flags) + "\n")

    set_plt_rcparams()
    for shot in sorted({name.split("_")[0] for name in runs}):
        shot_runs = {name.split("_", 1)[1]: rows for name, rows in runs.items() if name.startswith(shot)}

        fig, axes = plt.subplots(1, len(ETA), figsize=(13, 4), sharey=True)
        for ax, (key, title) in zip(axes, ETA):
            for label, rows in shot_runs.items():
                color, style, legend = RUN_STYLE.get(label, ("grey", ":", label))
                t = [row["t_rel_ms"] for row in rows]
                ax.plot(t, [row[f"eta_eff_{key}_A_per_W"] for row in rows],
                        color=color, ls=style, marker="o", markersize=3, label=legend)
            ax.set_title(f"{title} responsivity")
            ax.set_xlabel(r"$t - t_\mathrm{TQ}$ (ms)")
            ax.axvline(0, color="grey", lw=1)
        axes[0].set_ylabel(r"$\eta_\text{eff}$ (A W$^{-1}$)")
        axes[0].set_ylim(0, 0.3)
        axes[0].legend(fontsize="small")
        fig.suptitle(rf"\#{shot}")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(plots / f"eta_eff_{shot}.{ext}")
        plt.close(fig)

        fig, (ax, ax_share) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for label, rows in shot_runs.items():
            color, style, legend = RUN_STYLE.get(label, ("grey", ":", label))
            t = [row["t_rel_ms"] for row in rows]
            ax.plot(t, [row["total_power_W"] for row in rows], color=color, ls=style, marker="o",
                    markersize=3, label=legend)
            if label != "noImpurities":
                ax_share.plot(t, [row["share_W"] for row in rows], color=color, ls=style, marker="o",
                              markersize=3, label=f"W share, {legend}")
                ax_share.plot(t, [row["share_B"] for row in rows], color=color, ls=style, marker="s",
                              markersize=3, alpha=0.5, label=f"B share, {legend}")
        ax.set_yscale("log")
        ax.set_ylabel("Power (W)")  # sightline power for unit etendue (1 m^2 sr)
        # legends outside the axes, so they never cover data points
        ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1))
        ax_share.set_yscale("log")
        ax_share.set_ylabel("Share of power")
        ax_share.set_xlabel(r"$t - t_\mathrm{TQ}$ (ms)")
        ax_share.legend(fontsize="x-small", loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.align_ylabels((ax, ax_share))
        for a in (ax, ax_share):
            a.axvline(0, color="grey", lw=1)
        fig.suptitle(rf"\#{shot}")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(plots / f"power_{shot}.{ext}")
        plt.close(fig)
    print(f"Wrote {plots}")


if __name__ == "__main__":
    main()
