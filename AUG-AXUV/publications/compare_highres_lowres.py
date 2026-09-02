# %%
"""
Highres vs lowres synthetic brightness comparison.

For each (shot, sector) processed by plot_and_save_synthetic_measurements.sh,
loads the nominal_no_noise brightness_evolution.csv for every discovered
highres/lowres variant pair and computes:

  - per-channel time-averaged absolute relative error  |hi - lo| / |hi|
  - mean and max of those per-channel errors

Variant pairs are discovered automatically (e.g. norefl_highres ↔ norefl_lowres,
norefl_highres_masked ↔ norefl_lowres_masked, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from axuv.io import PROJECT_ROOT
from axuv.plotting import set_plt_rcparams

set_plt_rcparams()

project_dir = PROJECT_ROOT
output_dir  = project_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

COMBINATION = "nominal_no_noise"   # no noise → clean resolution comparison

CASES = [
    {"shot": "40673", "pfolder": "P1",  "sector": "S16"},
    {"shot": "40673", "pfolder": "P45", "sector": "S5"},
    {"shot": "41007", "pfolder": "P1",  "sector": "S16"},
    {"shot": "41007", "pfolder": "P45", "sector": "S5"},
]


# %% Helper functions

def _load_csv(path: Path):
    """Return (times, data) from a brightness_evolution.csv.

    data shape: (n_diodes, n_times).
    np.loadtxt skips # comment lines automatically.
    """
    raw = np.loadtxt(path, delimiter=",")
    return raw[:, 0], raw[:, 1:].T


def _find_variant_pairs(out_root: Path) -> list[tuple[str, str]]:
    """Return [(highres_variant, lowres_variant), ...] present under out_root."""
    variants = {d.name for d in out_root.iterdir() if d.is_dir()}
    pairs = []
    for v in sorted(variants):
        if "highres" in v:
            lo = v.replace("highres", "lowres")
            if lo in variants:
                pairs.append((v, lo))
    return pairs


# %% Load data and compute relative errors

results = {}   # (shot, sector, variant) → dict

for case in CASES:
    shot, pfolder, sector = case["shot"], case["pfolder"], case["sector"]
    out_root = output_dir / shot / pfolder

    if not out_root.exists():
        print(f"[skip] {shot}/{pfolder} — directory not found")
        continue

    pairs = _find_variant_pairs(out_root)
    if not pairs:
        print(f"[skip] {shot}/{pfolder} — no highres/lowres pairs found")
        continue

    for hi_v, lo_v in pairs:
        hi_csv = out_root / hi_v / COMBINATION / "brightness_evolution.csv"
        lo_csv = out_root / lo_v / COMBINATION / "brightness_evolution.csv"

        if not hi_csv.exists() or not lo_csv.exists():
            print(f"[skip] {shot} {sector} {hi_v}: missing CSV")
            continue

        times_hi, data_hi = _load_csv(hi_csv)
        times_lo, data_lo = _load_csv(lo_csv)

        assert np.allclose(times_hi, times_lo), (
            f"Time arrays differ between {hi_v} and {lo_v} for {shot}/{sector}"
        )

        # Per-channel absolute relative error averaged over all timesteps;
        # highres is the reference (denominator)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel_err_t = np.where(
                data_hi != 0,
                np.abs(data_hi - data_lo) / np.abs(data_hi),
                np.nan,
            )
        rel_err_per_channel = np.nanmean(rel_err_t, axis=1)   # (n_diodes,)

        key = (shot, sector, hi_v)
        results[key] = {
            "times":               times_hi,
            "data_hi":             data_hi,
            "data_lo":             data_lo,
            "rel_err_per_channel": rel_err_per_channel,
            "mean_rel_err":        float(np.nanmean(rel_err_per_channel)),
            "max_rel_err":         float(np.nanmax(rel_err_per_channel)),
            "lo_variant":          lo_v,
            "out_root":            out_root,
        }

        print(
            f"{shot}  {sector}  {hi_v} vs {lo_v}\n"
            f"  mean |rel err| = {results[key]['mean_rel_err']*100:.2f} %\n"
            f"  max  |rel err| = {results[key]['max_rel_err']*100:.2f} %"
        )


# %% Diagnostic plots — brightness and absolute difference per channel
# Inspect these to identify edge channels with large absolute differences.

for (shot, sector, hi_v), r in results.items():
    data_hi = r["data_hi"]
    data_lo = r["data_lo"]
    n = data_hi.shape[0]
    idx = np.arange(n)

    mean_hi   = np.nanmean(data_hi, axis=1)
    mean_lo   = np.nanmean(data_lo, axis=1)
    mean_diff = np.nanmean(np.abs(data_hi - data_lo), axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(idx, mean_hi, label="highres", linewidth=1)
    ax.plot(idx, mean_lo, label="lowres",  linewidth=1, linestyle="--")
    ax.set_ylabel(r"$\langle I \rangle$ [W/m$^2$]")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"\\#{shot}  {sector}  {hi_v} vs {r['lo_variant']}")

    ax = axes[1]
    ax.bar(idx, mean_diff, width=0.8, color="steelblue")
    ax.set_xlabel("Diode index")
    ax.set_ylabel(r"$\langle |I_\mathrm{hi} - I_\mathrm{lo}| \rangle$ [W/m$^2$]")
    ax.set_yscale("log")
    ax.set_xlim(-1, n)

    fname = f"abs_diff_{shot}_{sector}_{hi_v}"
    fig.savefig(r["out_root"] / f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(r["out_root"] / f"{fname}.pdf", format="PDF", bbox_inches="tight")
    plt.show()


# %% Channel exclusion — set these after inspecting the diagnostic plots above
# Specify per-sector lists of individual indices and/or contiguous ranges to exclude.
# Examples:
#   "S16": {"indices": [0, 1, 2, 93, 94, 95], "ranges": [(0, 3), (93, 96)]}
# Both are combined; leave empty to keep all channels for that sector.

EXCLUDE: dict[str, dict] = {
    "S16": {"indices": [45, 46, 48, 49, 85, 86, 87], "ranges": []},
    "S5":  {"indices": [45, 46, 48, 49, 86, 87], "ranges": []},
}


def _excluded_mask(n, indices, ranges):
    mask = np.zeros(n, dtype=bool)
    for i in indices:
        mask[i] = True
    for a, b in ranges:
        mask[a:b] = True
    return mask


# %% Summary table — full and with channels excluded

def _print_summary(label, results, exclude):
    print(f"\n{label}")
    print(f"{'Shot':<8}{'Sector':<8}{'Variant':<32}{'Mean |err| %':>14}{'Max |err| %':>13}")
    print("-" * 75)
    for (shot, sector, hi_v), r in results.items():
        rel  = r["rel_err_per_channel"]
        exc  = exclude.get(sector, {"indices": [], "ranges": []})
        mask = _excluded_mask(len(rel), exc["indices"], exc["ranges"])
        rel_kept = rel[~mask]
        if rel_kept.size == 0:
            continue
        print(
            f"{shot:<8}{sector:<8}{hi_v:<32}"
            f"{np.nanmean(rel_kept)*100:>13.2f} "
            f"{np.nanmax(rel_kept)*100:>12.2f}"
        )

_print_summary("All channels",     results, {"S16": {"indices": [], "ranges": []}, "S5": {"indices": [], "ranges": []}})
_print_summary("With exclusions",  results, EXCLUDE)


# %% Per-channel relative error bar charts — highlights excluded channels

for (shot, sector, hi_v), r in results.items():
    rel  = r["rel_err_per_channel"]
    n    = len(rel)
    exc  = EXCLUDE.get(sector, {"indices": [], "ranges": []})
    mask = _excluded_mask(n, exc["indices"], exc["ranges"])

    colors    = np.where(mask, "lightgray", "steelblue")
    rel_kept  = rel[~mask]
    mean_kept = float(np.nanmean(rel_kept)) if rel_kept.size else np.nan

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(n), rel * 100, width=0.8, color=colors)
    if np.isfinite(mean_kept):
        ax.axhline(mean_kept * 100, color="r", linewidth=1, linestyle="--",
                   label=f"mean (kept) = {mean_kept*100:.2f}\\%")
    ax.set_xlabel("Diode index")
    ax.set_ylabel(r"$|\Delta I / I_\mathrm{hi}|$ (\%)")
    ax.set_title(f"\\#{shot}  {sector}  {hi_v} vs {r['lo_variant']}")
    ax.set_xlim(-1, n)
    ax.set_xticks(range(n))
    ax.tick_params(axis="x", labelsize=6)
    ax.set_yscale("log")
    ax.legend()

    fname = f"rel_err_{shot}_{sector}_{hi_v}"
    fig.savefig(r["out_root"] / f"{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(r["out_root"] / f"{fname}.pdf", format="PDF", bbox_inches="tight")
    plt.show()
