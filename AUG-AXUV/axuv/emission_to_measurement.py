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

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from axuv.io import open_emission_data
from axuv.plotting import plot_time_evolution

# ── Filename pattern helpers ───────────────────────────────────────────────────
# Mirror the four save-names defined in calculate_emissions_for_raytransfer.py.

# Captures the resolution/masking variant (e.g. "lowres", "highres_masked").
_VARIANT_RE = re.compile(r"^emissions_((?:low|high)res(?:_masked)?)_")

# Strips the full prefix to leave only the time value string.
_TIME_RE = re.compile(r"^emissions_(?:low|high)res(?:_masked)?_")

# ── Processing constants ───────────────────────────────────────────────────────
_NOISE_SEED: int = 0

_COMBINATIONS: list[tuple[str, bool, bool]] = [
    # (subfolder_name,      noise,  degraded)
    ("nominal_no_noise", False, False),
    ("nominal_noise", True, False),
    ("degraded_no_noise", False, True),
    ("degraded_noise", True, True),
]


if __name__ == "__main__":
    import argparse

    plt.rcParams.update(
        {"font.size": 14, "figure.dpi": 300, "figure.constrained_layout.use": True}
    )

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
        default=None,
        metavar="N",
        help="Last diode index to include (0-based, exclusive). Defaults to all diodes.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=1e-5,
        metavar="V",
        help="Colour-scale lower limit for the log-normalised pcolormesh.",
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
            "(emissions_{lowres|highres}[_masked]_<time>.h5)."
        )

    # Sort each group by ascending physical time
    for files in groups.values():
        files.sort(key=lambda f: float(_TIME_RE.sub("", f.name).removesuffix(".h5")))

    # ── Resolve diode_last (peek at first file when not supplied) ─────────────
    diode_last: int = args.diode_last
    if diode_last is None:
        first_file = next(iter(groups.values()))[0]
        diode_last = open_emission_data(str(first_file))["diode_measurements"].shape[0]
        print(f"Detected {diode_last} diodes from '{first_file.name}'.")

    # ── Build output root ──────────────────────────────────────────────────────
    # <project_root>/output/<parent_dir_name>/<emissions_dir_name>/
    project_root = Path(__file__).parent.parent
    out_root = project_root / "output" / emissions_dir.parent.name / emissions_dir.name

    total_files = sum(len(v) for v in groups.values())
    print(
        f"\nProcessing {total_files} file(s) across {len(groups)} variant group(s).\n"
    )

    # ── Process each variant group ─────────────────────────────────────────────
    for variant, files in groups.items():
        filenames = [str(f) for f in files]

        # Extract times [ms] independently so they can be saved as times_ms.csv.
        # This mirrors the logic inside plot_time_evolution.
        times_ms = np.array(
            [float(_TIME_RE.sub("", f.name).removesuffix(".h5")) * 1e3 for f in files]
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

            fig, ax, pcm, data = plot_time_evolution(
                args.diode_first,
                diode_last,
                filenames,
                noise=noise,
                degraded=degraded,
                plot=True,
                vmin=args.vmin,
            )

            np.savetxt(out_dir / "diode_evolution.csv", data, delimiter=",")
            np.savetxt(out_dir / "times_ms.csv", times_ms, delimiter=",")
            fig.savefig(out_dir / "diode_evolution.png", bbox_inches="tight")
            plt.close(fig)

            print(f"    ✓ {subfolder}")

        print()

    print("Done.")
