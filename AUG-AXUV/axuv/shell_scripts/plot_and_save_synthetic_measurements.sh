#!/usr/bin/env bash
# Save all threshold-related figures from the SYNTHETIC data.
# Run this wherever the synthetic emission data lives (under axuv/data/<shot>/...).
#
# W_th thresholds are defined centrally in axuv/io.py (WTH_THRESHOLDS).
# Step 1 (plot_time_traces.py) computes the threshold crossing times and writes
# output/threshold_times.json; the per-dataset runs read the synthetic ("sim") times.
set -euo pipefail

# Resolve the repository root from this script's location so it runs anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Make the packages importable and force a headless matplotlib backend.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="Agg"

maindir="$REPO_ROOT/axuv"
datadir="$maindir/data"
pubdir="$REPO_ROOT/publications"

# Step 1: compute and cache threshold crossing times (also saves the time-trace figures)
python "$pubdir/plot_time_traces.py"

# Step 2: save the synthetic AXUV measurement figures (with W_th vertical lines)
# Sector 16
python "$maindir/emission_to_measurement.py" "$datadir/40673/P1/output"
python "$maindir/emission_to_measurement.py" "$datadir/41007/P1/output"

# Sector 5
python "$maindir/emission_to_measurement.py" "$datadir/40673/P45/output"
python "$maindir/emission_to_measurement.py" "$datadir/41007/P45/output"
