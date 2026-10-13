#!/usr/bin/env bash
# Save all threshold-related figures from the SYNTHETIC data.
# Run this wherever the Stage 2 emission data lives ($AXUV_OUTPUT_DIR/<shot>/<phase>/emissions/).
#
# W_th thresholds are defined centrally in axuv/io.py (WTH_THRESHOLDS).
# Step 1 (plot_time_traces.py) computes the threshold crossing times and writes
# $AXUV_OUTPUT_DIR/threshold_times.json; the per-dataset runs read the synthetic ("sim") times.
set -euo pipefail

# Resolve the repository root from this script's location so it runs anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Make the packages importable and force a headless matplotlib backend.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="Agg"

maindir="$REPO_ROOT/axuv"
pubdir="$REPO_ROOT/publications"
export AXUV_OUTPUT_DIR="${AXUV_OUTPUT_DIR:-$REPO_ROOT/output}"

# Step 1: compute and cache threshold crossing times (also saves the time-trace figures)
python "$pubdir/plot_time_traces.py"

# Step 2: save the synthetic AXUV measurement figures (with W_th vertical lines),
# on log and linear colour scales (linear files get a "_lin" suffix)
# Sector 16
python "$maindir/emission_to_measurement.py" "$AXUV_OUTPUT_DIR/40673/P1/emissions" --scale both
python "$maindir/emission_to_measurement.py" "$AXUV_OUTPUT_DIR/41007/P1/emissions" --scale both

# Sector 5
python "$maindir/emission_to_measurement.py" "$AXUV_OUTPUT_DIR/40673/P45/emissions" --scale both
python "$maindir/emission_to_measurement.py" "$AXUV_OUTPUT_DIR/41007/P45/emissions" --scale both
