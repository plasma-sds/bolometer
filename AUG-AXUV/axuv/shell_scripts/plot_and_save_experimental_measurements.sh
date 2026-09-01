#!/usr/bin/env bash
# Save all threshold-related figures from the EXPERIMENTAL data.
# Run this on the remote server that has AUG shotfile access.
#
# W_th thresholds are defined centrally in axuv/io.py (WTH_THRESHOLDS).
# Step 1 (plot_time_traces.py) computes the threshold crossing times and writes
# output/threshold_times.json; step 2 reads the experimental ("exp") times from it.
set -euo pipefail

# Resolve the repository root from this script's location so it runs anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Make the packages importable and force a headless matplotlib backend.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="Agg"

# Step 1: compute and cache threshold crossing times (also saves the time-trace figures)
python "$REPO_ROOT/publications/plot_time_traces.py"

# Step 2: save the experimental AXUV camera figures (with W_th vertical lines)
python "$REPO_ROOT/axuv_measurement/plotting_from_shotfiles.py"
