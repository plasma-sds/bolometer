#!/usr/bin/env bash
# Stage 2 (emissions) on a local machine, the counterpart of parallel_submit_raytransfer.sh.
# Runs calculate_emissions_for_raytransfer.py on every JOREK input file of the four
# datasets, with the lowres ray-transfer matrix of the chosen variant.
#
# Usage: run_emissions_local.sh <norefl|refl> [jobs]
#   jobs defaults to 8. Each process peaks at about 1.7 GB of memory, so size it
#   by free memory, not by core count.
set -uo pipefail

variant=${1:?usage: run_emissions_local.sh <norefl|refl> [jobs]}
jobs=${2:-8}

# Resolve the repository root from this script's location so it runs anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

maindir="$REPO_ROOT/axuv"
datadir="$maindir/data"
# Stage 2 writes to $AXUV_OUTPUT_DIR/<dataset>/emissions/, logs go to <dataset>/stdout/
export AXUV_OUTPUT_DIR="${AXUV_OUTPUT_DIR:-$REPO_ROOT/output}"
python="$REPO_ROOT/.venv/bin/python"
[ -x "$python" ] || python=python3

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="Agg"

# <dataset> <sector of its ray-transfer matrix>
for pair in "40673/P1 S16" "41007/P1 S16" "40673/P45 S5" "41007/P45 S5"; do
    read -r currentdir sector <<< "$pair"
    logdir="$AXUV_OUTPUT_DIR/$currentdir/stdout"
    mkdir -p "$logdir"
    matrix="$datadir/raytransfer_${sector}_${variant}.h5"
    echo "$(date +%T) working on $currentdir with $(basename "$matrix")"
    ls "$datadir/$currentdir/input" | parallel -j "$jobs" --halt never \
        "nice -n 5 $python -u $maindir/calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' --raytransfer-file $matrix --dataset $currentdir &> $logdir/output_${variant}_{}.txt"
    echo "$(date +%T) $currentdir done, exit $?, $(ls "$AXUV_OUTPUT_DIR/$currentdir/emissions" | grep -c "emissions_${variant}_") ${variant} output files"
done
