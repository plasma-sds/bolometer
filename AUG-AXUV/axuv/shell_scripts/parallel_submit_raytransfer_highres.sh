#!/usr/bin/env bash
# Stage 2 (emissions) on the cluster with the highres (1 cm x 1 cm) ray-transfer
# matrices: every JOREK input of the four datasets, 20 parallel jobs, each pinned
# to one core with taskset.
# The highres matrices must be uploaded (axuv.publish_data) and listed in
# axuv/data_index.json first. Add "highres_refl" to VARIANTS for the runs with
# reflections.
set -uo pipefail

VARIANTS="highres_norefl"

# Resolve the repository root from this script's location so it runs anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

maindir="$REPO_ROOT/axuv"
# Stage 2 writes to $AXUV_OUTPUT_DIR/<dataset>/emissions/, logs go to <dataset>/stdout/
export AXUV_OUTPUT_DIR="${AXUV_OUTPUT_DIR:-$REPO_ROOT/output}"
python="$REPO_ROOT/.venv/bin/python"
[ -x "$python" ] || python=python3

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="Agg"
export PLASMASDS_QUIET=1

for variant in $VARIANTS; do
    # <dataset> <sector of its ray-transfer matrix>
    for pair in "40673/P1 S16" "41007/P1 S16" "40673/P45 S5" "41007/P45 S5"; do
        read -r currentdir sector <<< "$pair"
        logdir="$AXUV_OUTPUT_DIR/$currentdir/stdout"
        mkdir -p "$logdir"
        # Fetch everything once, before the workers start, so none of them downloads
        matrix=$("$python" -m axuv.fetch_data --paths "raytransfer/raytransfer_${sector}_${variant}.h5") || exit 1
        inputs=$("$python" -m axuv.fetch_data --paths "jorek/$currentdir") || exit 1
        "$python" -m axuv.fetch_data atomic geometry || exit 1
        echo "Working on $currentdir with $(basename "$matrix")..."
        echo "$inputs" | parallel -j 20 \
            "nice -n 5 taskset -c {#} $python -u $maindir/calculate_emissions_for_raytransfer.py {} --raytransfer-file $matrix --dataset $currentdir --resolution-r 120 --resolution-z 220 &> $logdir/output_${variant}_{/}.txt"
        echo "$currentdir done"
    done
done
