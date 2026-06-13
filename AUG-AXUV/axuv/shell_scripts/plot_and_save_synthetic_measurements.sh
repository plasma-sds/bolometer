export maindir="/tok/u/lefer/work/AUG-AXUV/axuv/"
export datadir="$maindir/data"
export pubdir="/tok/u/lefer/work/AUG-AXUV/publications/"

# Change THRESHOLDS to regenerate all figures at a different level.
# Must match the THRESHOLDS list in publications/plot_time_traces.py.
THRESHOLDS="0.9 0.1"

# Compute and export threshold crossing times (skips already-cached entries)
python $pubdir/plot_time_traces.py

# Sector 16
python $maindir/emission_to_measurement.py $datadir/40673/P1/output --thresholds $THRESHOLDS
python $maindir/emission_to_measurement.py $datadir/41007/P1/output --thresholds $THRESHOLDS

# Sector 5
python $maindir/emission_to_measurement.py $datadir/40673/P45/output --thresholds $THRESHOLDS
python $maindir/emission_to_measurement.py $datadir/41007/P45/output --thresholds $THRESHOLDS
