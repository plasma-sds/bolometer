export maindir="/tok/u/lefer/work/AUG-AXUV/axuv/"
export datadir="$maindir/data"

# Sector 16
python $maindir/emission_to_measurement.py $datadir/40673/P1/output
python $maindir/emission_to_measurement.py $datadir/41007/P1/output

# Sector 5
python $maindir/emission_to_measurement.py $datadir/40673/P45/output
python $maindir/emission_to_measurement.py $datadir/41007/P45/output
