export datadir="/tok/u/lefer/work/AUG-AXUV/AXUV/data"

export currentdir="40673/P1"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output

ls $datadir/$currentdir/input | parallel -j 10 "nice -n 5 taskset -c {#} python3 -u calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' &> $datadir/$currentdir/stdout/output_{}.txt" 

wait

echo "$currentdir done"

export currentdir="41007/P1"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output

ls $datadir/$currentdir/input | parallel -j 10 "nice -n 5 taskset -c {#} python3 -u calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' &> $datadir/$currentdir/stdout/output_{}.txt" 

wait

echo "$currentdir done"

