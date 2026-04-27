export maindir="/tok/u/lefer/work/AUG-AXUV/axuv/"
export datadir="$maindir/data"

export currentdir="40673/P1"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output

echo "Working on $currentdir..."

ls $datadir/$currentdir/input | parallel -j 20 "nice -n 5 taskset -c {#} python3 -u $maindir/calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' --raytransfer-file $datadir/raytransfer_S16_norefl.h5 &> $datadir/$currentdir/stdout/output_{}.txt"

wait

echo "$currentdir done"

export currentdir="41007/P1"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output

echo "Working on $currentdir..."

ls $datadir/$currentdir/input | parallel -j 20 "nice -n 5 taskset -c {#} python3 -u $maindir/calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' --raytransfer-file $datadir/raytransfer_S16_norefl.h5 &> $datadir/$currentdir/stdout/output_{}.txt"

wait

echo "$currentdir done"

export currentdir="40673/P45"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output

echo "Working on $currentdir..."

ls $datadir/$currentdir/input | parallel -j 20 "nice -n 5 taskset -c {#} python3 -u $maindir/calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' --raytransfer-file $datadir/raytransfer_S5_norefl.h5 &> $datadir/$currentdir/stdout/output_{}.txt"

wait

echo "$currentdir done"

export currentdir="41007/P45"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output


echo "Working on $currentdir..."

ls $datadir/$currentdir/input | parallel -j 20 "nice -n 5 taskset -c {#} python3 -u $maindir/calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' --raytransfer-file $datadir/raytransfer_S5_norefl.h5 &> $datadir/$currentdir/stdout/output_{}.txt"

wait

echo "$currentdir done"

#######################
# With reflections ON #
#######################

export currentdir="40673/P1"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output

echo "Working on $currentdir..."

ls $datadir/$currentdir/input | parallel -j 20 "nice -n 5 taskset -c {#} python3 -u $maindir/calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' --raytransfer-file $datadir/raytransfer_S16_refl.h5 &> $datadir/$currentdir/stdout/output_{}.txt"

wait

echo "$currentdir done"

export currentdir="41007/P1"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output

echo "Working on $currentdir..."

ls $datadir/$currentdir/input | parallel -j 20 "nice -n 5 taskset -c {#} python3 -u $maindir/calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' --raytransfer-file $datadir/raytransfer_S16_refl.h5 &> $datadir/$currentdir/stdout/output_{}.txt"

wait

echo "$currentdir done"

export currentdir="40673/P45"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output

echo "Working on $currentdir..."

ls $datadir/$currentdir/input | parallel -j 20 "nice -n 5 taskset -c {#} python3 -u $maindir/calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' --raytransfer-file $datadir/raytransfer_S5_refl.h5 &> $datadir/$currentdir/stdout/output_{}.txt"

wait

echo "$currentdir done"

export currentdir="41007/P45"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output


echo "Working on $currentdir..."

ls $datadir/$currentdir/input | parallel -j 20 "nice -n 5 taskset -c {#} python3 -u $maindir/calculate_emissions_for_raytransfer.py '$datadir/$currentdir/input/{}' --raytransfer-file $datadir/raytransfer_S5_refl.h5 &> $datadir/$currentdir/stdout/output_{}.txt"

wait

echo "$currentdir done"
