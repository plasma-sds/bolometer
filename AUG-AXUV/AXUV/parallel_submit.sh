export datadir="/tokp/work/lefer/AXUV/data"

export currentdir="LF_HV_Ne10"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output

ls $datadir/$currentdir/input | parallel -j 10 "nice -n 5 taskset -c {#} python3 -u voxels_sector_16.py '$datadir/$currentdir/input/{}' &> $datadir/$currentdir/stdout/output_{}.txt" 

wait

echo "$currentdir done"

# export currentdir="SF_FV_Ne0.12"

# mkdir -p $datadir/$currentdir/stdout
# mkdir -p $datadir/$currentdir/output

# ls $datadir/$currentdir/input | parallel -j 10 "nice -n 5 taskset -c {#} python3 -u voxels_sector_16.py '$datadir/$currentdir/input/{}' &> $datadir/$currentdir/stdout/output_{}.txt" 

# wait

# echo "$currentdir done"
