export datadir="/tok/u/lefer/work/AUG-AXUV/AXUV/data"

export currentdir="test_uniform"

mkdir -p $datadir/$currentdir/stdout
mkdir -p $datadir/$currentdir/output

python3 calculate_emissions_for_raytransfer.py $datadir/$currentdir/input/step02400_out.h5
