# used once to generate voxel grid

export datadir="/tokp/work/lefer/AXUV/data"
export currentdir="LF_FV_Ne0.10"


nice python3 voxels_sector_16.py $datadir/$currentdir/input/step02400_out.h5
