#!/bin/bash --login
#
#SBATCH --account=hwrf-anen
#SBATCH --job-name=convert_nc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=vjet
#SBATCH --time=01:00:00
#SBATCH --output=may.out
#
module load intel
module load szip
module load hdf5
module load netcdf
module load ncl
module load wgrib2
#
basedir='/lfs3/projects/sso/Chris.Rozoff/h2/'
#
cd ${basedir}
for p in `cat dir_may.txt`
do
 bash g2nc_tes.sh $p
done
