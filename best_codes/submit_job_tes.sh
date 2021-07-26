#!/bin/sh -login
#
#PBS -d .
#PBS -l partition=vjet:tjet
#PBS -l procs=1
#PBS -A hwrf-anen
#PBS -N convert_nc
#PBS -lvmem=2500M
#PBS -j oe
#PBS -l walltime=02:00:00
#
module load intel
module load netcdf
module load ncl
module load wgrib2
#
basedir='/lfs3/projects/sso/Chris.Rozoff/h2/'
#
cd ${basedir}
echo $alpha
bash g2nc_tes.sh $alpha
