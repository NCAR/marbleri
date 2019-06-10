#!/usr/bin/env bash
#SBATCH --job-name=hwrf_proc
#SBATCH --account=P48503002
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=dav
#SBATCH --mem=300G
#SBATCH --output=hwrf_proc.%j.out
module purge
module load gnu/7.3.0 openmpi/3.1.2 python/3.6.8 cuda/10.0
source /glade/work/dgagne/ncar_pylib_dl_10/bin/activate
cd ~/marbleri/
export OMP_NUM_THREADS=1
python setup.py install
python -u scripts/process_hwrf.py config/process_hwrf_casper.yml
