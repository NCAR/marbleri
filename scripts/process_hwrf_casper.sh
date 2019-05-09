#!/usr/bin/env bash
#SBATCH --job-name=hwrf_proc
#SBATCH --account=P48503002
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=dav
#SBATCH --mem=200G
module purge
module load gnu/7.3.0 openmpi-x/3.1.0 python/3.6.4 cuda/9.2
source /glade/work/dgagne/ncar_pylib_dl/bin/activate
cd ~/marbleri/
python setup.py install
python -u scripts/process_hwrf.py config/process_hwrf_casper.yml >& process_hwrf.log
