#!/usr/bin/env bash
#SBATCH --job-name=hwrf_proc
#SBATCH --account=P48503002
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:8
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --output=hwrf_proc.%j.out
module purge
module load gnu/7.3.0 openmpi-x/3.1.0 python/3.6.8 cuda/10.1
source /glade/work/dgagne/ncar_pylib_dl_10/bin/activate
cd ~/marbleri/
python setup.py install
mpirun python -u scripts/train_ml_models.py config/train_hwrf_casper.yml
