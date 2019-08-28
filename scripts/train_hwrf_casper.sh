#!/usr/bin/bash -l
#SBATCH --job-name=htrain
#SBATCH --account=NAML0001
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=1000G
#SBATCH --reservation=casper_8xV100
#SBATCH --output=hwrf_train.%j.out
module purge
module load gnu/7.3.0 python/3.6.8 openmpi/3.1.2 cuda/10.0
module load netcdf
source /glade/work/dgagne/ncar_pylib_dl_10/bin/activate
cd ~/marbleri/
#horovodrun -np 2 python -u scripts/train_ml_models.py config/train_hwrf_casper.yml
export CUDA_VISIBLE_DEVICES="5"
python -u scripts/train_ml_models_single_gpu.py config/train_hwrf_casper_dv.yml
