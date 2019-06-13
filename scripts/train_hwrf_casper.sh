#!/usr/bin/bash -l
#SBATCH --job-name=htrain
#SBATCH --account=P48503002
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=500G
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --output=hwrf_train.%j.out
module purge
module load gnu/7.3.0 python/3.6.8 openmpi/3.1.2 cuda/10.0
source /glade/work/dgagne/ncar_pylib_dl_10/bin/activate
cd ~/marbleri/
python setup.py install
horovodrun -np 2 python -u scripts/train_ml_models.py config/train_hwrf_casper.yml
