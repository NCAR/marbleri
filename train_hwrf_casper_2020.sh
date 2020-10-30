#!/usr/bin/bash -l
#SBATCH --job-name=htrain20
#SBATCH --account=NAML0001
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=256G
#SBATCH --output=hwrf_train.%j.out
module load cuda/10.1
export PATH="$HOME/miniconda3/envs/hfip/bin:$PATH"
cd ~/marbleri/
python -u train_hwrf_ml.py config/hwrf_train_2020.yml -t
