#!/usr/bin/bash -l
#SBATCH --job-name=htrainrt
#SBATCH --account=NAML0001
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=200G
#SBATCH --output=hwrf_train.%j.out
module load cuda/11 cudnn nccl
export PATH="$HOME/miniconda3/envs/marbleri/bin:$PATH"
cd ~/marbleri/
#python -u train_hwrf_ml.py config/hwrf_train_2020_realtime.yml -t
python -u train_hwrf_ml.py config/hwrf_train_2020_realtime_3h.yml -t
