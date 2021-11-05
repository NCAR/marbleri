#!/bin/bash -l
#PBS -N htrainrt
#PBS -A NAML0001
#PBS -l select=1:ncpus=16:ngpus=1:mem=256GB
#PBS -l gpu_type=v100
#PBS -l walltime=4:00:00
#PBS -q gpgpu
#PBS -j oe 
module load cuda/11 cudnn nccl
export PATH="/glade/work/dgagne/miniconda3/envs/marbleri/bin:$PATH"
cd ~/marbleri/
python -u train_hwrf_ml.py hwrf_train_2020_2021-11-03.yml -t
