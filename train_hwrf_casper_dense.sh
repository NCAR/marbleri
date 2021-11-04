#!/bin/bash -l
#PBS -N htrainrt
#PBS -A NAML0001
#PBS -l select=1:ncpus=16:mem=256GB
#PBS -l walltime=2:00:00
#PBS -q casper
#PBS -j oe 
module load cuda/11 cudnn
export PATH="/glade/work/dgagne/miniconda3/envs/marbleri/bin:$PATH"
cd ~/marbleri/
python -u train_hwrf_ml.py config/hwrf_train_2020_dense_2021-11-03.yml -t
