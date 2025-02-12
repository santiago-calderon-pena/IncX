#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=incremental_baseline_gpu
#SBATCH --gres=gpu
#SBATCH -p nmes_gpu
python get_saliency_maps.py
