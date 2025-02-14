#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=simil
python experiments/comparison/get_similarity_comparison.py
