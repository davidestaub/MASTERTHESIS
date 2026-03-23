#!/bin/bash
#SBATCH -n 10
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:80g
#SBATCH --time=4:00:00

python solid_with_analytic2.py
