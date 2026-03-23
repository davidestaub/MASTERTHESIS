#!/bin/bash
#SBATCH -n 10
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20g
#SBATCH --time=4:00:00

python fluid_only.py
