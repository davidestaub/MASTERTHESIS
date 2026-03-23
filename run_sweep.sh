#!/bin/bash
#SBATCH -n 10
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:20g
#SBATCH --time=300:00:00

wandb agent staub/uncategorized/7qlx947e

