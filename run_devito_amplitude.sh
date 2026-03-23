#!/bin/bash
#SBATCH -n 10
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=1 
#SBATCH --gres=gpumem:10g
#SBATCH --time=01:00:00


python amplitude_response_DEVITO.py 10  
