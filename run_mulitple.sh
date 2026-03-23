#!/bin/bash
#SBATCH -n 10
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=12:00:00

# Assign command-line arguments to variables
N_1=$1
N_NEURONS=$2
N_HIDDEN_LAYERS=$3

# Copy the original configuration file to a new file that includes the parameter values
CONFIG_FILE="config_n1_${N_1}_neurons_${N_NEURONS}_hidden_${N_HIDDEN_LAYERS}.ini"
cp config.ini $CONFIG_FILE

# Use `sed` to replace the values in the new configuration file
sed -i "s/^n_1=.*/n_1=${N_1}/" $CONFIG_FILE
sed -i "s/^n_neurons = .*/n_neurons = ${N_NEURONS}/" $CONFIG_FILE
sed -i "s/^n_hidden_layers = .*/n_hidden_layers = ${N_HIDDEN_LAYERS}/" $CONFIG_FILE

# Run the python script with the new configuration file
python elastic_wave_main_2.py $CONFIG_FILE 0
