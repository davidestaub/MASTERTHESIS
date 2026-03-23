#!/bin/bash
#SBATCH -n 10
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
#SBATCH --time=120:00:00

# Assign command-line arguments to variables
N_1=$1
N_NEURONS=$2
N_HIDDEN_LAYERS=$3
N_HIDDEN_LAYERS_AFTER=$4
N_NEURONS_AFTER=$5
NN_TYPE=$6
TAG=$7
T1=$8  # New variable for t1
SIGMA=$9

# Copy the original configuration file to a new file that includes the parameter values
CONFIG_FILE="config_n1_${N_1}_neurons_${N_NEURONS}_hidden_${N_HIDDEN_LAYERS}_hiddenafter_${N_HIDDEN_LAYERS_AFTER}_neuronsafter_${N_NEURONS_AFTER}_type_${NN_TYPE}_tag_${TAG}_t1_${T1}_sigma${SIGMA}.ini"
cp config.ini $CONFIG_FILE

# Use `sed` to replace the values in the new configuration file
sed -i "s/^[[:space:]]*n_1=.*/n_1=${N_1}/" $CONFIG_FILE
sed -i "s/^[[:space:]]*n_neurons = .*/n_neurons = ${N_NEURONS}/" $CONFIG_FILE
sed -i "s/^[[:space:]]*n_hidden_layers = .*/n_hidden_layers = ${N_HIDDEN_LAYERS}/" $CONFIG_FILE
sed -i "s/^[[:space:]]*n_hidden_layers_after = .*/n_hidden_layers_after = ${N_HIDDEN_LAYERS_AFTER}/" $CONFIG_FILE
sed -i "s/^[[:space:]]*n_neurons_after = .*/n_neurons_after = ${N_NEURONS_AFTER}/" $CONFIG_FILE
sed -i "s/^[[:space:]]*nn_type = .*/nn_type = ${NN_TYPE}/" $CONFIG_FILE
sed -i "s/^[[:space:]]*tag=.*$/tag=${TAG}/" $CONFIG_FILE
# Replace the t1 value under the [initial_condition] section
sed -i "/^\[initial_condition\]/,/^\[.*\]/ s/^t1=.*/t1=${T1}/" $CONFIG_FILE
sed -i "s/^[[:space:]]*sigma_quake = .*$/sigma_quake = ${SIGMA}/" $CONFIG_FILE

# Run the python script with the new configuration file
CUDA_LAUNCH_BLOCKING=1 python elastic_wave_main_2.py $CONFIG_FILE 5
