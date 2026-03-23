#!/bin/bash

# This script prints the sbatch commands for the top 3 runs for each configuration
# without actually submitting them.

# Load configurations from the provided configurations file
CONFIG_FILE="configurations_top3_runs_each.txt"

# Function to get maximum length among arrays
max_length() {
  max=0
  for array in "$@"; do
    current_length=$(eval "echo \${#$array[@]}")
    if (( current_length > max )); then
      max=$current_length
    fi
  done
  echo $max
}

# Read each line from the configuration file
while IFS=: read -r nn_type n1 n_hidden_layers n_neurons n_hidden_layers_after n_neurons_after; do
  
    # Convert comma-separated strings into arrays
    IFS=',' read -ra n1_array <<< "$n1"
    IFS=',' read -ra n_hidden_layers_array <<< "$n_hidden_layers"
    IFS=',' read -ra n_neurons_array <<< "$n_neurons"
    IFS=',' read -ra n_hidden_layers_after_array <<< "$n_hidden_layers_after"
    IFS=',' read -ra n_neurons_after_array <<< "$n_neurons_after"

    # Get the maximum length among the arrays
    num_runs=$(max_length n1_array n_hidden_layers_array n_neurons_array n_hidden_layers_after_array n_neurons_after_array)

    # For each set of parameters, print the sbatch command
    for (( i=0; i<num_runs; i++ )); do
        current_n1="${n1_array[$i]:-${n1_array[0]}}"
        current_n_hidden_layers="${n_hidden_layers_array[$i]:-${n_hidden_layers_array[0]}}"
        current_n_neurons="${n_neurons_array[$i]:-${n_neurons_array[0]}}"
        current_n_hidden_layers_after="${n_hidden_layers_after_array[$i]:-${n_hidden_layers_after_array[0]:-1}}"  # Default to 1 if not specified
        current_n_neurons_after="${n_neurons_after_array[$i]:-${n_neurons_after_array[0]:-1}}"  # Default to 1 if not specified

        # Print the sbatch command to the console
        echo "sbatch run_multiple_2.sh $current_n1 $current_n_neurons $current_n_hidden_layers $current_n_hidden_layers_after $current_n_neurons_after $nn_type $nn_type"
        # Uncomment the line below to actually submit the job
        sbatch run_multiple_2.sh $current_n1 $current_n_neurons $current_n_hidden_layers $current_n_hidden_layers_after $current_n_neurons_after $nn_type $nn_type
    done
done < "$CONFIG_FILE"
