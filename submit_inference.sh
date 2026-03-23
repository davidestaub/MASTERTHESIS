#!/bin/bash

# Loop from 0 to 30
for i in {0..40}; do
    # Calculate the next number
    next=$((i + 1))
    
    # Submit the job with sbatch
    sbatch run.sh "$i" "$next"
done
