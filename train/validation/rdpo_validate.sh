#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3

#this directory contains many subdirectories. Each subdirectory is a checkpoint directory
#model_dir="../LLaMA-Factory/output_models/qwen2.5-0.5b-genrm_dpo-2000"
model_dir="/home/flowingpurplecrane/RDPO/train/LLaMA-Factory/output_models/qwen2.5-0.5b-genrm_dpo-2000"
#we will create a results directory here.
output_dir="/home/flowingpurplecrane/RDPO/train/validation"
#number of samples to use for validation
num_samples=10



#get the model name
model_name=$(basename "$model_dir")

# create the results folder in the output_dir
# we temporarily create the name first
results_dir="${output_dir}/results_${model_name}"

# Check if the results_dir already exists
if [ -d "$results_dir" ]; then
    echo "Error: Results directory '$results_dir' already exists. Aborting to prevent overwrite."
    exit 1
else
    #can continue making the results directory
    echo "Results directory '$results_dir' does not exist. Safe to proceed."
    mkdir -p "$results_dir"
fi

# Create summary file
summary_path="${results_dir}/validation_summary.txt"

# Process each checkpoint
for checkpoint_dir in ${model_dir}/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then
        checkpoint_num=$(basename "$checkpoint_dir" | cut -d'-' -f2)
        
        if [ "$checkpoint_num" -ge 1 ] && [ "$checkpoint_num" -lt 100 ]; then
            checkpoint_output="${results_dir}/checkpoint-${checkpoint_num}/"
            
            echo "Processing checkpoint-${checkpoint_num}"
            bash validate_single.sh "$checkpoint_dir" "$checkpoint_output" "$summary_path" "$num_samples"
        else
            echo "Skipping checkpoint-${checkpoint_num} (≥ 100)"
        fi
    fi
done