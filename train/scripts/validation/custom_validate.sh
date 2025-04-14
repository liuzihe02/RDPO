#!/bin/bash
set -e

#this file does validation on the math500 dataset

export CUDA_VISIBLE_DEVICES=1

#this directory contains many subdirectories. Each subdirectory is a checkpoint directory
model_dir="../../LLaMA-Factory/output_models/train-qwen2.5-0.5b-genrm-sft-no_veri"
#we will create a results directory in the current folder
output_dir="./"
#number of samples to use for validation
num_samples=500

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
        checkpoint_output="${results_dir}/checkpoint-${checkpoint_num}/"
        
        echo "Processing checkpoint-${checkpoint_num}"
        bash validate_single.sh "$checkpoint_dir" "$checkpoint_output" "$summary_path" "$num_samples"
    fi
done
