#!/bin/bash
set -e

# Take model_dir as first argument
model_dir="$1"
data="$2"
num_samples="$3"

if [ -z "$model_dir" ] || [ -z "$data" ] || [ -z "$num_samples" ]; then
    echo "Usage of this script: $0 <model_dir> <data> <num_samples>"
    exit 1
fi

#we will create a results directory in the results folder
output_dir="../evaluate-results"

# these are evaluation scripts for Critique-Fine-Tuning

# cd ../evaluate_math/scripts
# bash evaluate_qwen.sh ${model_path} ${output_dir} ${summary_path}

# cd ../../evaluate_gpqa/scripts
# bash evaluate_gpqa.sh ${model_path} ${output_dir} ${summary_path}

# cd ../../evaluate_mmlu-pro
# bash mmlu-pro-eval.sh ${model_path} ${output_dir} ${summary_path} 0

#get the model name - like train-qwen2.5-0.5b-genrm-rdpo
run=$(basename "$model_dir")

# create the results folder in the output_dir
# we temporarily create the name first
results_dir="${output_dir}/${run}-eval-${data}"

# Check if the results_dir already exists
if [ -d "$results_dir" ]; then
    echo "WARNING: Results directory '$results_dir' already exists. Overwriting..."
    rm -rf "$results_dir"  # Remove the existing directory
    mkdir -p "$results_dir"  # Create a fresh directory
else
    # Create the results directory if it doesn't exist
    echo "Results directory '$results_dir' does not exist. Creating it now."
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
        # EVALUATION SETTINGS

        #merge lora adapters with base model - so vllm interprets them as one whole model
        python3 merge_adapters.py -c "$checkpoint_dir"
        
        bash validate_single.sh "$checkpoint_dir" "$checkpoint_output" "$summary_path" "$data" "$num_samples"
    fi
done

