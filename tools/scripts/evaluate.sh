#!/bin/bash
set -e

#this file does validation on the math500 dataset

#make sure this is the right number!
export CUDA_VISIBLE_DEVICES=0

#this directory contains many subdirectories. Each subdirectory is a checkpoint directory
model_dir="../../train/LLaMA-Factory/output_models/train-qwen2.5-0.5b-genrm-sft-no_veri"

#we will create a results directory in the results folder
output_dir="../evaluate-results"

#what data to evaluate on; these are available
#"math,math-500,minerva_math,gsm8k,olympiadbench,amc23,aime24,theoremqa"
data="gsm8k,math-500"
#number of samples to use for validation
#if multiple datasets are provided, then we take num_samples from EACH dataset
num_samples=500

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
        # EVALUATION SETTINGS

        #merge lora adapters with base model - so vllm interprets them as one whole model
        python3 merge_adapters.py -c "$checkpoint_dir"
        
        bash validate_single.sh "$checkpoint_dir" "$checkpoint_output" "$summary_path" "$data" "$num_samples"
    fi
done

