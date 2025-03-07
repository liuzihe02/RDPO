#!/bin/bash
set -ex # emable error tracing

export CUDA_VISIBLE_DEVICES=0,1,2,3

#this directory contains many subdirectories. Each subdirectory is a checkpoint directory
model_dir="/home/flowingpurplecrane/RDPO/train/LLaMA-Factory/output_models/qwen2.5-0.5B-cft_WebInstruct-CFT-4K"
summary_path="/home/flowingpurplecrane/RDPO/train/validation/validation_summary.txt"

models_dir_name=$(basename "$model_dir")
summary_parent_dir=$(dirname "$summary_path")

#loop through all the checkpoint directories
for checkpoint_dir in ${model_dir}/checkpoint-*; do
#check if its actually a dictionary
    if [ -d "$checkpoint_dir" ]; then
        #extract the checkpoint number
        checkpoint_num=$(basename "$checkpoint_dir" | cut -d'-' -f2)
        #checkpoint number must be valid
        if [ "$checkpoint_num" -ge 1 ] && [ "$checkpoint_num" -lt 100 ]; then
            #get the outpuit directory
            output_dir="${summary_parent_dir}/${models_dir_name}-checkpoint-${checkpoint_num}/"

            echo "Processing checkpoint-${checkpoint_num}"
            echo "Model path: ${checkpoint_dir}"
            echo "Output dir: ${output_dir}"
            #validate this checkpoint using the provided parameters
            bash validate_single.sh "$checkpoint_dir" "$output_dir" "$summary_path"
        else
            echo "Skipping checkpoint-${checkpoint_num} as it's >= 100"
        fi
    fi
done