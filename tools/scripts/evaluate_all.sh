#!/bin/bash
set -e

#make sure this is the right number!
export CUDA_VISIBLE_DEVICES=0

#each directory here contains many subdirectories. Each subdirectory is a checkpoint directory
# we evaluate ALL THESE runs all at one go
model_dirs=(
    "../../train/LLaMA-Factory/output_models/train-qwen2.5-3b-genrm-sft-no_veri"
)

#what data to evaluate on; these are available
#"math,math-500,minerva_math,gsm8k,olympiadbench,amc23,aime24,theoremqa"
data="gsm8k,math-500"
#number of samples to use for validation
#if multiple datasets are provided, then we take num_samples from EACH dataset
num_samples=500

for model_dir in "${model_dirs[@]}"; do
    echo "=============== RUNNING VALIDATION FOR : $model_dir ==================="
    bash evaluate_single.sh "$model_dir" "$data" "$num_samples"
    echo "=============== ENDING VALIDATION FOR : $model_dir ==================="
done
