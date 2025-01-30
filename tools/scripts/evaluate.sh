#!/bin/bash
set -ex

model_path="/path/to/model"
output_dir="../evaluation_output"
summary_path="../evaluation_summary.txt"

export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ../evaluate_math/scripts
bash evaluate_qwen.sh ${model_path} ${output_dir} ${summary_path}

cd ../../evaluate_gpqa/scripts
bash evaluate_gpqa.sh ${model_path} ${output_dir} ${summary_path}

cd ../../evaluate_mmlu-pro
bash mmlu-pro-eval.sh ${model_path} ${output_dir} ${summary_path} 0
