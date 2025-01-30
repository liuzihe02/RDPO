#!/bin/bash

cd ../../LLaMA-Factory
PROJECT_NAME="CFT"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=7
export WANDB_API_KEY=f318ffd0dcf5d31701fd33aee12e57e9cf15444f
export WANDB_PROJECT=$PROJECT_NAME
export WANDB_MODE=disabled
MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export MASTER_PORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="critique-model"
export WANDB_RUN_NAME=$MODEL_NAME

FORCE_TORCHRUN=1 llamafactory-cli train ../scripts/train_qwen2_5-math-7b-cft/qwen2.5-math-7b-cft-webinstruct-50k.yaml
