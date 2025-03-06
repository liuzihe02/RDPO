#!/bin/bash

#change directory
cd ../../LLaMA-Factory
PROJECT_NAME="RDPO"
#make all gpus avaiable for training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=f318ffd0dcf5d31701fd33aee12e57e9cf15444f
export WANDB_PROJECT=$PROJECT_NAME
#disable WANDB
export WANDB_MODE=disabled
#randomly select a port number for distributed training config
MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export MASTER_PORT
#Configures PyTorch's CUDA memory allocation to use expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="rdpo"
export WANDB_RUN_NAME=$MODEL_NAME

FORCE_TORCHRUN=1 llamafactory-cli train ../scripts/train_qwen2_5-0.5b-instruct-genrm_dpo/qwen2.5-0.5b-genrm_dpo-400.yaml
