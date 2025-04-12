#!/bin/bash

#change directory
cd ../../LLaMA-Factory
PROJECT_NAME="RDPO"
#make all gpus avaiable for training
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export WANDB_API_KEY=f318ffd0dcf5d31701fd33aee12e57e9cf15444f
export WANDB_PROJECT=$PROJECT_NAME
#disable WANDB
export WANDB_MODE=disabled
#randomly select a port number for distributed training config
MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export MASTER_PORT
#Configures PyTorch's CUDA memory allocation to use expandable segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="dpo"
export WANDB_RUN_NAME=$MODEL_NAME

# Get config file path
CONFIG_FILE="../scripts/train-qwen2.5-0.5b-genrm-dpo/train.yaml"

# Extract output directory from yaml file using grep with a pattern that anchors to the beginning of the line
OUTPUT_DIR=$(grep -m1 "^output_dir:" "$CONFIG_FILE" | cut -d':' -f2 | tr -d ' ')

# Check if output directory exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory '$OUTPUT_DIR' already exists. Aborting to prevent overwrite."
    exit 1
else
    echo "Output directory '$OUTPUT_DIR' does not exist. Safe to proceed."
fi

# Continue with training if the directory doesn't exist
FORCE_TORCHRUN=1 llamafactory-cli train $CONFIG_FILE
