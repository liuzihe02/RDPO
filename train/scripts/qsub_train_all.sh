#!/bin/bash

# Request resources
#$ -S /bin/bash
#$ -l tmem=100G
#$ -l gpu=true
#$ -l h_rt=24:00:00
#$ -l gpu_type=h100
#$ -pe gpu 8
#$ -P aihub_ucl
#$ -N train-rdpo-all
#$ -j y

# Print some information about the job
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Starting on: $(date)"
echo "Using GPU devices: $CUDA_VISIBLE_DEVICES"

# Load environment setup
# Source the bashrc to get all the environment variables
# also includes CUDA stuff
source $HOME/.bashrc

# Activate virtual environment
source $HOME/RDPO/venv4/bin/activate

# FROM HERE ON OUT USE THE SCRATCH SPACE
# the scratch env variable should already be defined in the bashrc file
# Navigate to scripts directory
cd $SCRATCH/RDPO/train/scripts

# List of directories that contain their own train.sh
train_dirs=(
    "train-qwen2.5-0.5b-genrm-sft-no_veri"
    "train-qwen2.5-0.5b-genrm-sft-veri"
    "train-qwen2.5-0.5b-genrm-dpo"
    "train-qwen2.5-0.5b-genrm-rdpo"
)

# Save the original working directory
original_dir=$(pwd)

for dir in "${train_dirs[@]}"; do
    echo "=============== RUNNING TRAINING FOR : $dir ==================="
    # Change to appropriate directory
    cd "$dir"
    bash train.sh
    echo "=============== ENDING TRAINING FOR : $dir ==================="
    # Change back to scripts directory
    cd "$original_dir"
done

echo "All training jobs completed on: $(date)"

# Deactivate virtual environment
deactivate
