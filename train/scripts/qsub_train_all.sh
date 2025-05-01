#!/bin/bash

# Request resources
#$ -S /bin/bash
#$ -l tmem=100G
#$ -l gpu=true
#$ -l h_rt=24:00:00
#$ -l gpu_type=h100
#$ -pe gpu 2
#$ -P aihub_ucl
#$ -N train-rdpo
#$ -j y
#$ -cwd

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

#change directory to the scripts directory
#our output models is stored in the scratch space
cd $HOME/RDPO/train/scripts

# List of directories that contain their own train.sh
# this is just the folder name itself without any path prefixes
train_dirs=(
    "train-qwen2.5-3b-genrm-sft-veri"
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
