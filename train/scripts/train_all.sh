#!/bin/bash
set -e

# List of directories that contain their own train.sh
train_dirs=(
    "test_dpo"
    "test_rdpo"
    "test_sft"
)

# Save the original working directory
original_dir=$(pwd)

for dir in "${train_dirs[@]}"; do
    echo "=============== RUNNING TRAINING FOR : $dir ==================="
    #change to appropriate directory
    cd "$dir"
    bash train.sh
    echo "=============== ENDING TRAINING FOR : $dir ==================="
    #change back to scripts directory
    cd "$original_dir"
done
