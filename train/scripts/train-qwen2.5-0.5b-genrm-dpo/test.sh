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

# Get config file path
CONFIG_FILE="../scripts/train-qwen2.5-0.5b-genrm-dpo/qwen2.5-0.5b-genrm-dpo.yaml"

# Extract output directory from yaml file using grep with a pattern that anchors to the beginning of the line
OUTPUT_DIR=$(grep -m1 "^output_dir:" "$CONFIG_FILE" | cut -d':' -f2 | tr -d ' ')

# Check if output directory exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory '$OUTPUT_DIR' already exists. Aborting to prevent overwrite."
    exit 1
else
    echo "Output directory '$OUTPUT_DIR' does not exist. Safe to proceed."
fi

# Check memory status
echo "Current memory status:"
free -h

# Debug settings
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# Run with glibc path and use python3 with output logging
LD_LIBRARY_PATH=/share/apps/glibc-2.28/lib:$LD_LIBRARY_PATH \
FORCE_TORCHRUN=1 \
python3 -m llamafactory.cli.llamafactory_cli train $CONFIG_FILE --verbose true 2>&1 | tee training_output.log
```

testing script
```bash
#!/bin/bash
#$ -S /bin/bash
#$ -l tmem=4G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -cwd
#$ -j y
#$ -N test_llamafactory
#$ -l h_rt=00:30:00
#$ -P aihub_ucl

# Stop on first error, print commands as they run:
set -euxo pipefail

# 1) Load or source your desired Python environment.
#    Example: using the system Python 3.11.9 from /share/apps:
source /share/apps/source_files/python/python-3.11.9.source

# (Alternatively, if you have a venv:)
# source /path/to/your_venv/bin/activate

# 2) Print debugging info:
which python3
python3 --version

# 3) Simple test script:
python3 <<EOF
import torch
print("PyTorch version:", torch.__version__)
print("Is CUDA available? ", torch.cuda.is_available())

try:
    import llamafactory
    print("Successfully imported LLaMA Factory!")
except ImportError as e:
    print("Failed to import LLaMA Factory:", e)
EOF
