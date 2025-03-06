#!/bin/bash
set -ex

cd ../download_data
# download cft data, output means the current directory
python3 download_cft_data_hf.py --output "../../train/LLaMA-Factory/data" --config 4k 50k
#download genrm data
python3 download_genrm_data_hf.py --output "../../train/LLaMA-Factory/data" --num_samples 200
