#!/bin/bash
set -ex

cd ../download_data
# download cft data, output means the current directory
python3 download_cft_data_hf.py --output . --config 4k 50k
#download genrm data
python3 download_genrm_data_hf.py --output . --num_samples 200
