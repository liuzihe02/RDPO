#!/bin/bash
set -ex

cd ../download_data
# which datasets to download
python download_cft_data_hf.py --config 4k 50k
