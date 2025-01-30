set -ex

model_path=$1
output_dir=$2
summary_path=$3

datasets=("gpqa")

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python run_open.py \
        --model $model_path \
        --dataset $dataset \
        --form qwen \
        --output_dir $output_dir \
        --summary_path $summary_path
done

