set -ex

model_path=$1
output_dir=$2
summary_path=$3
n_shot=$4

python evaluate_from_local.py \
    --ntrain $n_shot \
    --model $model_path \
    --save_dir $output_dir \
    --global_record_file $summary_path

