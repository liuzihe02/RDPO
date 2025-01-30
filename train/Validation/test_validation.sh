
model_dir="/data/yubo/models/Qwen2.5-Math-7B-CFT"
output_dir="../../Validation"
summary_path="../../Validation/validation_summary.txt"

export CUDA_VISIBLE_DEVICES=4,5,6,7
bash validate_single.sh $model_dir $output_dir $summary_path