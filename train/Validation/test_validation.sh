
model_dir="/data/yubo/models/Qwen2.5-Math-7B-CFT"
summary_path="../../Validation/validation_summary.txt"

export CUDA_VISIBLE_DEVICES=0,1,2,3
bash validate_single.sh $model_dir $summary_path