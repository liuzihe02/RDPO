# #!/bin/bash
# set -e

# #make sure this is the right number!
# export CUDA_VISIBLE_DEVICES=0

# #each directory here contains many subdirectories. Each subdirectory is a checkpoint directory
# # we evaluate ALL THESE runs all at one go
# model_dirs=(
#     "../../train/LLaMA-Factory/output_models/train-qwen2.5-3b-genrm-sft-no_veri"
# )

# #what data to evaluate on; these are available
# #"math,math-500,minerva_math,gsm8k,olympiadbench,amc23,aime24,theoremqa"
# data="gsm8k,math-500"
# #number of samples to use for validation
# #if multiple datasets are provided, then we take num_samples from EACH dataset
# num_samples=500

# for model_dir in "${model_dirs[@]}"; do
#     echo "=============== RUNNING VALIDATION FOR : $model_dir ==================="
#     bash evaluate_single.sh "$model_dir" "$data" "$num_samples"
#     echo "=============== ENDING VALIDATION FOR : $model_dir ==================="
# done

#this scripts launches one eval per gpu; concurrently
#!/bin/bash
set -e

# List all gpus available
cuda_devices=(0)

# Each directory here contains many subdirectories with checkpoints
model_dirs=(
    "${SCRATCH}/output_models/train-qwen2.5-3b-genrm-sft-no_veri"
# "../../train/LLaMA-Factory/output_models/train-qwen2.5-0.5b-genrm-sft-no_veri"    
)

# Evaluation datasets
data="gsm8k,math-500"
num_samples=500

# Assert that number of model directories matches number of GPUs
if [ ${#model_dirs[@]} -ne ${#cuda_devices[@]} ]; then
    echo "Error: Number of model directories (${#model_dirs[@]}) must match number of GPUs (${#cuda_devices[@]})"
    exit 1
fi

# Launch one evaluation per GPU
for i in "${!model_dirs[@]}"; do
    gpu_id=${cuda_devices[$i]}
    
    # Run each evaluation in background with specific GPU
    (
        export CUDA_VISIBLE_DEVICES=$gpu_id
        echo "=============== RUNNING VALIDATION FOR: ${model_dirs[$i]} on GPU $gpu_id ==================="
        bash evaluate_single.sh "${model_dirs[$i]}" "$data" "$num_samples"
        echo "=============== ENDING VALIDATION FOR: ${model_dirs[$i]} on GPU $gpu_id ==================="
    ) &
    
    # Optional: Small delay to prevent launch conflicts
    sleep 0.5
done

# Wait for all background processes to complete
wait
echo "All evaluations completed"
