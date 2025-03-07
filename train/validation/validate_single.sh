set -ex

PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
SUMMARY_PATH=$3
SPLIT="test"
#this is an important parameter which decides how many test samples we want
NUM_TEST_SAMPLE=10

mkdir -p $OUTPUT_DIR
cd ../../tools/evaluate_math

DATA_NAME="math-500"
TOKENIZERS_PARALLELISM=false \
#the max tokens is an important parameter
#we also added use_safetensors
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --summary_path ${SUMMARY_PATH} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --max_tokens_per_call 1024 \
    --save_outputs \
    --use_safetensors

