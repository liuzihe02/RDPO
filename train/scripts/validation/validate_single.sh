set -e

#these paths are all relative to the current working directory
PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
SUMMARY_PATH=$3
SPLIT="test"
#this is an important parameter which decides how many test samples we want
NUM_TEST_SAMPLE=$4

mkdir -p $OUTPUT_DIR

# Record cur directory, before changing to math folder
CURRENT_DIR=$(pwd)

#cd to the evaluate_math folder
cd ../../../tools/evaluate_math

DATA_NAME="math-500"
TOKENIZERS_PARALLELISM=false \
# need to use absolute paths here as we've changed directory

#remember to activate vllm

#max_tokens_per_call is how many we allow to generate; default 2048
#this is also how many we allow as input

#we also added use_safetensors
python3 -u math_eval.py \
    --model_name_or_path ${CURRENT_DIR}/${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${CURRENT_DIR}/${OUTPUT_DIR}  \
    --summary_path ${CURRENT_DIR}/${SUMMARY_PATH}\
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --max_tokens_per_call 2048 \
    --save_outputs \
    --use_safetensors


