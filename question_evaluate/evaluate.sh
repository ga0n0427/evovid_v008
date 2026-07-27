#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <merged_solver_model> <save_name>" >&2
    exit 2
fi

MODEL_NAME=$1
SAVE_NAME=$2
STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH first}
PYTHON_BIN=${PYTHON_BIN:-python3}
QUESTION_EVALUATE_GPUS=${QUESTION_EVALUATE_GPUS:-${QUESTION_GENERATE_GPUS:-0,1,2,3,4,5,6,7}}
QUESTION_EVALUATE_NUM_SAMPLES=${QUESTION_EVALUATE_NUM_SAMPLES:-10}
QUESTION_EVALUATE_BATCH_SIZE=${QUESTION_EVALUATE_BATCH_SIZE:-8}
QUESTION_EVALUATE_MAX_TOKENS=${QUESTION_EVALUATE_MAX_TOKENS:-4096}
QUESTION_EVALUATE_MAX_MODEL_LEN=${QUESTION_EVALUATE_MAX_MODEL_LEN:-8192}
QUESTION_EVALUATE_GPU_MEMORY=${QUESTION_EVALUATE_GPU_MEMORY:-0.8}

IFS=',' read -r -a GPU_IDS <<< "${QUESTION_EVALUATE_GPUS}"
if (( ${#GPU_IDS[@]} == 0 )); then
    echo "QUESTION_EVALUATE_GPUS must contain at least one GPU id." >&2
    exit 2
fi

export VLLM_DISABLE_COMPILE_CACHE=1
pids=()
for shard_index in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$shard_index]}
    input_file="${STORAGE_PATH}/generated_question/${SAVE_NAME}_${shard_index}.json"
    output_file="${STORAGE_PATH}/generated_question/${SAVE_NAME}_${shard_index}_results.json"
    if [[ ! -f "${input_file}" ]]; then
        echo "Generated-question shard not found: ${input_file}" >&2
        exit 1
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m question_evaluate.evaluate \
        --model "${MODEL_NAME}" \
        --save_name "${SAVE_NAME}" \
        --suffix "${shard_index}" \
        --input_file "${input_file}" \
        --output_file "${output_file}" \
        --num_samples "${QUESTION_EVALUATE_NUM_SAMPLES}" \
        --batch_size "${QUESTION_EVALUATE_BATCH_SIZE}" \
        --max_tokens "${QUESTION_EVALUATE_MAX_TOKENS}" \
        --max_model_len "${QUESTION_EVALUATE_MAX_MODEL_LEN}" \
        --gpu_memory_utilization "${QUESTION_EVALUATE_GPU_MEMORY}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
