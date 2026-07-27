#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <merged_questioner_model> <total_samples> <save_name> [video_data]" >&2
    exit 2
fi

MODEL_NAME=$1
TOTAL_SAMPLES=$2
SAVE_NAME=$3
VIDEO_DATA=${4:-${SOLVER_VIDEO_DATA:-}}

STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH first}
if [[ -z "${VIDEO_DATA}" ]]; then
    echo "Set SOLVER_VIDEO_DATA or pass a JSON/JSONL video dataset as the fourth argument." >&2
    exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
QUESTIONER_PROMPT=${QUESTIONER_PROMPT:-examples/format_prompt/questioner_strict.jinja}
QUESTION_GENERATE_GPUS=${QUESTION_GENERATE_GPUS:-0,1,2,3,4,5,6,7}
QUESTION_WINDOW_SIZE=${QUESTION_WINDOW_SIZE:-8}
QUESTION_WINDOW_STARTS=${QUESTION_WINDOW_STARTS:-0,1,2,3,4,5,6,7,8}
QUESTION_GENERATE_SEED=${QUESTION_GENERATE_SEED:-1}
QUESTION_GENERATE_BATCH_SIZE=${QUESTION_GENERATE_BATCH_SIZE:-16}
QUESTION_GENERATE_MAX_TOKENS=${QUESTION_GENERATE_MAX_TOKENS:-1024}
QUESTION_GENERATE_MAX_MODEL_LEN=${QUESTION_GENERATE_MAX_MODEL_LEN:-8192}
QUESTION_GENERATE_GPU_MEMORY=${QUESTION_GENERATE_GPU_MEMORY:-0.8}

IFS=',' read -r -a GPU_IDS <<< "${QUESTION_GENERATE_GPUS}"
if (( ${#GPU_IDS[@]} == 0 )); then
    echo "QUESTION_GENERATE_GPUS must contain at least one GPU id." >&2
    exit 2
fi

OUTPUT_DIR="${STORAGE_PATH}/generated_question"
MANIFEST_DIR="${OUTPUT_DIR}/manifests/${SAVE_NAME}"
mkdir -p "${OUTPUT_DIR}" "${MANIFEST_DIR}"

MANIFEST_CMD=(
    "${PYTHON_BIN}" -m question_generate.build_window_manifest
    --video_data "${VIDEO_DATA}"
    --output_dir "${MANIFEST_DIR}"
    --save_name "${SAVE_NAME}"
    --total_samples "${TOTAL_SAMPLES}"
    --window_size "${QUESTION_WINDOW_SIZE}"
    --window_starts "${QUESTION_WINDOW_STARTS}"
    --num_shards "${#GPU_IDS[@]}"
    --seed "${QUESTION_GENERATE_SEED}"
)
if [[ -n "${PREPROCESSED_VIDEO_DIR:-}" ]]; then
    MANIFEST_CMD+=(--preprocessed_video_dir "${PREPROCESSED_VIDEO_DIR}")
fi
"${MANIFEST_CMD[@]}"

export VLLM_DISABLE_COMPILE_CACHE=1
pids=()
for shard_index in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$shard_index]}
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -m question_generate.question_generate \
        --model "${MODEL_NAME}" \
        --prompt_template "${QUESTIONER_PROMPT}" \
        --manifest "${MANIFEST_DIR}/${SAVE_NAME}_${shard_index}.jsonl" \
        --output "${OUTPUT_DIR}/${SAVE_NAME}_${shard_index}.json" \
        --batch_size "${QUESTION_GENERATE_BATCH_SIZE}" \
        --max_tokens "${QUESTION_GENERATE_MAX_TOKENS}" \
        --max_model_len "${QUESTION_GENERATE_MAX_MODEL_LEN}" \
        --gpu_memory_utilization "${QUESTION_GENERATE_GPU_MEMORY}" \
        --seed "$((QUESTION_GENERATE_SEED + shard_index))" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
