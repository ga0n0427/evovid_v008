#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_DIR}"

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/main.sh <base_model> <model_abbr>" >&2
    exit 2
fi

BASE_MODEL=$1
MODEL_ABBR=$2

STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH first}
PREPROCESSED_VIDEO_DIR=${PREPROCESSED_VIDEO_DIR:?Set PREPROCESSED_VIDEO_DIR first}
SOLVER_VIDEO_DATA=${SOLVER_VIDEO_DATA:?Set SOLVER_VIDEO_DATA first}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
PYTHON_ENV_BIN=$(dirname "${PYTHON_BIN}")
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_DIR}/examples/evovid_main.yaml}
QUESTIONER_PROMPT=${QUESTIONER_PROMPT:-${PROJECT_DIR}/examples/format_prompt/questioner_strict.jinja}
QUESTIONER_DATA=${QUESTIONER_DATA:-${STORAGE_PATH}/data/evovid_questioner_train.jsonl}
QUESTIONER_ITERATION_DATA_DIR=${QUESTIONER_ITERATION_DATA_DIR:-${STORAGE_PATH}/data/questioner_iterations}
QUESTIONER_DATA_SEED=${QUESTIONER_DATA_SEED:-1}
EVOVID_EXPECTED_ROWS=${EVOVID_EXPECTED_ROWS:-5778}
EVOVID_SETUP_ONLY=${EVOVID_SETUP_ONLY:-0}

NUM_ITERATIONS=${NUM_ITERATIONS:-3}
QUESTIONER_STEPS=${QUESTIONER_STEPS:-20}
SOLVER_STEPS=${SOLVER_STEPS:-20}
SOLVER_GENERATION_SAMPLES=${SOLVER_GENERATION_SAMPLES:-5778}

LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
QUESTIONER_GROUP_SIZE=${QUESTIONER_GROUP_SIZE:-8}
SOLVER_GROUP_SIZE=${SOLVER_GROUP_SIZE:-8}
GRPO_PROMPT_BATCH_SIZE=${GRPO_PROMPT_BATCH_SIZE:-16}
GRPO_MINIBATCH_SIZE=${GRPO_MINIBATCH_SIZE:-${GRPO_PROMPT_BATCH_SIZE}}
QUESTIONER_SOLVER_SAMPLES=${QUESTIONER_SOLVER_SAMPLES:-10}
QUESTION_EVALUATE_NUM_SAMPLES=${QUESTION_EVALUATE_NUM_SAMPLES:-10}

WANDB_PROJECT=${WANDB_PROJECT:-evovid-main-qwen3vl4b}
WANDB_MODE=${WANDB_MODE:-online}
WANDB_DIR=${WANDB_DIR:-${STORAGE_PATH}/wandb}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${MODEL_ABBR}-paper-recipe}
WANDB_TAGS=${WANDB_TAGS:-evovid,paper-2605.21931,qwen3-vl-4b}

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${PYTHON_ENV_BIN}:${PATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_DISABLE_COMPILE_CACHE=1
export CONFIG_PATH PREPROCESSED_VIDEO_DIR SOLVER_VIDEO_DATA QUESTIONER_DATA QUESTIONER_PROMPT
export QUESTIONER_STEPS SOLVER_STEPS SOLVER_GENERATION_SAMPLES
export LORA_RANK LORA_ALPHA LORA_DROPOUT
export QUESTIONER_GROUP_SIZE SOLVER_GROUP_SIZE GRPO_PROMPT_BATCH_SIZE GRPO_MINIBATCH_SIZE
export QUESTIONER_SOLVER_SAMPLES QUESTION_EVALUATE_NUM_SAMPLES
export WANDB_PROJECT WANDB_MODE WANDB_DIR WANDB_RUN_GROUP WANDB_TAGS PYTHON_BIN

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "evovid Python not found or not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
for required_path in "${CONFIG_PATH}" "${QUESTIONER_PROMPT}" "${SOLVER_VIDEO_DATA}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 2
    fi
done

"${PYTHON_BIN}" -c '
import peft, torch, transformers, wandb
assert peft.__version__ == "0.18.1", f"Expected peft 0.18.1, found {peft.__version__}"
print(f"environment: peft={peft.__version__}, torch={torch.__version__}, transformers={transformers.__version__}, wandb={wandb.__version__}")
'

PREPARE_CMD=(
    "${PYTHON_BIN}" -m scripts.prepare_evovid_main_data
    --input "${SOLVER_VIDEO_DATA}"
    --preprocessed_video_dir "${PREPROCESSED_VIDEO_DIR}"
    --expected_count "${EVOVID_EXPECTED_ROWS}"
)

if [[ "${EVOVID_SETUP_ONLY}" == "1" ]]; then
    "${PREPARE_CMD[@]}" --validate_only
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    if (( gpu_count < 8 )); then
        echo "Expected 8 GPUs, found ${gpu_count}." >&2
        exit 2
    fi
    if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
        echo "W&B online auth is not configured yet; run: ${PYTHON_BIN%/python}/wandb login" >&2
    fi
    effective_batch=$((GRPO_PROMPT_BATCH_SIZE * QUESTIONER_GROUP_SIZE))
    effective_minibatch=$((GRPO_MINIBATCH_SIZE * QUESTIONER_GROUP_SIZE))
    echo "setup_ok: iterations=${NUM_ITERATIONS}, q_steps=${QUESTIONER_STEPS}, s_steps=${SOLVER_STEPS}, group=${QUESTIONER_GROUP_SIZE}, effective_batch=${effective_batch}, effective_minibatch=${effective_minibatch}, gpus=${gpu_count}"
    exit 0
fi

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
    echo "W&B online auth is required. Run: ${PYTHON_BIN%/python}/wandb login" >&2
    exit 2
fi

mkdir -p "${STORAGE_PATH}" "${WANDB_DIR}" "${STORAGE_PATH}/reward_audit"
export EVOVID_REWARD_AUDIT_DIR=${EVOVID_REWARD_AUDIT_DIR:-${STORAGE_PATH}/reward_audit}
"${PREPARE_CMD[@]}" --output "${QUESTIONER_DATA}"

QUESTIONER_SAMPLES_PER_ITERATION=$((QUESTIONER_STEPS * GRPO_PROMPT_BATCH_SIZE))
"${PYTHON_BIN}" -m scripts.prepare_questioner_iteration_data \
    --input "${QUESTIONER_DATA}" \
    --output_dir "${QUESTIONER_ITERATION_DATA_DIR}" \
    --num_iterations "${NUM_ITERATIONS}" \
    --samples_per_iteration "${QUESTIONER_SAMPLES_PER_ITERATION}" \
    --seed "${QUESTIONER_DATA_SEED}"

MODEL_ROOT="${STORAGE_PATH}/models"

# Iteration 1 starts new LoRA adapters from the immutable Base model.
# Later iterations replace these empty values with the previous role's adapter.
QUESTIONER_ADAPTER=""
SOLVER_ADAPTER=""
SOLVER_MERGED="${BASE_MODEL}"

for ((i = 1; i <= NUM_ITERATIONS; i++)); do
    QUESTIONER_EXPERIMENT="${MODEL_ABBR}_questioner_v${i}"
    SOLVER_EXPERIMENT="${MODEL_ABBR}_solver_v${i}"
    QUESTIONER_ITERATION_DATA=$(printf "%s/iteration_%03d.jsonl" "${QUESTIONER_ITERATION_DATA_DIR}" "${i}")

    echo "=== Iteration ${i}: Questioner ==="
    QUESTIONER_DATA="${QUESTIONER_ITERATION_DATA}" bash scripts/questioner_train_penalty.sh \
        "${BASE_MODEL}" \
        "${QUESTIONER_ADAPTER}" \
        "${SOLVER_MERGED}" \
        "${QUESTIONER_EXPERIMENT}"

    QUESTIONER_ACTOR="${MODEL_ROOT}/${QUESTIONER_EXPERIMENT}/global_step_${QUESTIONER_STEPS}/actor"
    QUESTIONER_ADAPTER="${QUESTIONER_ACTOR}/lora_adapter"
    QUESTIONER_MERGED="${QUESTIONER_ACTOR}/huggingface"

    echo "=== Iteration ${i}: Solver ==="
    bash scripts/solver_train.sh \
        "${BASE_MODEL}" \
        "${SOLVER_ADAPTER}" \
        "${QUESTIONER_MERGED}" \
        "${SOLVER_MERGED}" \
        "${SOLVER_EXPERIMENT}"

    SOLVER_ACTOR="${MODEL_ROOT}/${SOLVER_EXPERIMENT}/global_step_${SOLVER_STEPS}/actor"
    SOLVER_ADAPTER="${SOLVER_ACTOR}/lora_adapter"
    SOLVER_MERGED="${SOLVER_ACTOR}/huggingface"
done
