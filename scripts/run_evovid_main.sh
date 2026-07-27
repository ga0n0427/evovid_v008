#!/usr/bin/env bash
# Thin launcher for the EvoVid paper recipe; use --check for setup validation.

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}
MODEL_ABBR=${MODEL_ABBR:-qwen3vl4b_evovid}

export PREPROCESSED_VIDEO_DIR=${PREPROCESSED_VIDEO_DIR:?Set PREPROCESSED_VIDEO_DIR}
export SOLVER_VIDEO_DATA=${SOLVER_VIDEO_DATA:?Set SOLVER_VIDEO_DATA}
export STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH}

export PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
export CONFIG_PATH=${CONFIG_PATH:-${PROJECT_DIR}/examples/evovid_main.yaml}
export QUESTIONER_PROMPT=${QUESTIONER_PROMPT:-${PROJECT_DIR}/examples/format_prompt/questioner_strict.jinja}

export NUM_ITERATIONS=${NUM_ITERATIONS:-3}
export QUESTIONER_STEPS=${QUESTIONER_STEPS:-20}
export SOLVER_STEPS=${SOLVER_STEPS:-20}
export SOLVER_GENERATION_SAMPLES=${SOLVER_GENERATION_SAMPLES:-5778}
export QUESTIONER_GROUP_SIZE=${QUESTIONER_GROUP_SIZE:-8}
export SOLVER_GROUP_SIZE=${SOLVER_GROUP_SIZE:-8}
export GRPO_PROMPT_BATCH_SIZE=${GRPO_PROMPT_BATCH_SIZE:-16}
export QUESTIONER_SOLVER_SAMPLES=${QUESTIONER_SOLVER_SAMPLES:-10}
export QUESTION_EVALUATE_NUM_SAMPLES=${QUESTION_EVALUATE_NUM_SAMPLES:-10}
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-64}
export LORA_DROPOUT=${LORA_DROPOUT:-0.05}

export WANDB_PROJECT=${WANDB_PROJECT:-evovid-main-qwen3vl4b}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${MODEL_ABBR}-paper-recipe}
export WANDB_TAGS=${WANDB_TAGS:-evovid,paper-2605.21931,qwen3-vl-4b}

if [[ "${1:-}" == "--check" ]]; then
    export EVOVID_SETUP_ONLY=1
    shift
fi
if (( $# != 0 )); then
    echo "Usage: bash scripts/run_evovid_main.sh [--check]" >&2
    exit 2
fi

exec bash "${PROJECT_DIR}/scripts/main.sh" "${MODEL_PATH}" "${MODEL_ABBR}"
