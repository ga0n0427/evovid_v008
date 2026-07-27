#!/usr/bin/env bash
# V008: grounded Questioner and evidence-grounded Solver, without explicit
# <think> tags. Data and output locations are supplied by environment variables.

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

export MODEL_PATH=${MODEL_PATH:-/home/gpuuser03/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17}
export MODEL_ABBR=${MODEL_ABBR:-V008}
export PREPROCESSED_VIDEO_DIR=${PREPROCESSED_VIDEO_DIR:-/dataset/gpuuser03/evovid_data/V003/data/preprocessed_videos}
export SOLVER_VIDEO_DATA=${SOLVER_VIDEO_DATA:-/dataset/gpuuser03/evovid_data/V003/data/evovid_preprocessed.jsonl}
export STORAGE_PATH=${STORAGE_PATH:-/dataset/gpuuser03/evovid_data/V008_evovid_v008}
export PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
NINJA_BIN=$(dirname "${PYTHON_BIN}")/ninja
if [[ ! -x "${NINJA_BIN}" ]]; then
    echo "Required FlashInfer JIT dependency is missing: ${NINJA_BIN}" >&2
    exit 2
fi
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export VLLM_DISABLE_COMPILE_CACHE=1
export CONFIG_PATH=${PROJECT_DIR}/examples/evovid_main.yaml
export QUESTIONER_PROMPT=${PROJECT_DIR}/examples/format_prompt/questioner_strict.jinja

export NUM_ITERATIONS=${NUM_ITERATIONS:-8}
export QUESTIONER_STEPS=${QUESTIONER_STEPS:-20}
export SOLVER_STEPS=${SOLVER_STEPS:-20}
export QUESTIONER_MAX_RESPONSE_LENGTH=4096
export SOLVER_GENERATION_SAMPLES=${SOLVER_GENERATION_SAMPLES:-5778}
export EVOVID_EXPECTED_ROWS=5778

export QUESTIONER_GROUP_SIZE=8
export SOLVER_GROUP_SIZE=8
export GRPO_PROMPT_BATCH_SIZE=16
export GRPO_MINIBATCH_SIZE=8
export QUESTIONER_SOLVER_SAMPLES=10
export QUESTION_EVALUATE_NUM_SAMPLES=10

export QUESTIONER_TRAIN_GPUS=0,1,2,3
export QUESTIONER_N_GPUS=4
export SERVER_GPUS=4,5,6,7
export SOLVER_TRAIN_GPUS=0,1,2,3,4,5,6,7
export SOLVER_N_GPUS=8

export LORA_RANK=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.05

export WANDB_PROJECT=evovid-main-qwen3vl4b
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_RUN_GROUP=V008-grounded-reasoning-v1-v8
export WANDB_TAGS=evovid,V008,grounded-questioner,evidence-reasoning,segment,no-think,a-d,qwen3-vl-4b

exec bash "${PROJECT_DIR}/scripts/run_evovid_main.sh" "$@"
