#!/usr/bin/env bash
# Usage:
#   bash scripts/solver_train.sh <base_model> <previous_solver_adapter_or_empty> \
#       <questioner_merged_model> <previous_solver_merged_model> <experiment_name>

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_DIR}"

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <base_model> <previous_solver_adapter_or_empty> <questioner_merged_model> <previous_solver_merged_model> <experiment_name>" >&2
    exit 2
fi

BASE_MODEL=$1
SOLVER_INIT_ADAPTER=${2:-}
QUESTIONER_MERGED_MODEL=$3
SOLVER_EVAL_MERGED_MODEL=$4
EXPERIMENT_NAME=$5

STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH first}
PREPROCESSED_VIDEO_DIR=${PREPROCESSED_VIDEO_DIR:?Set PREPROCESSED_VIDEO_DIR first}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_DIR}/examples/evovid_main.yaml}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_EXCLUDE_MODULES=${LORA_EXCLUDE_MODULES:-}
SOLVER_STEPS=${SOLVER_STEPS:-20}
SOLVER_GENERATION_SAMPLES=${SOLVER_GENERATION_SAMPLES:-5778}
SOLVER_FORMAT_WEIGHT=${SOLVER_FORMAT_WEIGHT:-0.1}
SOLVER_TEMPORAL_WEIGHT=${SOLVER_TEMPORAL_WEIGHT:-0.3}
SOLVER_CURATION_MIN=${SOLVER_CURATION_MIN:-0.3}
SOLVER_CURATION_MAX=${SOLVER_CURATION_MAX:-0.8}
SOLVER_VIDEO_DATA=${SOLVER_VIDEO_DATA:?Set SOLVER_VIDEO_DATA to the preprocessed video JSON/JSONL file}
SOLVER_GROUP_SIZE=${SOLVER_GROUP_SIZE:-8}
GRPO_PROMPT_BATCH_SIZE=${GRPO_PROMPT_BATCH_SIZE:-16}
GRPO_MINIBATCH_SIZE=${GRPO_MINIBATCH_SIZE:-${GRPO_PROMPT_BATCH_SIZE}}
SOLVER_N_GPUS=${SOLVER_N_GPUS:-8}
SOLVER_TRAIN_GPUS=${SOLVER_TRAIN_GPUS:-0,1,2,3,4,5,6,7}
WANDB_PROJECT=${WANDB_PROJECT:-evovid-main-qwen3vl4b}

SAVE_ROOT="${STORAGE_PATH}/models/${EXPERIMENT_NAME}"
ACTOR_PATH="${SAVE_ROOT}/global_step_${SOLVER_STEPS}/actor"
SOLVER_TRAIN_FILE="${STORAGE_PATH}/generated_question/${EXPERIMENT_NAME}_train.json"

export VLLM_DISABLE_COMPILE_CACHE=1

echo "Generate questions with merged Questioner: ${QUESTIONER_MERGED_MODEL}"
bash question_generate/question_generate.bash \
    "${QUESTIONER_MERGED_MODEL}" \
    "${SOLVER_GENERATION_SAMPLES}" \
    "${EXPERIMENT_NAME}" \
    "${SOLVER_VIDEO_DATA}"

echo "Create pseudo labels with merged previous Solver: ${SOLVER_EVAL_MERGED_MODEL}"
bash question_evaluate/evaluate.sh \
    "${SOLVER_EVAL_MERGED_MODEL}" "${EXPERIMENT_NAME}"

"${PYTHON_BIN}" -m question_evaluate.upload \
    --max_score "${SOLVER_CURATION_MAX}" \
    --min_score "${SOLVER_CURATION_MIN}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --output_file "${SOLVER_TRAIN_FILE}"

"${PYTHON_BIN}" -c 'import json, pathlib, sys; rows=json.loads(pathlib.Path(sys.argv[1]).read_text()); print(f"Curated Solver rows: {len(rows)}"); sys.exit(0 if rows else 1)' "${SOLVER_TRAIN_FILE}"

echo "Train Solver from Base + adapter: ${SOLVER_INIT_ADAPTER:-<new LoRA>}"
TRAIN_CMD=(
    "${PYTHON_BIN}" -m verl.trainer.main
    config="${CONFIG_PATH}"
    data.train_files="${SOLVER_TRAIN_FILE}"
    data.val_files="${SOLVER_TRAIN_FILE}"
    data.preprocessed_video_dir="${PREPROCESSED_VIDEO_DIR}"
    data.val_preprocessed_video_dir="${PREPROCESSED_VIDEO_DIR}"
    data.max_response_length=4096
    data.rollout_batch_size="${GRPO_PROMPT_BATCH_SIZE}"
    worker.actor.model.model_path="${BASE_MODEL}"
    worker.actor.model.freeze_vision_tower=false
    worker.actor.model.lora.rank="${LORA_RANK}"
    worker.actor.model.lora.alpha="${LORA_ALPHA}"
    worker.actor.model.lora.dropout="${LORA_DROPOUT}"
    worker.actor.model.lora.target_modules="${LORA_TARGET_MODULES}"
    worker.actor.optim.lr=2.0e-6
    worker.rollout.n="${SOLVER_GROUP_SIZE}"
    worker.actor.global_batch_size="${GRPO_MINIBATCH_SIZE}"
    trainer.project_name="${WANDB_PROJECT}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.save_checkpoint_path="${SAVE_ROOT}"
    trainer.total_epochs=100
    trainer.max_steps="${SOLVER_STEPS}"
    trainer.save_freq="${SOLVER_STEPS}"
    trainer.save_limit=1
    trainer.save_model_only=true
    trainer.find_last_checkpoint=false
    data.format_prompt=./examples/format_prompt/solver.jinja
    worker.reward.reward_function=./examples/reward_function/evo_vid_solver_reward.py:compute_score
    worker.reward.reward_function_kwargs.format_weight="${SOLVER_FORMAT_WEIGHT}"
    worker.reward.reward_function_kwargs.temporal_weight="${SOLVER_TEMPORAL_WEIGHT}"
    trainer.val_freq=-1
    trainer.val_before_train=false
    trainer.run_final_validation=false
    trainer.n_gpus_per_node="${SOLVER_N_GPUS}"
    worker.actor.micro_batch_size_per_device_for_update=1
    worker.actor.micro_batch_size_per_device_for_experience=1
)

if [[ -n "${SOLVER_INIT_ADAPTER}" ]]; then
    TRAIN_CMD+=("worker.actor.model.lora.init_adapter_path=${SOLVER_INIT_ADAPTER}")
fi
if [[ -n "${LORA_EXCLUDE_MODULES}" ]]; then
    TRAIN_CMD+=("worker.actor.model.lora.exclude_modules=${LORA_EXCLUDE_MODULES}")
fi

CUDA_VISIBLE_DEVICES="${SOLVER_TRAIN_GPUS}" "${TRAIN_CMD[@]}"

if [[ ! -f "${ACTOR_PATH}/lora_adapter/adapter_config.json" ]]; then
    echo "Solver training did not produce a LoRA adapter: ${ACTOR_PATH}/lora_adapter" >&2
    exit 1
fi

echo "Merge Solver LoRA for evaluation only"
"${PYTHON_BIN}" -m scripts.merge_lora_adapter \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${ACTOR_PATH}/lora_adapter" \
    --output_dir "${ACTOR_PATH}/huggingface"

echo "Solver adapter: ${ACTOR_PATH}/lora_adapter"
echo "Solver merged model: ${ACTOR_PATH}/huggingface"
