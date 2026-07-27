#!/usr/bin/env bash
# Usage:
#   bash scripts/questioner_train_penalty.sh <base_model> \
#       <previous_questioner_adapter_or_empty> <solver_merged_model> <experiment_name>

set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_DIR}"

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <base_model> <previous_questioner_adapter_or_empty> <solver_merged_model> <experiment_name>" >&2
    exit 2
fi

BASE_MODEL=$1
QUESTIONER_INIT_ADAPTER=${2:-}
SOLVER_MERGED_MODEL=$3
EXPERIMENT_NAME=$4

STORAGE_PATH=${STORAGE_PATH:?Set STORAGE_PATH first}
QUESTIONER_DATA=${QUESTIONER_DATA:?Set QUESTIONER_DATA first}
PREPROCESSED_VIDEO_DIR=${PREPROCESSED_VIDEO_DIR:?Set PREPROCESSED_VIDEO_DIR first}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python)}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_DIR}/examples/evovid_main.yaml}
QUESTIONER_PROMPT=${QUESTIONER_PROMPT:-${PROJECT_DIR}/examples/format_prompt/questioner_strict.jinja}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_EXCLUDE_MODULES=${LORA_EXCLUDE_MODULES:-}
QUESTIONER_STEPS=${QUESTIONER_STEPS:-20}
QUESTIONER_MAX_RESPONSE_LENGTH=${QUESTIONER_MAX_RESPONSE_LENGTH:-1024}
QUESTIONER_TRAIN_GPUS=${QUESTIONER_TRAIN_GPUS:-0,1,2,3}
QUESTIONER_N_GPUS=${QUESTIONER_N_GPUS:-4}
QUESTIONER_GROUP_SIZE=${QUESTIONER_GROUP_SIZE:-8}
GRPO_PROMPT_BATCH_SIZE=${GRPO_PROMPT_BATCH_SIZE:-16}
GRPO_MINIBATCH_SIZE=${GRPO_MINIBATCH_SIZE:-${GRPO_PROMPT_BATCH_SIZE}}
QUESTIONER_SOLVER_SAMPLES=${QUESTIONER_SOLVER_SAMPLES:-10}
QUESTIONER_INVALID_REWARD=${QUESTIONER_INVALID_REWARD:-0.0}
QUESTIONER_TEMPORAL_WEIGHT=${QUESTIONER_TEMPORAL_WEIGHT:-0.1}
QUESTIONER_DIVERSITY_THRESHOLD=${QUESTIONER_DIVERSITY_THRESHOLD:-0.5}
WANDB_PROJECT=${WANDB_PROJECT:-evovid-main-qwen3vl4b}
EVOVID_SOLVER_PORTS=${EVOVID_SOLVER_PORTS:-5000,5001,5002,5003}
EVOVID_SOLVER_RAW_AUDIT_DIR=${EVOVID_SOLVER_RAW_AUDIT_DIR:-${STORAGE_PATH}/solver_raw_audit}
EVOVID_SOLVER_RAW_AUDIT_TASKS=${EVOVID_SOLVER_RAW_AUDIT_TASKS:-2}
EVOVID_SOLVER_RAW_AUDIT_CANDIDATES=${EVOVID_SOLVER_RAW_AUDIT_CANDIDATES:-2}
export EVOVID_SOLVER_PORTS EVOVID_SOLVER_RAW_AUDIT_DIR
export EVOVID_SOLVER_RAW_AUDIT_TASKS EVOVID_SOLVER_RAW_AUDIT_CANDIDATES

SAVE_ROOT="${STORAGE_PATH}/models/${EXPERIMENT_NAME}"
ACTOR_PATH="${SAVE_ROOT}/global_step_${QUESTIONER_STEPS}/actor"
RUN_ID=$(date +%s%N)
SERVER_PID=""
SERVER_PGID=""

cleanup_solver_service() {
    if [[ -n "${SERVER_PGID}" ]] && kill -0 -- "-${SERVER_PGID}" 2>/dev/null; then
        kill -TERM -- "-${SERVER_PGID}" 2>/dev/null || true
    fi
    if [[ -n "${SERVER_PID}" ]]; then
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup_solver_service EXIT

wait_solver_services() {
    local port
    IFS=',' read -r -a ports <<< "${EVOVID_SOLVER_PORTS}"
    for _ in $(seq 1 300); do
        local ready=0
        for port in "${ports[@]}"; do
            if "${PYTHON_BIN}" -c 'import socket,sys; s=socket.create_connection(("127.0.0.1",int(sys.argv[1])),timeout=1); s.close()' "${port}" 2>/dev/null; then
                ready=$((ready + 1))
            fi
        done
        if (( ready == ${#ports[@]} )); then
            echo "All ${ready} Solver services are ready."
            return 0
        fi
        sleep 2
    done
    echo "Solver services did not become ready on ports ${EVOVID_SOLVER_PORTS}." >&2
    return 1
}

echo "Start merged Solver service: ${SOLVER_MERGED_MODEL}"
setsid bash vllm_service_init/start.sh "${SOLVER_MERGED_MODEL}" "${RUN_ID}" &
SERVER_PID=$!
SERVER_PGID=$SERVER_PID
wait_solver_services

echo "Train Questioner from Base + adapter: ${QUESTIONER_INIT_ADAPTER:-<new LoRA>}"
TRAIN_CMD=(
    "${PYTHON_BIN}" -m verl.trainer.main
    config="${CONFIG_PATH}"
    data.train_files="${QUESTIONER_DATA}"
    data.val_files="${QUESTIONER_DATA}"
    data.preprocessed_video_dir="${PREPROCESSED_VIDEO_DIR}"
    data.val_preprocessed_video_dir="${PREPROCESSED_VIDEO_DIR}"
    data.max_response_length="${QUESTIONER_MAX_RESPONSE_LENGTH}"
    data.rollout_batch_size="${GRPO_PROMPT_BATCH_SIZE}"
    worker.actor.model.model_path="${BASE_MODEL}"
    worker.actor.model.freeze_vision_tower=false
    worker.actor.model.lora.rank="${LORA_RANK}"
    worker.actor.model.lora.alpha="${LORA_ALPHA}"
    worker.actor.model.lora.dropout="${LORA_DROPOUT}"
    worker.actor.model.lora.target_modules="${LORA_TARGET_MODULES}"
    worker.actor.optim.lr=1.0e-6
    trainer.project_name="${WANDB_PROJECT}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.save_checkpoint_path="${SAVE_ROOT}"
    trainer.total_epochs=1000
    trainer.max_steps="${QUESTIONER_STEPS}"
    trainer.save_freq="${QUESTIONER_STEPS}"
    trainer.save_limit=1
    trainer.save_model_only=true
    trainer.find_last_checkpoint=false
    worker.reward.reward_function=./examples/reward_function/evo_vid_questioner_reward.py:compute_score
    worker.reward.reward_function_kwargs.solver_samples="${QUESTIONER_SOLVER_SAMPLES}"
    worker.reward.reward_function_kwargs.lambda_temporal="${QUESTIONER_TEMPORAL_WEIGHT}"
    worker.reward.reward_function_kwargs.diversity_distance_threshold="${QUESTIONER_DIVERSITY_THRESHOLD}"
    worker.reward.reward_function_kwargs.invalid_reward="${QUESTIONER_INVALID_REWARD}"
    worker.reward.reward_function_kwargs.cleanup_result_files=true
    trainer.val_freq=-1
    trainer.val_before_train=false
    trainer.run_final_validation=false
    trainer.n_gpus_per_node="${QUESTIONER_N_GPUS}"
    data.format_prompt="${QUESTIONER_PROMPT}"
    worker.rollout.n="${QUESTIONER_GROUP_SIZE}"
    worker.actor.global_batch_size="${GRPO_MINIBATCH_SIZE}"
)

if [[ -n "${QUESTIONER_INIT_ADAPTER}" ]]; then
    TRAIN_CMD+=("worker.actor.model.lora.init_adapter_path=${QUESTIONER_INIT_ADAPTER}")
fi
if [[ -n "${LORA_EXCLUDE_MODULES}" ]]; then
    TRAIN_CMD+=("worker.actor.model.lora.exclude_modules=${LORA_EXCLUDE_MODULES}")
fi

CUDA_VISIBLE_DEVICES="${QUESTIONER_TRAIN_GPUS}" "${TRAIN_CMD[@]}"

if [[ ! -f "${ACTOR_PATH}/lora_adapter/adapter_config.json" ]]; then
    echo "Questioner training did not produce a LoRA adapter: ${ACTOR_PATH}/lora_adapter" >&2
    exit 1
fi

echo "Merge Questioner LoRA for evaluation only"
"${PYTHON_BIN}" -m scripts.merge_lora_adapter \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${ACTOR_PATH}/lora_adapter" \
    --output_dir "${ACTOR_PATH}/huggingface"

echo "Questioner adapter: ${ACTOR_PATH}/lora_adapter"
echo "Questioner merged model: ${ACTOR_PATH}/huggingface"
