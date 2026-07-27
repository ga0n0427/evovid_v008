#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <merged_model_path> <run_id>" >&2
    exit 2
fi

MODEL_PATH=$1
RUN_ID=$2
export VLLM_DISABLE_COMPILE_CACHE=1

PYTHON_BIN=${PYTHON_BIN:-python3}
SERVER_GPUS=${SERVER_GPUS:-4,5,6,7}
EVOVID_SOLVER_PORTS=${EVOVID_SOLVER_PORTS:-5000,5001,5002,5003}
QUESTIONER_SOLVER_SAMPLES=${QUESTIONER_SOLVER_SAMPLES:-10}
SOLVER_SERVER_GPU_MEMORY=${SOLVER_SERVER_GPU_MEMORY:-0.8}
SOLVER_SERVER_MAX_TOKENS=${SOLVER_SERVER_MAX_TOKENS:-4096}
SOLVER_SERVER_MAX_MODEL_LEN=${SOLVER_SERVER_MAX_MODEL_LEN:-8192}

IFS=',' read -r -a gpu_ids <<< "${SERVER_GPUS}"
IFS=',' read -r -a ports <<< "${EVOVID_SOLVER_PORTS}"
if (( ${#gpu_ids[@]} != ${#ports[@]} || ${#gpu_ids[@]} != 4 )); then
    echo "SERVER_GPUS and EVOVID_SOLVER_PORTS must each contain four entries." >&2
    exit 2
fi

pids=()

cleanup() {
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

echo "Starting merged Solver services for RUN_ID=${RUN_ID}: ${MODEL_PATH}"
for index in "${!gpu_ids[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_ids[$index]}" "${PYTHON_BIN}" -u vllm_service_init/start_vllm_server.py \
        --port "${ports[$index]}" \
        --model_path "${MODEL_PATH}" \
        --gpu_mem_util "${SOLVER_SERVER_GPU_MEMORY}" \
        --max_tokens "${SOLVER_SERVER_MAX_TOKENS}" \
        --num_candidates "${QUESTIONER_SOLVER_SAMPLES}" \
        --max_model_len "${SOLVER_SERVER_MAX_MODEL_LEN}" &
    pids+=("$!")
done

wait
