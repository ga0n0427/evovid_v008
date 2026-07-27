# EvoVid V008

Minimal training repository for the V008 EvoVid self-play recipe.

The repository contains only the V008 Questioner/Solver prompts, reward
functions, preprocessing, question generation, pseudo-labeling, LoRA training,
and the local vLLM Solver service.

## Requirements

- Linux
- Python 3.11
- CUDA 12.8-compatible NVIDIA driver
- 8 GPUs for the original V008 topology
- A preprocessed 5,778-row video JSONL and its referenced `.pt` tensors
- Access to `Qwen/Qwen3-VL-4B-Instruct`, or a local model snapshot

## Environment

```bash
conda create -n evovid_v008_clean python=3.11 -y
conda activate evovid_v008_clean

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install -e . --no-deps
```

FlashInfer 0.6.15 requires a working CUDA toolchain at runtime. `ninja` is
explicitly pinned because FlashInfer JIT compilation needs the executable.

## Data

Preprocessing is intentionally separate from training. The V008 cache uses the
nominal 32,768-pixel, 16-frame recipe below. Qwen patch-size rounding may make
the stored frame shape slightly smaller than the nominal cap.

```bash
python scripts/preprocess_videos.py \
  --input_file /path/to/evovid_raw.jsonl \
  --output_dir /path/to/preprocessed_videos \
  --output_file /path/to/evovid_preprocessed.jsonl \
  --video_fps 2 \
  --video_max_frames 16 \
  --video_min_pixels 32768 \
  --video_max_pixels 32768 \
  --video_total_pixels 524288 \
  --workers 64 \
  --skip_errors
```

The training launcher expects these paths:

```bash
export SOLVER_VIDEO_DATA=/path/to/evovid_preprocessed.jsonl
export PREPROCESSED_VIDEO_DIR=/path/to/preprocessed_videos
export STORAGE_PATH=/path/to/output/V008
```

Every row in `SOLVER_VIDEO_DATA` must resolve to an existing preprocessed
tensor. The setup check validates this contract before training.

## Model and authentication

The default model is downloaded from Hugging Face:

```bash
export MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct
```

An existing immutable snapshot may be used instead:

```bash
export MODEL_PATH=/path/to/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/<revision>
```

For offline logging:

```bash
export WANDB_MODE=offline
```

For online logging, authenticate with `wandb login` or set `WANDB_API_KEY`.
No credentials are stored in this repository.

## Validate without starting training

```bash
conda activate evovid_v008_clean
export SOLVER_VIDEO_DATA=/path/to/evovid_preprocessed.jsonl
export PREPROCESSED_VIDEO_DIR=/path/to/preprocessed_videos
export STORAGE_PATH=/path/to/output/V008
export WANDB_MODE=offline

bash scripts/run_V008.sh --check
```

The check validates Python packages, the input manifest, preprocessed tensors,
and the expected eight-GPU topology. It does not start model training.

## Run V008

```bash
conda activate evovid_v008_clean
export SOLVER_VIDEO_DATA=/path/to/evovid_preprocessed.jsonl
export PREPROCESSED_VIDEO_DIR=/path/to/preprocessed_videos
export STORAGE_PATH=/path/to/output/V008
export WANDB_MODE=online

bash scripts/run_V008.sh
```

V008 defaults:

- 8 self-play iterations
- 20 Questioner steps and 20 Solver steps per iteration
- Qwen3-VL-4B-Instruct
- LoRA rank 32, alpha 64, dropout 0.05
- 8 rollout candidates
- Questioner prompt batch 16 and minibatch 8
- Effective global batch 128 and effective minibatch 64 with group size 8
- 5,778 generated questions per iteration
- 10 Solver candidates for pseudo-labeling
- 16 frames with the V008 32k preprocessing recipe

## Repository layout

```text
examples/
  evovid_main.yaml
  format_prompt/
    questioner_strict.jinja
    solver.jinja
  reward_function/
question_generate/
question_evaluate/
scripts/
verl/
vllm_service_init/
```
