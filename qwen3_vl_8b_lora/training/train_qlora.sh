#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/swift ]]; then
  echo "Run training/setup_gpu.sh first." >&2
  exit 1
fi
if [[ ! -f data/final/train_350.jsonl || ! -f data/final/validation_50.jsonl ]]; then
  echo "The exported 350/50 datasets are missing." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/artifacts/training/qwen3_vl_8b_qlora}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_STEPS="${MAX_STEPS:--1}"

.venv/bin/swift sft \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dataset "$PROJECT_ROOT/data/final/train_350.jsonl" \
  --val_dataset "$PROJECT_ROOT/data/final/validation_50.jsonl" \
  --tuner_type lora \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant true \
  --bnb_4bit_compute_dtype bfloat16 \
  --torch_dtype bfloat16 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --num_train_epochs 3 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --optim paged_adamw_8bit \
  --attn_impl "$ATTN_IMPL" \
  --max_length "$MAX_LENGTH" \
  --save_strategy epoch \
  --eval_strategy epoch \
  --save_total_limit 2 \
  --load_best_model_at_end true \
  --metric_for_best_model loss \
  --greater_is_better false \
  --logging_steps 5 \
  --report_to tensorboard \
  --dataset_num_proc 4 \
  --dataloader_num_workers 2 \
  --seed 42 \
  --output_dir "$OUTPUT_DIR"

echo "Training finished. Checkpoints: $OUTPUT_DIR"
