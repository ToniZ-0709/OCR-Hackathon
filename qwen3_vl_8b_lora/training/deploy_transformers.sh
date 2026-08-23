#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ADAPTER="${1:-}"
if [[ -z "$ADAPTER" || ! -d "$ADAPTER" ]]; then
  echo "Usage: bash training/deploy_transformers.sh /path/to/adapter-checkpoint" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
PORT="${PORT:-8000}"

.venv/bin/swift deploy \
  --adapters "$ADAPTER" \
  --infer_backend transformers \
  --quant_method bnb \
  --quant_bits 4 \
  --torch_dtype bfloat16 \
  --temperature 0 \
  --max_new_tokens 128 \
  --served_model_name fmcg-qwen3-vl-8b-lora \
  --host 0.0.0.0 \
  --port "$PORT"

