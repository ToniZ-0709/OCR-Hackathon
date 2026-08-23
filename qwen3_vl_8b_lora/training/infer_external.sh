#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-}"
ADAPTER="${2:-}"
if [[ "$MODE" != "base" && "$MODE" != "adapter" ]]; then
  echo "Usage: bash training/infer_external.sh base" >&2
  echo "   or: bash training/infer_external.sh adapter /path/to/checkpoint" >&2
  exit 1
fi
if [[ "$MODE" == "adapter" && -z "$ADAPTER" ]]; then
  echo "Adapter checkpoint is required." >&2
  exit 1
fi
if [[ ! -f data/final/external_test_200.jsonl ]]; then
  echo "External data is absent. Rebuild the archive with --include-external-test." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
RESULT_PATH="$PROJECT_ROOT/artifacts/${MODE}_external_predictions.jsonl"
mkdir -p "$PROJECT_ROOT/artifacts"

COMMON_ARGS=(
  --infer_backend transformers
  --val_dataset "$PROJECT_ROOT/data/final/external_test_200.jsonl"
  --quant_method bnb
  --quant_bits 4
  --torch_dtype bfloat16
  --temperature 0
  --max_new_tokens 128
  --result_path "$RESULT_PATH"
  --stream false
)

if [[ "$MODE" == "base" ]]; then
  .venv/bin/swift infer --model Qwen/Qwen3-VL-8B-Instruct "${COMMON_ARGS[@]}"
else
  .venv/bin/swift infer --adapters "$ADAPTER" "${COMMON_ARGS[@]}"
fi

echo "Saved: $RESULT_PATH"
