#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STREAMLIT_PYTHON="${STREAMLIT_PYTHON:-/root/team15/streamlit_venv/bin/python}"

# The 20 GB MIG is reserved for vLLM. Keep OCR on CPU by default.
export OCR_DEVICE="${OCR_DEVICE:-cpu}"
export OCR_CPU_THREADS="${OCR_CPU_THREADS:-4}"
export OCR_PARALLEL_WORKERS="${OCR_PARALLEL_WORKERS:-3}"
export VLM_BATCH_WORKERS="${VLM_BATCH_WORKERS:-6}"
export OMP_NUM_THREADS="${OCR_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${OCR_MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OCR_OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${OCR_NUMEXPR_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${STREAMLIT_CUDA_VISIBLE_DEVICES:-}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:25241/v1}"
export VLLM_MODEL_ID="${VLLM_MODEL_ID:-fmcg-qwen3-vl-4b-lora}"

cd "$APP_DIR"

exec "$STREAMLIT_PYTHON" -m streamlit run "$APP_DIR/app.py" \
  --server.address "${STREAMLIT_ADDRESS:-127.0.0.1}" \
  --server.port "${STREAMLIT_PORT:-8501}"
