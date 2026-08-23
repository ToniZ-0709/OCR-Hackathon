#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3 was not found." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
"$VENV_PYTHON" -m pip install "torch>=2.8,<2.12"
"$VENV_PYTHON" -m pip install -r training/requirements-training.txt

if [[ "${INSTALL_FLASH_ATTN:-1}" == "1" ]]; then
  if ! "$VENV_PYTHON" -m pip install "flash-attn>=2.8,<3" --no-build-isolation; then
    echo "flash-attn installation failed. Training can use SDPA by setting ATTN_IMPL=sdpa." >&2
  fi
fi

"$VENV_PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY

echo "GPU environment is ready. Run:"
echo "  .venv/bin/python training/preflight.py"
echo "  bash training/train_qlora.sh"

