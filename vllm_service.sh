#!/usr/bin/env bash
set -euo pipefail

VLLM_PYTHON="/root/team15/vllm_venv/bin/python"
MODEL_DIR="${MODEL_DIR:-/root/team15/qwen3_vl_4b_merged}"
PID_FILE="/root/team15/vllm.pid"
LOG_FILE="/root/team15/vllm.log"
API_PORT="25241"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.44}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
VLLM_COMPAT_DIR="/root/team15/Phase3/vllm_compat"

find_api_pid() {
  local recorded_pid=""
  local recorded_args=""

  if [[ -r "$PID_FILE" ]]; then
    recorded_pid="$(tr -cd '0-9' < "$PID_FILE")"
    if [[ -n "$recorded_pid" && -r "/proc/$recorded_pid/cmdline" ]]; then
      recorded_args="$(tr '\0' ' ' < "/proc/$recorded_pid/cmdline")"
      if [[ "$recorded_args" == *"vllm.entrypoints.openai.api_server"* && "$recorded_args" == *"--port $API_PORT"* ]]; then
        printf '%s\n' "$recorded_pid"
        return 0
      fi
    fi
  fi

  pgrep -f "^.*python.*-m vllm\.entrypoints\.openai\.api_server .*--port $API_PORT( |$)" | head -n 1
}

start_vllm() {
  local existing_pid=""
  existing_pid="$(find_api_pid || true)"
  if [[ -n "$existing_pid" ]]; then
    echo "vLLM is already running with PID $existing_pid."
    echo "$existing_pid" > "$PID_FILE"
    return 0
  fi

  : > "$LOG_FILE"
  nohup setsid env \
    LD_LIBRARY_PATH=/usr/local/cuda/compat/lib \
    PYTHONPATH="$VLLM_COMPAT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    VLLM_ATTENTION_BACKEND="$ATTENTION_BACKEND" \
    "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --served-model-name fmcg-qwen3-vl-4b-lora \
    --host 127.0.0.1 \
    --port "$API_PORT" \
    --max-model-len 4096 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enforce-eager \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --mm-processor-kwargs '{"max_pixels":802816}' \
    > "$LOG_FILE" 2>&1 < /dev/null &

  local new_pid=$!
  echo "$new_pid" > "$PID_FILE"
  echo "vLLM started with PID $new_pid."
  echo "Run: $0 logs"
}

stop_vllm() {
  local api_pid=""
  local process_group=""
  api_pid="$(find_api_pid || true)"

  if [[ -z "$api_pid" ]]; then
    rm -f "$PID_FILE"
    echo "vLLM is not running."
    return 0
  fi

  process_group="$(ps -o pgid= -p "$api_pid" | tr -d '[:space:]')"
  if [[ -z "$process_group" || ! "$process_group" =~ ^[0-9]+$ ]]; then
    echo "Cannot resolve the vLLM process group for PID $api_pid." >&2
    return 1
  fi

  echo "Stopping vLLM process group $process_group..."
  kill -TERM -- "-$process_group" 2>/dev/null || true

  for _ in $(seq 1 15); do
    if ! kill -0 -- "-$process_group" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "vLLM stopped."
      return 0
    fi
    sleep 1
  done

  echo "vLLM did not stop after 15 seconds; forcing SIGKILL..."
  kill -KILL -- "-$process_group" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "vLLM stopped forcefully."
}

status_vllm() {
  local api_pid=""
  api_pid="$(find_api_pid || true)"
  if [[ -z "$api_pid" ]]; then
    rm -f "$PID_FILE"
    echo "vLLM is stopped."
    return 1
  fi

  echo "$api_pid" > "$PID_FILE"
  ps -p "$api_pid" -o pid,ppid,pgid,stat,etime,args
  curl -sS -o /dev/null -w 'Health: HTTP %{http_code}\n' "http://127.0.0.1:$API_PORT/health" || true
}

case "${1:-}" in
  start)
    start_vllm
    ;;
  stop)
    stop_vllm
    ;;
  restart)
    stop_vllm
    start_vllm
    ;;
  status)
    status_vllm
    ;;
  logs)
    tail -f "$LOG_FILE"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac
