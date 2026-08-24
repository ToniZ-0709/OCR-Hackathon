# vLLM service for the fine-tuned Qwen3-VL-4B

## Current server state

| Item | Value |
|---|---|
| Python environment | `/root/team15/vllm_venv` |
| Merged BF16 checkpoint | `/root/team15/qwen3_vl_4b_merged` |
| Best LoRA adapter | `/root/team15/qwen3_vl_4b_best_adapter` |
| Served model ID | `fmcg-qwen3-vl-4b-lora` |
| API | `http://127.0.0.1:25241/v1` |
| Service script | `/root/team15/Phase3/vllm_service.sh` |
| Log | `/root/team15/vllm.log` |
| PID file | `/root/team15/vllm.pid` |

The server is currently sharing the 20 GB MIG with an unrelated GPU job. The service therefore reads the clean merged BF16 checkpoint and quantizes it through vLLM's BitsAndBytes loader. This is different from serving the ms-swift pre-quantized export, which is not compatible with the vLLM 0.11 Qwen3-VL loader.

Current defaults:

- BitsAndBytes 0.50.1 on-load quantization.
- GPU memory utilization 0.44.
- Maximum model length 4,096 tokens.
- Maximum six active sequences.
- Maximum 1,024 batched prefill tokens per scheduling step.
- Maximum 802,816 image pixels, equivalent to the 1,024-image-token training cap.
- Vision attention forced to PyTorch SDPA.

The SDPA compatibility shim is in `vllm_compat/sitecustomize.py`. It is required because the installed FlashAttention PTX was compiled with a newer CUDA toolchain than the server's NVIDIA driver can execute. The shim is loaded only by this service and does not modify the shared Python installation.

## One-time environment setup

The current server is already configured. On a recreated environment, install:

```bash
/root/team15/vllm_venv/bin/pip install -r /root/team15/Phase3/requirements-vllm.txt
```

## Start and watch the service

```bash
/root/team15/Phase3/vllm_service.sh start
/root/team15/Phase3/vllm_service.sh logs
```

Wait for:

```text
Application startup complete
```

Press `Ctrl+C` to leave the log view. This does not stop vLLM.

## Check the endpoint

```bash
/root/team15/Phase3/vllm_service.sh status
curl -s http://127.0.0.1:25241/v1/models
```

Expected model ID:

```text
fmcg-qwen3-vl-4b-lora
```

Streamlit must use:

```text
VLLM_BASE_URL=http://127.0.0.1:25241/v1
VLLM_MODEL_ID=fmcg-qwen3-vl-4b-lora
```

## Stop or restart

```bash
/root/team15/Phase3/vllm_service.sh stop
/root/team15/Phase3/vllm_service.sh restart
```

The stop command resolves the live process by API port, sends `SIGTERM` to the full process group, waits up to 15 seconds, and then uses `SIGKILL` only if required. Do not stop vLLM with `kill -TERM "$(cat /root/team15/vllm.pid)"`, because the PID file may be stale and the EngineCore child can survive.

## Verified deployment evidence

- vLLM 0.11.0 health endpoint returned HTTP 200.
- `/v1/models` returned `fmcg-qwen3-vl-4b-lora`.
- One real validation image completed successfully.
- Six concurrent image requests completed 6 of 6 in 6.078 seconds.
- Measured concurrent smoke-test throughput was 0.987 images per second.
- vLLM remained healthy after the six-request test.

The six-request result is a serving smoke test, not the 50-image quality benchmark. See `LORA_TRAINING_REPORT.md` for model-quality metrics.
