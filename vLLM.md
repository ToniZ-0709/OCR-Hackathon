# vLLM server for the fine-tuned Qwen3-VL-8B

This is the verified setup for the current H100 20 GB MIG server.

## Current deployment

- vLLM environment: `/root/team15/vllm_venv`
- Merged fine-tuned model: `/root/team15/qwen3_vl_8b_merged`
- Original LoRA adapter, kept unchanged: `/root/team15/best_adapter`
- API port: `25241`
- API model ID: `fmcg_lora`
- Maximum request length: `4096` tokens
- vLLM log: `/root/team15/vllm.log`
- vLLM PID file: `/root/team15/vllm.pid`

The adapter is merged into a separate BF16 checkpoint for serving. vLLM 0.11 has a Qwen3-VL dynamic-LoRA bug that crashes on image requests, even when the adapter contains only language-model weights. Do not add `--enable-lora` or `--lora-modules` to this command.

## Start vLLM

Run on the GPU server:

```bash
source /root/team15/vllm_venv/bin/activate

nohup env LD_LIBRARY_PATH=/usr/local/cuda/compat/lib \
  python -m vllm.entrypoints.openai.api_server \
  --model /root/team15/qwen3_vl_8b_merged \
  --served-model-name fmcg_lora \
  --host 0.0.0.0 \
  --port 25241 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --enforce-eager \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --skip-mm-profiling \
  > /root/team15/vllm.log 2>&1 < /dev/null &

echo $! > /root/team15/vllm.pid
```

Startup normally takes about 30 seconds. Follow it with:

```bash
tail -f /root/team15/vllm.log
```

The server is ready when the log contains `Application startup complete`.

## Check vLLM

```bash
curl -i http://127.0.0.1:25241/health
curl -s http://127.0.0.1:25241/v1/models
```

Expected results:

- `/health` returns HTTP 200.
- `/v1/models` lists `fmcg_lora`.

## Start ngrok

From the current `~` prompt, run:

```bash
/root/ngrok http 25241
```

Leave that terminal open. ngrok will print the public forwarding URL. Use that URL as the Streamlit API base, adding `/v1`. For example:

```text
https://your-current-ngrok-url.ngrok-free.dev/v1
```

If you started ngrok in the foreground, press `Ctrl+C` to stop it. If the public URL reports offline, check the local tunnel and log:

```bash
curl -s http://127.0.0.1:4040/api/tunnels
tail -n 30 /root/team15/ngrok.log
```

## Streamlit settings

- API Base URL: `<public URL printed above>/v1`
- Model ID: `fmcg_lora`

## Stop the services

Check the recorded process before stopping it:

```bash
ps -p "$(cat /root/team15/vllm.pid)" -o pid,cmd
ps -p "$(cat /root/team15/ngrok.pid)" -o pid,cmd
```

Then stop the exact recorded processes:

```bash
kill -TERM "$(cat /root/team15/vllm.pid)"
kill -TERM "$(cat /root/team15/ngrok.pid)"
```

## Verified on this server

- CUDA and vLLM imports succeeded on the H100 20 GB MIG.
- Local `/health` and `/v1/models` returned HTTP 200.
- A 206 x 206 image request succeeded.
- The largest bundled image, 1500 x 2000 and 2,946 prompt tokens, succeeded.
- Public ngrok `/health` and `/v1/models` returned HTTP 200.
