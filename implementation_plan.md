# Phase 3: Image Summary Pipeline — Implementation Plan

> All architecture decisions, improvements, and Docker setup guidelines are maintained here.

---

## 1. Core Changes from Phase 2

| Aspect | Phase 2 | Phase 3 |
|--------|:------:|:-------:|
| **Output** | CSV: `brand_name`, `product_name` | JSONL: `summary` (free-text) |
| **Task** | Extract brand + product fields | Describe image + embed brand/product in text |
| **Pipeline** | 3 layers (Regex → NER → Heuristic) | **2 steps** (OCR → VLM Summarize) |
| **VLM** | None | Qwen3-VL-4B FP16 via vLLM |
| **GPU** | T4 15GB (Kaggle) | **H100 Mini 20GB** |
| **Metric** | F1 + CER (single score) | F1 + Hallucination Rate (two scores) |
| **Empty Image** | `" "` (space) | `""` (empty string) |


**Phase 2:** Regex + NER + heuristics to extract structured fields from OCR text.

**Phase 3:** OCR reads text, VLM understands meaning. PaddleOCR handles text detection, VietOCR handles Vietnamese text recognition, Qwen3-VL-4B selects and embeds the relevant text into a natural summary.

---

## 2. Architecture

### 2.1 Physical deployment

```
LOCAL Windows                          REMOTE Linux Server (H100 20GB)
─────────────────────────              ─────────────────────────────────
Jupyter Notebook                       Docker + vLLM
├── Load Dataset (local folder scan)   └── Qwen3-VL-4B-Instruct (FP16)
├── PP-OCRv6 (detection)                    Port 25241 (firewall)
├── VietOCR (recognition)                   Automatic prefix caching
├── build_ocr_context()                     
└── vlm_summarize() ─── HTTP ──────►  http://<SERVER_IP>:25241/v1
```

### 2.2 Data flow (producer-consumer queue)

```
Producer Thread (CPU)            Queue (max 5)        Consumer Thread (GPU)
─────────────────────            ────────────          ──────────────────────
Open image                                          
Preprocess (CLAHE, gamma...)                       
PP-OCRv6 detect boxes                              
VietOCR recognize text                             
postprocess_ocr → ocr_text                         
Crop_Padding → box_data                            
         │                                         
         ├── put(image_id, img_pil, ocr_text, box_data) ──► Queue
         │                                                     │
         │                                   ◄── get ──────────┤
         │                                              │
         │                                    build_ocr_context()
         │                                    PROMPT_SUMMARY
         │                                    img.thumbnail(768)
         │                                    JPEG → base64
         │                                    vLLM API call
         │                                    parse response
         │                                    append to results
         │                                         │
         │                                    Checkpoint every 100
    
    queue.put(None) ──► signals end ──────►        break
```

**Why Queue instead of ThreadPoolExecutor:** The producer-consumer pattern makes the parallelism explicit — OCR (CPU) runs ahead filling the buffer while VLM (GPU) pulls items as fast as it can process them. The Queue acts as a decoupling buffer, allowing each side to run at its own pace.

### 2.3 Data format

| Step | Output | Description |
|------|--------|-------------|
| `run_ocr()` | `ocr_text: str` | Joined, deduplicated, whitespace-normalized text |
| `run_ocr()` | `box_data: list[dict]` | `[{text, area, box}, ...]` per region for prominence ranking |
| `build_ocr_context()` | `context: str` | Top-10 text lines ranked by area (largest first) |
| `vlm_summarize()` | `summary: str` | Free-text Vietnamese summary, or `""` if empty |

---

## 3. Component Decisions

### 3.1 Keep and enhance from Phase 2

| Component | Why |
|-----------|-----|
| `classify_image()` + `preprocess()` | Adaptive 6-category enhancement (tuned via the private dataset) |
| PP-OCRv6 (detection) | better accuracy than v5, native Vietnamese |
| VietOCR (vgg_transformer) | Vietnamese-optimized recognition with diacritic accuracy |
| `Sort_Boxes()` / `Crop_Padding()` | OCR preprocessing utilities |
| `postprocess_ocr()` | Text cleanup (whitespace + dedup) |

### 3.2 Remove in Phase 3

| Removed | Phase 2 purpose | Why removed |
|---------|-----------------|-------------|
| `BRAND_RULES` regex | Layer 1 brand matching via hand-crafted dictionary | VLM understands brands from image + OCR context — no regex needed |
| `ner_extract_brand()` (underthesea) | Layer 2 NER for unknown brands | VLM's semantic understanding replaces NER entirely |
| `extract_brand_from_boxes()` | Spatial heuristic fallback | Not needed — VLM sees the image directly |

### 3.3 New in Phase 3

| Component | Purpose |
|-----------|---------|
| `PROMPT_SUMMARY` | Unified prompt: 7 rules + few-shot examples + OCR context injection |
| `build_ocr_context()` | Format OCR text by prominence|
| `run_ocr()` | Combine detection + recognition in one call |
| `vlm_summarize()` | Single function: thumbnail → base64 → vLLM API → summary |
| vLLM API client (`openai`) | Lightweight HTTP calls to remote Docker container |
| Queue-based parallelism | CPU OCR ∥ GPU VLM via Thread + Queue (producer-consumer) |
| Anti-hallucination guards | Empty response patterns + prompt rules + no cache |
| Anti-hallucination guards | Empty response patterns + prompt rules + no cache |
---

## 4. GPU Server Setup Guide

### Environment

```
RKE2/Kubernetes container (Ubuntu 22.04)
├── H100 80GB HBM3 via MIG (~20 GB, 32 SMs)
├── NVIDIA Driver 550.54.15 + CUDA 12.4
├── No Docker daemon (socket unavailable)
├── No systemd
└── Python/vLLM runs directly on GPU
```

### Architecture

```
Local Windows                              Remote Server (H100 20GB MIG)
─────────────────────────                  ─────────────────────────────────
Jupyter Notebook (OCR)                     Python venv + vLLM
  PP-OCRv6 (detection)                     Qwen3-VL-4B-Instruct (FP16)
  VietOCR (recognition)                    Port 25241
  vlm_summarize() --- HTTP ------> http://<SERVER_IP>:25241/v1
```

---

### Step 0 -- Verify environment

```bash
whoami                        # root
nvidia-smi                    # confirms MIG 20GB partition
python3 --version             # should be 3.10+
```

---

### Step 1 -- Create team virtual environment

```bash
mkdir -p /root/team15
python3 -m venv /root/team15/venv
source /root/team15/venv/bin/activate
pip install -U pip
```

All subsequent commands run with this venv activated.

---

### Step 2 -- Install CUDA forward-compatibility package

The server's driver (550.54.15) caps at CUDA 12.5, but recent torch/vLLM need CUDA 12.8+. NVIDIA's forward-compat package bridges this gap (officially supported for H100-class GPUs).

```bash
# Install CUDA 12.8 forward-compat shim (one-time)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-compat-12-8
```

This installs to `/usr/local/cuda-12.8/compat`. You will need to prepend this to `LD_LIBRARY_PATH` every time you launch the server (Step 3).

---

### Step 3 -- Install vLLM with pinned torch version

Do **not** run `pip install vllm` bare -- the latest vLLM (0.26.0) pins torch 2.11.0 compiled for CUDA 13.0, which the compat shim cannot bridge. Qwen3-VL needs vllm >= 0.11.0. Pin to that version with torch for CUDA 12.8:

```bash
# Activate venv
source /root/team15/venv/bin/activate
pip install -U pip

# Install torch for CUDA 12.8 first (compat shim handles the driver mismatch)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Install vLLM 0.11.0 (first version with Qwen3-VL support)
pip install vllm==0.11.0

# Pin transformers (vLLM 0.11.0 breaks on transformers 5.x; Qwen3-VL needs >=4.57.0)
pip install "transformers>=4.57.0,<5.0.0"
```

**Sanity check** before launching the server:
```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH python -c "
import torch; torch.cuda.init(); print('CUDA init OK:', torch.cuda.get_device_name(0))
"
```
This confirms the compat shim + torch work together before you wait through a full model load.

---

### Step 4 -- Launch the vLLM server

```bash
# On the remote server, with venv activated
source /root/team15/venv/bin/activate

LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH \
VLLM_ATTENTION_BACKEND=TORCH_SDPA \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --host 0.0.0.0 \
  --port 25241 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --enable-chunked-prefill
```

First run downloads the model (~8 GB) to `~/.cache/huggingface`. Wait for `"Application startup complete"` in the logs.

**Flags explained:**
- `LD_LIBRARY_PATH=...compat` -- required every launch (driver too old for CUDA 12.8 without it)
- `VLLM_ATTENTION_BACKEND=TORCH_SDPA` -- uses PyTorch's SDPA backend (avoids FlashInfer/FlashAttn PTX compat issues on this driver)
- `--max-model-len 32768` -- default 262144 needs ~36GB KV cache; 32768 fits the 20GB MIG
- `--enable-prefix-caching` -- caches KV cache of shared prompt prefix (~600 tokens system prompt/rules) across requests, saving ~30-40% prefill time
- `--enable-chunked-prefill` -- splits prefill into chunks interleaved with decode, reduces TTFB for concurrent requests (10 consumer threads)

---

### Step 5 -- Verify locally on the server

Open a second SSH session (keep the server running), activate venv, then:

```bash
# Basic health check
curl http://localhost:25241/health

# List available models (meaningful for OpenAI-compatible API)
curl http://localhost:25241/v1/models
```

If the server says `"Server is ready"` in its logs but `/health` doesn't respond, try `/v1/models` instead — it's the more definitive API test. Do **not** proceed to Windows networking until the OpenAI-compatible endpoint responds.

---

### Step 6 -- Connect from Windows notebook

Once localhost on the remote server works, update `VLLM_BASE_URL` in `Main.ipynb`:

```python
VLLM_BASE_URL = "http://<SERVER_IP>:25241/v1"
```

**Alternative -- SSH tunnel** (more secure, no exposed port):
```powershell
ssh -p <PORT> -L 25241:localhost:25241 root@<IP> -i ~/.ssh/id_rsa
```
Then `VLLM_BASE_URL = "http://localhost:25241/v1"` works as if the GPU were local.

---

### Step 7 -- Quick inference test

Run the notebook on a few images. Check:

1. OCR output looks correct
2. VLM returns a Vietnamese summary
3. No hallucinated brands (verify against the image visually)

If all three pass, the pipeline works.

---

### Subsequent starts (once everything works)

Once the model is cached and the server has started successfully at least once, every future session follows the same 3-step pattern:

```bash
# 1. Activate venv (doesn't persist between terminal sessions)
source /root/team15/venv/bin/activate

# 2. Quick check GPU memory is free
nvidia-smi

# 3. Launch the server (all env vars required every time)
LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:$LD_LIBRARY_PATH VLLM_ATTENTION_BACKEND=TORCH_SDPA python -m vllm.entrypoints.openai.api_server   --model Qwen/Qwen3-VL-4B-Instruct   --host 0.0.0.0   --port 25241   --max-model-len 32768   --gpu-memory-utilization 0.85   --enable-prefix-caching   --enable-chunked-prefill
```

The model weights are cached at `~/.cache/huggingface` after the first download, so subsequent starts skip the download and load much faster.

---

### Server management

```bash
# Stop vLLM
Ctrl+C in the server SSH session

# Start again (no re-download needed)
source /root/team15/venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --host 0.0.0.0 \
  --port 25241 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-chunked-prefill

# Check memory usage
nvidia-smi
```

### Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `CUDA out of memory` | `--gpu-memory-utilization` too high for 20GB MIG | Lower to 0.75 or 0.70 |
| Model redownloads on restart | `~/.cache/huggingface` cleared between sessions | Check disk space; model stays cached if available |
| `command not found` after SSH | venv not activated | Run `source /root/team15/venv/bin/activate` first |
| Server starts but curl fails | Server still loading model (takes 1-3 min) | Wait and retry; check the SSH session for progress |
| Notebook can't connect | Wrong IP or port not reachable | Try SSH tunnel instead of direct connection |
## 5. Prompt Architecture

```
PROMPT_SUMMARY:
  Task: Write Vietnamese description
  Rules (7 mandatory):
    1. Describe context
    2. Summarize clearest text
    3. Embed brand/product naturally
    4. DO NOT hallucinate
    5. Ignore blurry/obscured text
    6. Return "" if empty
    7. Preserve original characters
  OCR Reference: __OCR_CONTEXT__
  Few-shot examples: 3 diverse cases
```

### Kaggle deployment note

PaddleOCR 3.7.0+ (PP-OCRv6) pulls PaddlePaddle with a OneDNN PirAttribute bug on Kaggle CPU runtime.
**Use PaddleOCR 2.8.0 on Kaggle** (same version as Phase 2) — it works reliably. The trade-off is losing the +4.6% detection accuracy from v6.

The  notebook uses PaddleOCR 2.8.0 by default.

### Anti-Hallucination Design

| Layer | Mechanism |
|-------|-----------|
| **Prompt rules** | 7 explicit constraints |
| **Empty detection** | `{'""', "''", "none", "null", "n/a", "na", " ", ""}` |
| **No cache** | Each image processed independently |
| **Temperature** | 0.3 (deterministic) |
| **repetition_penalty** | 1.05 (prevent loops) |

---

## 6. Adaptive Preprocessing Thresholds

Tuned via full dataset audit (1,202 images, CPU-only, ~44 seconds):

| Class | Threshold | % Dataset | Enhancement |
|-------|:---------:|:---------:|-------------|
| `overexposed` | mean > 200 or >30% pixels > 240 | 7% | CLAHE 3.5 + gamma 0.7 |
| `underexposed` | mean < 60 or >30% pixels < 30 | 7% | CLAHE 3.0 + gamma 1.5 |
| `blurry` | laplacian_var < 100 | 0.7% | Unsharp mask + CLAHE 2.0 |
| `low_contrast` | contrast < 42 | 5.5% | Histogram stretch + CLAHE 3.0 + sharpen |
| `complex` | saturation_std > 75 | 14% | Bilateral + CLAHE 2.5 + sharpen |
| `normal` | default | 65% | CLAHE 2.0 + sharpen |

Priority: overexposed → underexposed → blurry → low_contrast → complex → normal

---

## 7. VRAM Budget (L4 24GB)

| Component | Direct Load | Docker+vLLM |
|-----------|:-----------:|:-------------:|
| PP-OCRv6 | ~250 MB | ~250 MB |
| VietOCR | ~150 MB | ~150 MB |
| Qwen3-VL-4B | ~8.5 GB | ~8.5 GB |
| KV Cache | ~2-3 GB | ~2-3 GB |
| **Total** | **~11-12 GB** | **~11-12 GB** |
| **Headroom** | **~12 GB** | **~12 GB** |

---

## 8. Throughput Design

```
Time →
Img 1:  |--OCR(CPU)--|--VLM(GPU)----------------|
Img 2:                 |--OCR(CPU)--|--VLM(GPU)--|
Img 3:                                |--OCR(CPU)--...
```

- OCR: ~0.3-0.5s/image (CPU)
- VLM: ~1.5-2.5s/image (GPU, with prefix caching)
- Effective: ~1.5-2.5s/image, ~0.4-1.0 img/s

---

## 9. Notebook Structure

| Cell | Content | File |
|------|---------|------|
| 0-2 | Installation | `phase3-vllm-pipeline.ipynb` |
| 3 | Imports + patches | |
| 4-6 | Dataset loading | |
| 7-8 | Preprocessing | |
| 9-10 | PP-OCRv6 | |
| 11 | VietOCR | |
| 12 | OCR utilities + `run_ocr()` | |
| 13-14 | VLM Engine (PROMPT + vLLM client) | |
| 15 | `vlm_summarize()` | |
| 16-17 | Main Loop → JSONL | |

---

## 10. Open Questions

- [ ] **Brand list**: Placeholder `__BRAND_LIST__` in PROMPT_SUMMARY — inject when organizers provide the reference list
- [ ] **Fine-tuning**: LoRA fine-tuning of Qwen3-VL-4B on Vietnamese product data may reduce hallucination rate
- [ ] **Batch inference**: vLLM supports batching — test with batch_size=2-4 for higher throughput
