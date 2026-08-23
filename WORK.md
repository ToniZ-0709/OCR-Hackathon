# Work Log — Phase 3

> All changes, fixes, and improvements are logged here chronologically.

---

## 2026-07-24 — Complete Phase 3 Notebook + SGLang Architecture

### Summary
Built the Phase 3 image summary pipeline from scratch, migrated from Phase 2's 6-layer cascade to a 2-step OCR → VLM architecture, and implemented the SGLang Docker serving model.

### Major Milestones

#### 1. Initial Phase 3 Notebook Created
- Removed: `BRAND_RULES`, `LEARNED_BRANDS`, `ProductPredictor`, NER, `confidence_checking`, `predict_product()` orchestrator, `PROMPT_LAYER_3/4`, `parse_vlm_output()`
- Kept: `classify_image()`, `preprocess()`, `postprocess_ocr()`, PPOCR, VietOCR, `Sort_Boxes()`, `Crop_Padding()`
- Added: `PROMPT_SUMMARY`, `build_ocr_context()`, `run_ocr()` wrapper, `vlm_summarize()`, JSONL output

#### 2. 22 Bugs Identified and Fixed
| # | Bug | Severity | Fix |
|---|-----|----------|-----|
| 1 | PROMPT_SUMMARY placeholder had `ocr_context}` instead of `__OCR_CONTEXT__` | Critical | Changed to `__OCR_CONTEXT__` |
| 2 | `.replace("{ocr_context}", ...)` never matched | Critical | Changed to `.replace("__OCR_CONTEXT__", ...)` |
| 3 | Missing `ocr_version='PP-OCRv6'` — loaded v4 by default | Critical | Added `ocr_version='PP-OCRv6'` |
| 4 | Sequential loop — no parallelism | Critical | Added `ThreadPoolExecutor(max_workers=2)` |
| 5 | `transformers>=4.45.0` too low for Qwen3VL | High | Changed to `>=4.49.0` |
| 6 | Flash Attention 2 not installed | High | Added `flash-attn` to install |
| 7 | `[line[0] for line in result[0]]` wrong format for `rec=False` | Critical | Changed to `boxes = result[0]` |
| 8 | f-string missing `` — SyntaxError | Critical | Added opening braces |
| 9 | Checkpoint write outside lock — race condition | High | Snapshot under lock, I/O outside |
| 10 | `.png` fallback was dead code | Medium | Restructured try→jpg→png→empty |
| 11 | `use_angle_cls=True` useless with `rec=False` | Low | Removed |
| 12 | `thumbnail((1024, 1024))` too high for 4B VLM | Medium | Reduced to 768 |
| 13 | Dataset walk only matched `.jpg` | Low | Added `.png`, `.jpeg` |
| 14 | `except Exception as e: pass` hid errors | High | Added `traceback.print_exc()` + `error_count` |
| 15 | Empty summary detection too narrow | Medium | Added comprehensive `empty_patterns` |
| 16 | No throughput measurement | Medium | Added timing + avg calculation |
| 17 | `bitsandbytes` installed but unused | Low | Removed from install |
| 18 | VietOCR `beamsearch=True` wasted time | Low | Disabled for greedy decoding |
| 19 | Standalone digit noise in OCR | Low | Added `not text.isdigit()` gate |
| 20 | Thread safety: `results` list unsynchronized | High | Added `threading.Lock()` × 3 |
| 21 | PIL `_util` guard added to prevent Kaggle crash | Medium | Added full guard with fallback |
| 22 | Redundant `processed` counter | Low | Removed |

#### 3. Docker + SGLang Architecture Implemented
- Created `phase3-sglang-pipeline.ipynb` — lightweight notebook using OpenAI-compatible API
- Removed dependencies: `transformers`, `accelerate`, `flash-attn`, `bitsandbytes`
- Added dependencies: `openai` (API client)
- VLM runs as persistent Docker container → no reload on kernel restart

#### 4. Workflow: `classify_image()` Expanded
- Added 3 new image classes: `underexposed`, `low_contrast`, `blurry`
- Tuned thresholds via 1,202 image audit using `Tune.py`
- Final distribution: 65% normal, 14% complex, 7% over/underexposed, 5.5% low contrast, 0.7% blurry

#### 5. PP-OCRv6 Adopted
- Upgraded from PaddleOCR 2.8.0 → 3.7.0
- +4.6% detection accuracy over v5
- Native Vietnamese support (`lang='vi'`)

#### 6. Infrastructure
- Created `docker-setup.md` (merged into implementation later)
- Created `README.md` with full pipeline docs
- Created `phase3-migration-report.md` (merged into implementation later)
- Created `phase3_improvement_report.md` (merged into implementation later)

### Files Created
| File | Purpose |
|------|---------|
| `phase3-summary-pipeline.ipynb` | Direct model loading version |
| `phase3-sglang-pipeline.ipynb` | SGLang API version |
| `README.md` | Pipeline documentation |
| `implementation_plan.md` | All improvements, changes, Docker setup |

---

## 2026-07-24 — Hotfix: Main.ipynb truncated cell + Qwen2.5 cleanup

### Bug Found: Cell-17 in Main.ipynb was truncated (554 chars)
- Only the checkpoint fragment was present — the entire main loop body (task building, ThreadPoolExecutor, result collection) was missing.
- This would crash on execution with `NameError: name 'test_df' is not defined`.
- **Fix:** Rewrote cell-17 with the full 3102-char main loop code.

### Qwen2.5-VL-7B references removed
- All mentions of Qwen2.5-VL-7B and model upgrade path below Qwen3 removed from `implementation_plan.md`.
- Removed "Model Upgrade Path" section entirely (no alternative models recommended).
- Removed "Model swap" open question.
- Verified zero Qwen2.5 references remain in all files.

### Final file state

| File | Status | Notes |
|------|--------|-------|
| `Main.ipynb` | ✅ Fixed | Cell-17 now complete (3102 chars) |
| `phase3-summary-pipeline.ipynb` | ✅ Clean | All 22 bugs fixed |
| `README.md` | ✅ Clean | No Qwen2.5 references |
| `implementation_plan.md` | ✅ Clean | No Qwen2.5 references |
| `WORK.md` | ✅ Current | This entry |

### Issues 1-3: Minor fixes applied

| Issue | Fix | Files |
|-------|-----|-------|
| **Prompt examples lost Vietnamese diacritics** | Restored: "Ảnh chụp hộp sữa Vinamilk Flex không đường..." (was "Anh chup hop sua...") | Main.ipynb cell-14, phase3-summary-pipeline.ipynb cell-14 |
| `show_log=False` may warn on PaddleOCR 3.7.0 | Kept — low risk, silently ignored if unsupported | Noted |
| `import torch` still present | Kept — VietOCR's Predictor internally loads torch anyway, so removing it wouldn't save the ~2GB package. Comment added. | Noted |

---

## 2026-07-25 — README pipeline diagram accuracy fix

### Issue: README pipeline diagram was inaccurate vs. actual `Main.ipynb` code

The original diagram implied `build_ocr_context` and `PROMPT_SUMMARY` ran "inside the SGLang API", and only showed `ocr_text` flowing to the VLM. Neither matched the actual code.

### Fixes applied to `README.md`

| Was (incorrect) | Now (matches code) |
|------------------|---------------------|
| Diagram implied build_ocr_context/PROMPT_SUMMARY run in SGLang | Clarified: they run in the notebook kernel; only the `chat.completions.create` call crosses to Docker via HTTP |
| Only `ocr_text` shown flowing to VLM | `box_data` (per-region, with `area` for prominence ranking) also flows to VLM |
| `img_pil` vs `img_cv2` not distinguished | Explicit: **original** `img_pil` (not preprocessed) is sent to VLM; preprocessing is OCR-only |
| `thumbnail((768,768)) → JPEG → base64` not shown | Added as explicit step in `vlm_summarize` |
| `empty_patterns` guard not shown | Added |
| Filter step (skip <8px / pure digits) not shown | Added inside `run_ocr` per-box loop |
| `postprocess_ocr` + `box_data` construction not shown | Both shown as separate outputs of `run_ocr` |

### Verified against `Main.ipynb` cells
- Cell 8: `classify_image` + `preprocess` + `postprocess_ocr` ✓
- Cell 10: PaddleOCR with `ocr_version='PP-OCRv6'`, `lang='vi'`, `rec=False` ✓
- Cell 12: `run_ocr` returns `(ocr_text, box_data)` with `area` field ✓
- Cell 14: `build_ocr_context` ranks `box_data` by area, top-10 ✓
- Cell 15: `vlm_summarize` thumbnails to 768, base64-encodes, calls SGLang ✓
- Cell 17: `process_one_image` passes `img_pil` (original) to `vlm_summarize` ✓

---

## 2026-07-25 21:58 — Docker setup rewritten as 7-step guideline

### Changes to `implementation_plan.md` §4

| Old (was) | New (now) |
|-----------|-----------|
| Mix of "no Ubuntu needed" + old WSL2 steps | Table comparing Linux vs Windows: 3 things you DON'T need on Windows ✅ |
| Step 1: "Enable WSL Integration" (irrelevant) | Removed — Docker Desktop auto-manages this |
| Step 3: Commented-out `cyankiwi/Qwen3-VL-8B-AWQ` link | Removed — only Qwen3-VL-4B-Instruct |
| Separate "Verify" + "Network" sections | Merged into 7 numbered steps in logical order |
| No clear "before GPU" vs "when GPU arrives" split | §4.2 = "do now" (3 steps), §4.3 = "when GPU arrives" (4 steps) |

### Files created/updated
- `implementation_plan.md` §4 — fully rewritten
- `.claude/CLAUDE.md` — created with project rules and session protocol

---

## 2026-07-25 22:30 — Final architecture fix: remote server, not Windows Docker

### Correction
The GPU is an **H100 80GB (MIG 20GB)** on a rented Linux server accessed via SSH. Earlier guides incorrectly assumed Windows Docker Desktop would host SGLang. Updated to reflect the real 2-machine setup:

```
Local Windows (Jupyter + OCR) → HTTP → Remote Linux (Docker + SGLang + Qwen3-VL-4B)
```

### Changes to `implementation_plan.md` §4
- Added note: "Docker Desktop on your local Windows is **not used** — all Docker runs on remote server"
- All commands changed from PowerShell to SSH/bash
- Added firewall port 25241 instructions
- Added SSH tunnel alternative for connection
- Container management moved to remote server commands

### Status
- Remote server has H100 GPU confirmed (`nvidia-smi` output captured)
- Docker + NVIDIA Container Toolkit still need to be installed on remote server
- Notebook `Main.ipynb` ready — just needs `SGLANG_BASE_URL` updated to server IP

---

## 2026-07-25 23:15 — Implementation plan restructured (Phase 3 beta removed)

### Context
User clarified that the 6-layer VLM pipeline was a transitional beta, not Phase 2. Official Phase 2 is at `Team-15---ArrayOfSunshine/`.

### Changes to `implementation_plan.md`
- Section 1: replaced "Evolution Overview" (3 columns) with clean "Core Changes from Phase 2" (Phase 2 vs Phase 3)
- Section 3: renamed "Removed from Phase 3 Beta" → "Remove in Phase 3", trimmed to Phase 2 components only
- All "Phase 3 beta" references cleaned out

### Pending
- [ ] Install Docker + NVIDIA Toolkit on remote server
- [ ] Start SGLang container on H100
---

## 2026-07-28 — SGLang → vLLM migration (SGLang dependency-hell fix)

### Problem
SGLang 0.4.6 doesn't support Qwen3-VL (no `qwen3_vl` module). Upgrading SGLang leads to CUDA 13 dependency conflicts incompatible with the server's CUDA 12.4 + torch 2.6.0 stack. 10+ hours lost to dependency resolution.

### Fix
Switched to vLLM, which supports Qwen3-VL natively with no version conflicts:

| | SGLang | vLLM |
|---|--------|------|
| Qwen3-VL support | Need v0.5.3+ (broken deps) | Native since v0.8.0+ |
| Install | Complex dependency chain | `pip install vllm` |
| CUDA compat | 0.5.6+ needs CUDA 13 | Works on CUDA 12.4 |

### Files updated
- `Main.ipynb` — `SGLANG_*` variables → `VLLM_*`, model name updated
- `implementation_plan.md` — section 4 rewritten for vLLM server command
- `README.md` — all SGLang references replaced
- `.claude/CLAUDE.md` — SGLang → vLLM
- `WORK.md` — this entry

### Problem
NotebookEdit tool created new cells alongside old ones instead of replacing them. The notebook accumulated duplicate cells and stale Kaggle code (`kagglehub`, `DATA_PATHS`, `/kaggle/input`) that couldn't be removed via partial edits.

### Fix
Rewrote `Main.ipynb` from scratch using a Python build script (`build_nb.py` → deleted after use):

| Cell | Content |
|:----:|---------|
| 0 | Title |
| 1-2 | Installation |
| 3-4 | Imports + NumPy/PIL compat fixes |
| 5-6 | Dataset loading from `C:/.../images/` (no CSV) |
| 7-8 | Preprocessing (6 categories) |
| 9-12 | OCR pipeline (PP-OCRv6 + VietOCR) |
| 13-15 | VLM Engine (PROMPT_SUMMARY + SGLang API) |
| 16-17 | Queue-based producer-consumer main loop |

### Cleanup
- Deleted `build_nb.py`
- Verified zero references to `kagglehub`, `DATA_PATHS`, `/kaggle`


## 2026-07-28 — SGLang -> vLLM migration + CUDA compat fix

### Problem
SGLang 0.4.6 does not support Qwen3-VL. Upgrading hits CUDA 13 conflicts.
vLLM needs CUDA 12.8 but server driver caps at 12.5.

### Fix
- Switched to vLLM 0.11.0 (minimum for Qwen3-VL support)
- Added CUDA 12.8 forward-compat package to bridge driver gap
- Pinned torch 2.8.0+cu128 before installing vLLM
- Set LD_LIBRARY_PATH + VLLM_ATTENTION_BACKEND=TORCH_SDPA at launch
- Added --max-model-len 32768 to fit the 20GB MIG slice

### Files updated
- Main.ipynb — VLLM_BASE_URL, VLLM_CLIENT, VLLM_MODEL (was SGLANG_*)
- implementation_plan.md — Steps 2-7 rewritten with CUDA compat + pinned versions
- README.md — SGLANG_CLIENT -> VLLM_CLIENT
- .claude/CLAUDE.md — all SGLang -> vLLM
- WORK.md — this entry

## 2026-07-29 — Context report + KaggleVer2 finalized

Created context_report.md with full architecture, resources, fine-tuning path.
Confirmed KaggleVer2 uses PaddleOCR 3.7.0 + GPU + PIR fix.
Ready for fine-tuning via LoRA on H100.

