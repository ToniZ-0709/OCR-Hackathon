# Technical Training Report: Qwen3-VL-4B QLoRA for FMCG Grounding

This report contains the measured results of the Qwen3-VL-4B run completed on 24 August 2026. Every training and evaluation number below is derived from the saved 4B artifacts.

## 1. Run identity

| Item | Measured value |
|---|---|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` |
| Training method | Rank-16 QLoRA, 4-bit NF4 with double quantization |
| Framework | ms-swift 4.5.2 with PEFT 0.19.1 |
| Hardware | NVIDIA H100 80GB HBM3 MIG 2g.20gb, 19.625 GiB visible VRAM |
| Run directory | `qwen3_vl_4b_qlora/v0-20260824-154839` |
| Training duration | 639.2676 seconds, or 10 minutes 39.3 seconds |
| Completed optimization | 3 epochs, 132 optimizer steps |
| Selected checkpoint | `checkpoint-88`, epoch 2 |
| Selection rule | Lowest validation loss |

## 2. Dataset and provenance

| Dataset item | Value |
|---|---:|
| Total curated samples | 400 |
| Training split | 350 |
| Validation split | 50 |
| Full-set PRESENT labels | 226, or 56.50% |
| Full-set ABSENT labels | 174, or 43.50% |
| Validation PRESENT labels | 28 |
| Validation ABSENT labels | 22 |
| Preflight minimum sequence length | 231 tokens |
| Preflight maximum sequence length | 469 tokens |
| Preflight mean sequence length | 413.06 tokens |
| Context limit | 2,048 tokens |

The labels were originally model-assisted. Three independent visual-agent audits reviewed all 400 images under the FMCG-only policy, applied 21 corrections, and required no API keys. There was no independent human gold-standard transcription of every small label. The dataset must therefore be described as model-assisted and multi-agent-audited, not as a human-annotated brand benchmark.

## 3. Model and trainable parameters

The exact foundation-model parameter count was computed from all 713 tensors in the merged BF16 safetensors checkpoint. The adapter count was computed from all 504 tensors in `adapter_model.safetensors`.

| Parameter group | Count | Share of logical PEFT model |
|---|---:|---:|
| Frozen Qwen3-VL-4B foundation weights | 4,437,815,808 | 99.2612% |
| Trainable LoRA weights | 33,030,144 | 0.7388% |
| Logical total during PEFT adaptation | 4,470,845,952 | 100.0000% |

The ms-swift runtime prints 2,654.1880M stored parameter elements and 33.0301M trainable elements because the foundation linear weights are packed by BitsAndBytes NF4. That packed storage count is not the unquantized architecture parameter count and should not be used as the total model parameter value in the slide.

Architecture and adaptation scope:

- 24 vision-transformer blocks, all frozen.
- 36 language-model decoder layers.
- Multimodal aligner and patch merger frozen.
- LoRA applied to language-model `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` modules.
- No vision or aligner LoRA tensors.

## 4. Training configuration

| Hyperparameter | Configured value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Quantization | BitsAndBytes NF4, 4-bit, double quantization |
| Compute dtype | BF16 |
| Optimizer | Paged AdamW 8-bit |
| Peak learning rate | 1.0e-4 |
| Scheduler | Cosine decay with 5% linear warmup |
| Per-device training batch | 1 |
| Gradient accumulation | 8 |
| Effective batch | 8 samples per optimizer step |
| Per-device evaluation batch | 1 |
| Epochs | 3 |
| Save and evaluation cadence | End of each epoch |
| Attention implementation | PyTorch SDPA |
| Gradient checkpointing | Enabled |
| Image token cap | 1,024 |
| Random seed | 42 |

## 5. Training telemetry

### 5.1 Main results

| Training result | Value |
|---|---:|
| Initial logged training loss, step 1 | 4.20076561 |
| Final logged training loss, step 130 | 0.70670671 |
| Trainer mean loss across all 132 steps | 1.12138462 |
| Peak logged training token accuracy | 84.7125%, step 100 |
| Final logged training token accuracy | 83.6553%, step 130 |
| Training samples per second | 1.643 |
| Training steps per second | 0.206 |
| Peak logged GPU allocation | 7.15 GiB |

### 5.2 Validation by epoch

| Epoch | Step | Validation loss | Validation token accuracy | Result |
|---:|---:|---:|---:|---|
| 1 | 44 | 0.80163848 | 69.4952% | Checkpoint saved |
| 2 | 88 | **0.76511055** | 71.4286% | **Best checkpoint selected** |
| 3 | 132 | 0.79499555 | 72.5027% | Lower token error rate, but higher cross-entropy loss |

Epoch 3 improved token accuracy but worsened validation loss by 0.029885 compared with epoch 2. The configured selection criterion was validation loss, so `checkpoint-88` is the correct adapter to deploy.

### 5.3 Logged loss trajectory

| Step | Epoch | Train loss | Token accuracy | Learning rate |
|---:|---:|---:|---:|---:|
| 1 | 0.0229 | 4.2008 | 56.48% | 1.429e-5 |
| 5 | 0.1143 | 2.9924 | 64.39% | 7.143e-5 |
| 10 | 0.2286 | 2.6112 | 62.66% | 9.986e-5 |
| 15 | 0.3429 | 1.6837 | 65.04% | 9.899e-5 |
| 20 | 0.4571 | 1.5199 | 66.27% | 9.735e-5 |
| 25 | 0.5714 | 1.5814 | 66.23% | 9.497e-5 |
| 30 | 0.6857 | 1.3074 | 70.44% | 9.188e-5 |
| 35 | 0.8000 | 1.2131 | 72.29% | 8.812e-5 |
| 40 | 0.9143 | 1.5812 | 67.07% | 8.377e-5 |
| 45 | 1.0229 | 1.1941 | 73.22% | 7.888e-5 |
| 50 | 1.1371 | 0.7505 | 77.92% | 7.354e-5 |
| 55 | 1.2514 | 1.0082 | 77.60% | 6.782e-5 |
| 60 | 1.3657 | 0.7835 | 80.53% | 6.182e-5 |
| 65 | 1.4800 | 0.8703 | 75.46% | 5.564e-5 |
| 70 | 1.5943 | 0.9949 | 75.84% | 4.937e-5 |
| 75 | 1.7086 | 0.9325 | 77.55% | 4.311e-5 |
| 80 | 1.8229 | 0.9077 | 79.34% | 3.696e-5 |
| 85 | 1.9371 | 0.9163 | 76.94% | 3.101e-5 |
| 90 | 2.0457 | 0.7891 | 80.64% | 2.536e-5 |
| 95 | 2.1600 | 0.5747 | 83.66% | 2.010e-5 |
| 100 | 2.2743 | 0.6714 | 84.71% | 1.532e-5 |
| 105 | 2.3886 | 0.6389 | 84.00% | 1.108e-5 |
| 110 | 2.5029 | 0.7958 | 79.85% | 7.450e-6 |
| 115 | 2.6171 | 0.6910 | 83.87% | 4.490e-6 |
| 120 | 2.7314 | 0.6828 | 83.20% | 2.260e-6 |
| 125 | 2.8457 | 0.6696 | 83.85% | 7.700e-7 |
| 130 | 2.9600 | 0.7067 | 83.66% | 6.000e-8 |

The publication chart is `qwen3_vl_loss_chart.png`. The plotting script now fails when real telemetry is absent and never substitutes a synthetic loss curve.

## 6. Task-level evaluation on 50 validation images

Both models used the same frozen validation JSONL, image token cap, deterministic temperature 0, and 128 generated-token limit. Base and adapter predictions contain exactly 50 rows and were aligned by image ID.

| Metric | Base Qwen3-VL-4B | Qwen3-VL-4B + checkpoint-88 | Absolute change |
|---|---:|---:|---:|
| Gate accuracy | 56.00% | **94.00%** | +38.00 pp |
| Gate precision | 56.00% | **96.30%** | +40.30 pp |
| Gate recall / PRESENT response rate | **100.00%** | 92.86% | -7.14 pp |
| Gate F1 | 71.79% | **94.55%** | +22.75 pp |
| Negative rejection accuracy | 0.00% | **95.45%** | +95.45 pp |
| PRESENT-only macro token F1 | 23.89% | **49.46%** | +25.57 pp |
| PRESENT-only character similarity | 31.44% | **52.88%** | +21.44 pp |
| Output format compliance | 80.00% | **100.00%** | +20.00 pp |

Confusion matrices:

| Model | True PRESENT | False PRESENT | False ABSENT | True ABSENT |
|---|---:|---:|---:|---:|
| Base | 28 | 22 | 0 | 0 |
| Adapter | 26 | 1 | 2 | 21 |

The base model produced non-empty text for every validation image, including all 22 negatives. The adapter reduced false non-empty outputs from 22 to 1, a 95.45% reduction, while missing 2 of 28 positive images.

PRESENT-only text metrics are reported separately so that correct empty strings do not inflate description quality. Across all 50 samples, macro token F1 is 13.38% for base and 69.70% for the adapter, but the adapter value includes 21 perfect empty-versus-empty comparisons. PRESENT-only macro token F1, 49.46%, is the defensible description-quality number.

Normalized whole-sentence exact match on PRESENT images is 0% for both models. This is expected for free-form summaries and shows why whole-sentence exact match is too strict as the primary quality metric.

### Why Brand Exact Match is not reported

Every validation row has an empty `metadata.brands` list. There are no audited brand spans or entity-level gold labels, so a Brand Exact Match F1 score cannot be reproduced from this dataset. Use Gate F1 and PRESENT-only macro token F1 in the slide.

### Error audit

The three frozen-label gate mismatches are:

| Image | Frozen label | Adapter output gate | Visual audit |
|---|---|---|---|
| `priv_d_0160` | PRESENT | Empty | Genuine false negative. Branded Thiên Hồng candy cartons are visible. |
| `priv_d_0191` | ABSENT | Non-empty | The banner visibly advertises named FMCG food products. This frozen label is policy-ambiguous and likely incorrect. |
| `priv_d_0348` | PRESENT | Empty | Genuine false negative. X-Men For Boss Intense appears in a branded product-recall article. |

A post-hoc gate-only sensitivity calculation that treats `priv_d_0191` as PRESENT gives 96.00% gate accuracy, 100.00% gate precision, 93.10% gate recall, 96.43% gate F1, and 100.00% negative rejection. These adjudicated numbers are not the primary benchmark because the correction was identified after viewing model errors.

### Promotion-gate result

The adapter passes two configured gates:

- False non-empty reduction is 95.45%, above the required 25%.
- PRESENT-only macro token F1 increases from 23.89% to 49.46%.

It misses the configured maximum PRESENT recall drop: recall falls by 7.14 percentage points, while `evaluation.yaml` allows at most 5 points. The honest status is therefore two of three promotion conditions passed. The deployed model is suitable for system testing, but the two genuine false negatives should be added to a hard-example refinement set before claiming the promotion gate is fully passed.

## 7. Evaluation runtime

| Mode | Runtime for 50 images | Samples per second | Generated tokens |
|---|---:|---:|---:|
| Base, Transformers 4-bit | 44.5186 seconds | 1.123 | 919 |
| Adapter, Transformers 4-bit | 68.1478 seconds | 0.734 | 987 |

These runtimes measure sequential ms-swift evaluation and are not vLLM serving throughput.

## 8. Adapter and merged deployment artifacts

| Artifact | Measured value |
|---|---:|
| Best adapter file | 132,195,448 bytes, or 126.07 MiB |
| Adapter tensors | 504 |
| Adapter parameters | 33,030,144, stored as FP32 |
| Merged BF16 model tensors | 713 |
| Merged BF16 parameters | 4,437,815,808 |
| Merged BF16 safetensors | 8,875,719,328 bytes, or 8.266 GiB |

Remote deployment paths:

- Best adapter: `/root/team15/qwen3_vl_4b_best_adapter`
- Merged BF16 checkpoint: `/root/team15/qwen3_vl_4b_merged`
- Active API model ID: `fmcg-qwen3-vl-4b-lora`
- API: `http://127.0.0.1:25241/v1`
- Public Streamlit URL: `https://delay-buffer-unnerve.ngrok-free.dev`

The GPU was simultaneously used by an unrelated `/home/jovyan/ura-phase2-vlm` job. The active vLLM service therefore loads the clean BF16 checkpoint through vLLM's BitsAndBytes quantizer with `gpu-memory-utilization=0.44`. It uses SDPA for the vision path because the installed FlashAttention PTX was built with a newer CUDA toolchain than the server driver supports.

Verified serving state:

- vLLM 0.11.0 health endpoint returned HTTP 200.
- `/v1/models` returned `fmcg-qwen3-vl-4b-lora`.
- A real image request completed successfully.
- Six concurrent image requests completed 6 of 6 in 6.078 seconds, or 0.987 images per second.
- The service remained healthy after the concurrent request test.

## 9. Reproducibility artifacts

| Evidence | Local path |
|---|---|
| Full training summary | `qwen3_vl_4b_lora/artifacts/training/training_summary.json` |
| Final trainer state | `qwen3_vl_4b_lora/artifacts/training/trainer_state_final.json` |
| Best adapter metadata | `qwen3_vl_4b_lora/artifacts/best_adapter/` |
| Base predictions and metrics | `qwen3_vl_4b_lora/artifacts/evaluation/base_validation_*` |
| Adapter predictions and metrics | `qwen3_vl_4b_lora/artifacts/evaluation/adapter_validation_*` |
| Deterministic evaluator | `qwen3_vl_4b_lora/training/evaluate_predictions.py` |
| Training summarizer | `qwen3_vl_4b_lora/training/summarize_training.py` |
| Loss plotting script | `qwen3_vl_4b_lora/training/plot_loss.py` |

The 132 MB adapter weight is retained locally but excluded from Git by `.gitignore`. Small configs, metric JSON files, predictions, the final trainer state, and the plot are safe to commit.

## 10. Final conclusion

The Qwen3-VL-4B rank-16 QLoRA run solves most of the original false-description problem. The selected epoch-2 checkpoint improves negative rejection from 0% to 95.45%, gate F1 from 71.79% to 94.55%, and PRESENT-only token F1 from 23.89% to 49.46%. Two genuine positive images remain false negatives, and one frozen negative label is likely inconsistent with the written FMCG policy. Those limitations should remain visible in the presentation.
