# Qwen3-VL-4B slide replacement values

Use these values to update the presentation. They come from the completed 4B run and its saved artifacts.

## Model and architecture

| Slide field | Replace with |
|---|---|
| Foundation model | `Qwen/Qwen3-VL-4B-Instruct` |
| Vision blocks | 24, frozen |
| Language decoder layers | 36 |
| Foundation parameters | 4,437,815,808 |
| Trainable LoRA parameters | 33,030,144 |
| Logical PEFT total | 4,470,845,952 |
| Trainable share | 0.7388% |
| Frozen share | 99.2612% |
| Adapter size | 132,195,448 bytes, 132.20 MB decimal, or 126.07 MiB |
| LoRA configuration | rank 16, alpha 32, dropout 0.05 |

Do not use the 2,654.1880M number printed by the quantized runtime as the architecture parameter count. BitsAndBytes packs the frozen NF4 weights, so that runtime storage-element count is not comparable with the unquantized model parameter count.

## Training result

| Slide field | Replace with |
|---|---:|
| Initial logged training loss | 4.2008 at step 1 |
| Final logged training loss | 0.7067 at step 130 |
| Mean loss across all 132 optimizer steps | 1.1214 |
| Peak logged training token accuracy | 84.71% at step 100 |
| Epoch 1 validation loss | 0.8016 |
| Epoch 2 validation loss | **0.7651, best** |
| Epoch 3 validation loss | 0.7950 |
| Best checkpoint | checkpoint-88, epoch 2 |
| Training time | 639.27 seconds, or 10 minutes 39.3 seconds |
| Training speed | 1.643 samples/s, 0.206 optimizer steps/s |
| Peak logged GPU allocation | 7.15 GiB |

Use [qwen3_vl_loss_chart.png](qwen3_vl_loss_chart.png) for the loss figure.

## Validation table, 50 frozen samples

Recommended replacement for the old comparison table:

| Metric | Base 4B | 4B + QLoRA |
|---|---:|---:|
| Gate F1 | 71.79% | **94.55%** |
| Negative rejection accuracy | 0.00% | **95.45%** |
| PRESENT-only macro token F1 | 23.89% | **49.46%** |
| PRESENT-only character similarity | 31.44% | **52.88%** |
| Output format compliance | 80.00% | **100.00%** |

Additional gate metrics for speaker notes:

| Metric | Base 4B | 4B + QLoRA |
|---|---:|---:|
| Gate accuracy | 56.00% | 94.00% |
| Gate precision | 56.00% | 96.30% |
| Gate recall | 100.00% | 92.86% |
| False non-empty predictions | 22 | 1 |
| False-empty predictions | 0 | 2 |

Replace the old `Brand Exact Match F1` row with `PRESENT-only macro token F1`. Brand Exact Match cannot be calculated because all validation `metadata.brands` lists are empty.

Do not present the 42.00% all-sample exact-match value as description accuracy. It consists entirely of 21 correctly rejected negative images; exact match on the 28 PRESENT descriptions is 0% because the task generates free-form sentences.

## Deployment result

| Slide field | Replace with |
|---|---|
| Served model ID | `fmcg-qwen3-vl-4b-lora` |
| Merged source checkpoint | BF16, 8.266 GiB |
| Active vLLM mode | BitsAndBytes quantization on load |
| vLLM concurrency setting | 6 sequences |
| Verified concurrent batch | 6 of 6 requests successful |
| Verified batch wall time | 6.078 seconds |
| Verified batch throughput | 0.987 images/s |

The throughput value is a six-image smoke test on the shared server, not the 50-image model-quality benchmark.

## Dataset wording

Use this wording:

> 400 model-assisted labels, audited image-by-image by three independent visual agents under an FMCG-only policy; 350 training samples and 50 frozen validation samples; no API keys required.

Do not claim Gemini and an OpenAI API reached automated consensus. The original Gemini request logs are unavailable, and the final verification was the local multi-agent visual audit.

## Limitation to keep in the slide or speaker notes

The adapter passes the false-nonempty-reduction and token-F1 promotion conditions, but PRESENT recall falls by 7.14 percentage points, exceeding the configured maximum 5-point drop. Two genuine positive images were rejected. One apparent false positive, `priv_d_0191`, is likely a frozen-label policy error because the image visibly advertises named FMCG food products.
