# Validation metrics

Predictions: `artifacts\evaluation\adapter_validation_predictions.jsonl`

| Metric | Value |
|---|---:|
| Samples | 50 |
| Gate accuracy | 94.00% |
| Gate precision | 96.30% |
| Gate recall | 92.86% |
| Gate F1 | 94.55% |
| Negative rejection accuracy | 95.45% |
| Present response rate | 92.86% |
| Normalized exact match | 42.00% |
| Macro token F1 | 69.70% |
| Macro character similarity | 71.61% |
| PRESENT-only normalized exact match | 0.00% |
| PRESENT-only macro token F1 | 49.46% |
| PRESENT-only macro character similarity | 52.88% |
| Output format compliance | 100.00% |

Note: Brand Exact Match is not reported because the validation metadata has no brand-span annotations. Macro token F1 compares the complete generated description with the audited reference.
