# Validation metrics

Predictions: `artifacts\evaluation\base_validation_predictions.jsonl`

| Metric | Value |
|---|---:|
| Samples | 50 |
| Gate accuracy | 56.00% |
| Gate precision | 56.00% |
| Gate recall | 100.00% |
| Gate F1 | 71.79% |
| Negative rejection accuracy | 0.00% |
| Present response rate | 100.00% |
| Normalized exact match | 0.00% |
| Macro token F1 | 13.38% |
| Macro character similarity | 17.61% |
| PRESENT-only normalized exact match | 0.00% |
| PRESENT-only macro token F1 | 23.89% |
| PRESENT-only macro character similarity | 31.44% |
| Output format compliance | 80.00% |

Note: Brand Exact Match is not reported because the validation metadata has no brand-span annotations. Macro token F1 compares the complete generated description with the audited reference.
