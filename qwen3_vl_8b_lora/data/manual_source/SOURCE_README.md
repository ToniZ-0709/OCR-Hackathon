# Consolidated 400-image FMCG source

This directory preserves the original label source imported from the legacy dataset folder. It is archival input, not an independent gold annotation set.

## Source statistics

- Images: 400 (`priv_d_0001.jpg` through `priv_d_0400.jpg`)
- Source labels: 217 `PRESENT`, 183 `ABSENT`
- Source statuses: 397 `CONFIRMED`, 3 `REPAIRED`
- Source generation: model-assisted; original Gemini request logs were not retained

The source count above is intentionally different from the final reviewed count. The automated three-agent FMCG-only review applies 21 corrections and retains 22 branded non-FMCG findings as `ABSENT`, producing 226 `PRESENT` and 174 `ABSENT` final labels. See:

- `../../reports/agent_0001_0133.json`
- `../../reports/agent_0134_0266.json`
- `../../reports/agent_0267_0400.json`
- `../../reports/multi_agent_decision.json`

## Files

- `dataset_400_master.json`: source metadata, source gate, source summary, and source status
- `dataset_400_review.csv`: source review export
- `VERIFIER_AUDIT_REPORT.md`: legacy audit narrative retained for provenance

The canonical editable decisions live in `../../manual_review/manual_labels.csv`. The final QLoRA JSONL files are generated only by `python ../../pipeline.py manual-import` after the automated review has run.

## Scope policy

This LoRA dataset is for FMCG image description. A recognizable FMCG brand, named FMCG product, FMCG package, or FMCG promotional asset is `PRESENT`. Non-FMCG products and generic food without a recognizable brand or named product are `ABSENT` with an exact empty target.
