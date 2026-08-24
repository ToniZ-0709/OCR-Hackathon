# Multi-agent visual audit

## Coverage

Three independent agents inspected all 400 label images in bounded ranges:

- `agent_0001_0133.json`: 133 images
- `agent_0134_0266.json`: 133 images
- `agent_0267_0400.json`: 134 images

Every image ID appears exactly once across the three reports. The canonical review sheet was generated from the source and then updated by `python pipeline.py multi-agent-review`.

## Decision policy

The dataset is for FMCG image description, so the review uses `FMCG_ONLY`:

- Recognizable FMCG brand, named FMCG product, FMCG package, or FMCG promotional asset: `PRESENT`
- Generic food without a recognizable brand or named product: `ABSENT`
- Branded non-FMCG products such as electronics, vehicles, travel, appliances, games, and agricultural inputs: `ABSENT`

## Results

- Rows audited: 400
- Rows approved by visual agents: 400
- FMCG corrections applied: 21
  - 13 gate changes from `ABSENT` to `PRESENT`
  - 4 gate changes from `PRESENT` to `ABSENT`
  - 4 summary-only corrections
- Branded non-FMCG findings intentionally retained as `ABSENT`: 22
- Final labels: 226 `PRESENT`, 174 `ABSENT`
- Human review: 0
- API keys required: no

The exact row-level decisions are in `multi_agent_decision.json`. The final review sheet is `../manual_review/manual_labels.csv`.

## Limitations

The source labels were model-assisted, and the original Gemini request logs are unavailable. The agents checked visual grounding and scope, but did not independently establish a gold-standard OCR transcription for every tiny or low-resolution label. This dataset should therefore be described as model-assisted and multi-agent-audited, not as an independently human-annotated benchmark.
