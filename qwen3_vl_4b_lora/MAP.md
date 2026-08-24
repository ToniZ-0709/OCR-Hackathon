# Qwen3-VL-4B LoRA module map

```text
qwen3_vl_4b_lora/
├── configs/
│   ├── training.yaml                 Declarative rank-16 QLoRA configuration
│   └── evaluation.yaml               Reproducible validation metrics and gates
├── data/
│   ├── final/
│   │   ├── train_350.jsonl
│   │   ├── validation_50.jsonl
│   │   ├── all_400.jsonl
│   │   ├── external_test_200.jsonl
│   │   ├── DATASET_CARD.md
│   │   └── provenance.json
│   ├── manifests/                    Source and split manifests
│   ├── manual_review/                Canonical review sheet
│   ├── manual_source/                Original model-assisted labels
│   └── reports/                      Three-agent visual audit evidence
├── prompts/
│   └── qwen_training_v1.txt          Vietnamese FMCG grounding prompt
├── training/
│   ├── setup_gpu.sh                  Training environment setup
│   ├── preflight.py                  GPU, dataset, and token validation
│   ├── train_qlora.sh                Qwen3-VL-4B rank-16 launcher
│   ├── infer_validation.sh           Base and adapter validation inference
│   ├── infer_external.sh             Optional external-set inference
│   ├── evaluate_predictions.py       Deterministic task evaluator
│   ├── summarize_training.py         Artifact telemetry extractor
│   ├── merge_adapter_bf16.py         Clean BF16 adapter merge
│   ├── plot_loss.py                  Real-metric loss chart generator
│   ├── package_gpu_bundle.py         Portable bundle builder
│   └── requirements-training.txt
└── artifacts/
    ├── best_adapter/                 Checkpoint-88 metadata and local weight
    ├── evaluation/                   Base and adapter predictions and metrics
    ├── training/                     Final run summary and trainer state
    └── plots/                        Publication loss chart
```

`adapter_model.safetensors` is kept locally and ignored by Git. The remaining small evidence files are intended to be committed.
