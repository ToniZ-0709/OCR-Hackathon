# LoRA Module Directory Map

```text
qwen3_vl_8b_lora/
│
├── MAP.md                              <- Folder structure map
├── LORA_TRAINING_REPORT.md             <- Official technical report
│
├── configs/                            <- Training and evaluation configurations
│   ├── training.yaml
│   └── evaluation.yaml
│
├── data/                               <- Audited FMCG dataset
│   └── final/
│       ├── train_350.jsonl             <- 350 training samples
│       ├── validation_50.jsonl         <- 50 validation samples
│       ├── all_400.jsonl               <- 400 total samples
│       ├── DATASET_CARD.md
│       └── provenance.json
│
├── prompts/                            <- Vietnamese FMCG prompt templates
│   ├── fmcg_grounding_system.txt
│   └── user_instruction.txt
│
├── training/                           <- Executable scripts
│   ├── train_qlora.sh                  <- GPU training launcher
│   ├── infer_validation.sh             <- Validation benchmark script
│   ├── deploy_transformers.sh          <- REST API server launcher
│   ├── setup_gpu.sh                    <- Server environment setup script
│   ├── preflight.py                    <- Integrity and hardware verification
│   ├── plot_loss.py                    <- Loss curve generation script
│   └── requirements-training.txt
│
└── artifacts/                          <- Generated outputs and weights
    │
    ├── best_adapter/                   <- Final LoRA weights and configs
    │   ├── adapter_model.safetensors
    │   ├── adapter_config.json
    │   ├── trainer_state.json
    │   ├── logging.jsonl
    │   └── args.json
    │
    ├── plots/                          <- Visual analysis charts
    │   ├── qwen3_vl_loss_chart.png
    │   ├── train_loss.png
    │   ├── eval_loss.png
    │   ├── train_token_acc.png
    │   └── train_learning_rate.png
    │
    ├── raw_training_runs/              <- Complete run logs and checkpoints (v0 - v3)
    │   └── qwen3_vl_8b_qlora/
    │
    └── qwen3_vl_8b_lora_gpu_bundle.tar.gz <- Portable deployment archive
```
