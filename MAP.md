# Phase 3 directory map

```text
Phase 3/
├── README.md                         Main pipeline documentation
├── LORA_TRAINING_REPORT.md           Measured 4B training and evaluation report
├── SLIDE_METRICS_4B.md               Exact values to replace in the slide deck
├── qwen3_vl_loss_chart.png           Real 132-step loss and validation chart
├── vLLM.md                           vLLM 4B service reference
├── HUONG_DAN_DEPLOY_SERVER.md        Short server startup guide
├── requirements-vllm.txt             Reproducible vLLM environment pins
├── vllm_service.sh                   Managed vLLM start, stop, status, and logs
├── vllm_compat/
│   └── sitecustomize.py              Vision SDPA compatibility shim
├── streamlit_app/
│   ├── app.py                        Streamlit UI and batch pipeline
│   ├── start_remote.sh               Remote Streamlit launcher
│   ├── requirements.txt
│   └── README.md
├── qwen3_vl_4b_lora/                 Qwen3-VL-4B QLoRA module
│   ├── MAP.md
│   ├── configs/
│   ├── data/
│   ├── prompts/
│   ├── training/
│   └── artifacts/
├── Kaggle.ipynb                      Historical notebook pipeline
├── Local.ipynb                       Local notebook pipeline
├── Pipeline.drawio                   Architecture source diagram
└── submission.jsonl                  Existing submission output
```

Large safetensors, raw training runs, virtual environments, and local logs are excluded from Git.
