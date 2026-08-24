# Phase 3 Streamlit Web Application

## FMCG Multimodal Image Summarization UI

This folder contains the complete, production-ready Streamlit web application for Team ArrayOfSunshine (Phase 3).

---

## 1. Directory Structure

```text
streamlit_app/
├── .streamlit/
│   └── config.toml        # Streamlit UI theme and server limits
├── .venv/                 # Local Python virtual environment
├── app.py                 # Main interactive application
├── start_remote.sh        # Remote launcher with current 4B defaults
├── requirements.txt       # Python dependencies
├── packages.txt           # Debian system dependencies (libgl1 for OpenCV)
└── README.md              # Setup and deployment instructions
```

---

## 2. How to Run Locally (Windows PowerShell)

### Step 1: Navigate to Directory and Activate Virtual Environment

```powershell
cd "C:\HCMUT\Projects\HACKATHON\2nd_URA\Phase 3\streamlit_app"

# Enable script execution for current session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Requirements (If Not Already Installed)

```powershell
pip install -r requirements.txt
```

### Step 3: Launch Streamlit App

```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 3. Run Streamlit and vLLM on the Same Remote Server

The remote deployment keeps two independent Python environments on the same machine:

- `/root/team15/vllm_venv` for vLLM and the GPU model.
- `/root/team15/streamlit_venv` for Streamlit and OCR.

Start Streamlit with:

```bash
/root/team15/Phase3/streamlit_app/start_remote.sh
```

The launcher uses these safe defaults:

- VLM API: `http://127.0.0.1:25241/v1`
- Model ID: `fmcg-qwen3-vl-4b-lora`
- OCR device: CPU, leaving the 20 GB MIG available for vLLM
- Batch OCR: 3 independent CPU engines by default, so images can run in parallel without sharing one PaddleOCR/VietOCR instance
- Streamlit address: `127.0.0.1:8501`

Start ngrok on the same server:

```bash
/root/ngrok http 8501
```

Open the HTTPS URL printed by ngrok. The vLLM API remains internal at `127.0.0.1:25241` and should not be exposed publicly. See `../HUONG_DAN_DEPLOY_SERVER.md` for the complete procedure.

---

## 4. Key Features

* Single Image Demo: Upload an image, view adaptive preprocessing categories, inspect detected OCR text bounding boxes, and see the real-time Vietnamese summary.
* Batch Image and ZIP Processing: Upload a `.zip` archive or multiple images, run multi-threaded concurrent inference (with configurable worker count), view live progress, and export results directly as `submission.jsonl`.
* Pure VLM Mode: Skip preprocessing and OCR, then send original images directly to Qwen3-VL with up to 6 concurrent HTTP requests by default.
* Backend Health Check: One-click connection test with the remote vLLM server.
* Configurable Inference: Interactive sliders for temperature, max tokens, and OCR context length.
