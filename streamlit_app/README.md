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
├── streamlit_app.py       # Application replica
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

## 3. How to Deploy to Streamlit Community Cloud

1. Push this folder to your GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository and select `streamlit_app/app.py` as the entry point.
4. The `packages.txt` file will automatically install the required `libgl1` and `libglib2.0-0` binaries on the Streamlit Debian server.
5. In the web app sidebar, enter your live ngrok backend URL (e.g. `https://<your-subdomain>.ngrok-free.dev/v1`) to connect to your remote vLLM server.

---

## 4. Key Features

* Single Image Demo: Upload an image, view adaptive preprocessing categories, inspect detected OCR text bounding boxes, and see the real-time Vietnamese summary.
* Batch Image and ZIP Processing: Upload a `.zip` archive or multiple images, run multi-threaded concurrent inference (with configurable worker count), view live progress, and export results directly as `submission.jsonl`.
* Backend Health Check: One-click connection test with the remote vLLM server.
* Configurable Inference: Interactive sliders for temperature, max tokens, and OCR context length.
