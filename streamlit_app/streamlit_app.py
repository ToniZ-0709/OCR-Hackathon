import os
import sys
import ssl
import io
import time
import json
import base64
import re
import zipfile
import urllib.request
import urllib3
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── 1. CRITICAL: Import Torch First on Windows (Avoids DLL collisions) ───
import torch

# ─── 2. SSL & Network Patches for Windows (Enables VietOCR downloads) ───
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

urllib3.disable_warnings()
try:
    _orig_get = requests.get
    def _unverified_get(url, *args, **kwargs):
        kwargs['verify'] = False
        return _orig_get(url, *args, **kwargs)
    requests.get = _unverified_get

    _orig_request = requests.Session.request
    def _unverified_request(self, method, url, *args, **kwargs):
        kwargs['verify'] = False
        return _orig_request(self, method, url, *args, **kwargs)
    requests.Session.request = _unverified_request
except Exception:
    pass

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import streamlit as st
from openai import OpenAI

# Fix PIL._util compatibility
import PIL
if not hasattr(PIL, "_util"):
    class _util: pass
    PIL._util = _util
if not hasattr(PIL._util, "is_directory"):
    PIL._util.is_directory = os.path.isdir

# Fix numpy sctypes for older dependencies
try:
    _ = np.sctypes
except AttributeError:
    np.sctypes = {
        "float": [np.float16, np.float32, np.float64],
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "complex": [np.complex64, np.complex128]
    }

# Fix pkg_resources for Python 3.12 / setuptools >= 70
try:
    import pkg_resources
except ImportError:
    try:
        import setuptools
        import pkg_resources
    except Exception:
        pass

# ─── Streamlit Page Configuration ───
st.set_page_config(
    page_title="Team ArrayOfSunshine | FMCG Multimodal Summary",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .badge-success { background-color: #dcfce7; color: #166534; }
    .badge-error { background-color: #fee2e2; color: #991b1b; }
    .badge-warning { background-color: #fef3c7; color: #92400e; }
    .badge-category { background-color: #e0f2fe; color: #0369a1; border: 1px solid #bae6fd; }
    .result-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.25rem;
        margin-top: 1rem;
        font-size: 1.1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ─── Model & Engine Initialization (Cached) ───
@st.cache_resource(show_spinner="Loading High-Precision OCR Models...")
def load_ocr_engines():
    has_gpu = torch.cuda.is_available()
    
    # 1. VietOCR Transformer Recognition Engine
    recognizer = None
    try:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor
        config = Cfg.load_config_from_name("vgg_transformer")
        config["device"] = "cuda:0" if has_gpu else "cpu"
        config["predictor"]["beamsearch"] = False
        recognizer = Predictor(config)
    except Exception as e:
        st.error(f"VietOCR Init Error: {e}")
        recognizer = None

    # 2. Text Detection Engine (PaddleOCR with OpenCV auto-fallback)
    detector = None
    try:
        from paddleocr import TextDetection
        detector = TextDetection(
            model_name="PP-OCRv4_mobile_det",
            device="gpu:0" if has_gpu else "cpu",
            limit_side_len=1536,
            limit_type="max",
            thresh=0.3,
            box_thresh=0.3,
            unclip_ratio=2.0,
        )
    except Exception:
        try:
            from paddleocr import PaddleOCR
            detector = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="vi",
                device="gpu" if has_gpu else "cpu",
            )
        except Exception:
            detector = "cv2_fallback"

    return detector, recognizer, has_gpu

# ─── Image Preprocessing & OCR Functions ───
def classify_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mb = gray.mean()
    if mb > 200 or (gray > 240).sum() / gray.size > 0.30: return "overexposed"
    elif mb < 60 or (gray < 30).sum() / gray.size > 0.30: return "underexposed"
    elif cv2.Laplacian(gray, cv2.CV_64F).var() < 100: return "blurry"
    elif gray.std() < 42: return "low_contrast"
    elif cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].std() > 75: return "complex"
    return "normal"

def gamma_correct(img, g=1.3):
    table = np.array([(i / 255.0) ** (1.0 / g) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def preprocess(img_pil, max_dim=1536, min_dim=800):
    bgr = np.array(img_pil.convert("RGB"))[:, :, ::-1]
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim: scale = max_dim / max(h, w)
    elif max(h, w) < min_dim: scale = min_dim / max(h, w)
    else: scale = 1.0
    bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    cat = classify_image(bgr)
    k = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])

    def clahe(img, clip):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    if cat == "normal": bgr = cv2.filter2D(clahe(bgr, 2.0), -1, k)
    elif cat == "overexposed": bgr = gamma_correct(clahe(bgr, 3.5), 0.7)
    elif cat == "underexposed": bgr = gamma_correct(clahe(bgr, 3.0), 1.5)
    elif cat == "low_contrast":
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.normalize(lab[:, :, 0], None, 0, 255, cv2.NORM_MINMAX)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        bgr = cv2.filter2D(bgr, -1, k)
    elif cat == "blurry":
        g = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3)
        bgr = cv2.addWeighted(bgr, 1.5, g, -0.5, 0)
        bgr = clahe(bgr, 2.0)
    elif cat == "complex":
        bgr = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
        bgr = cv2.filter2D(clahe(bgr, 2.5), -1, k)
    return bgr[:, :, ::-1], cat

def postprocess_ocr(text):
    if not text: return ""
    t = re.sub(r"\s+", " ", text).strip().split()
    if not t: return ""
    r = [t[0]]
    for tok in t[1:]:
        if tok.lower() != r[-1].lower(): r.append(tok)
    return " ".join(r)

def crop_padding(image, bbox, pad=8):
    box = np.array(bbox, dtype=int)
    x_min = max(0, np.min(box[:, 0]) - pad)
    x_max = min(image.shape[1], np.max(box[:, 0]) + pad)
    y_min = max(0, np.min(box[:, 1]) - pad)
    y_max = min(image.shape[0], np.max(box[:, 1]) + pad)
    return image[y_min:y_max, x_min:x_max]

def sort_boxes(boxes):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: b[0][1])
    threshold = np.median([abs(b[2][1] - b[0][1]) for b in boxes]) * 0.3 if len(boxes) > 0 else 10
    sorted_boxes, cur_line, base_y = [], [boxes[0]], boxes[0][0][1]
    for b in boxes[1:]:
        if abs(b[0][1] - base_y) <= threshold: cur_line.append(b)
        else:
            sorted_boxes.extend(sorted(cur_line, key=lambda item: item[0][0]))
            cur_line, base_y = [b], b[0][1]
    if cur_line:
        sorted_boxes.extend(sorted(cur_line, key=lambda item: item[0][0]))
    return sorted_boxes

def poly_area(p):
    p = np.asarray(p, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def detect_regions_opencv(img_cv2, max_boxes=16):
    """Fast, 100% standalone morphological text candidate detector."""
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_img, w_img = img_cv2.shape[:2]
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 16 or h < 8 or (w * h) < 150: continue
        if w > w_img * 0.98 and h > h_img * 0.98: continue
        poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        boxes.append(poly)
        
    # Add header, center, footer semantic crops
    boxes.append([[0, 0], [w_img, 0], [w_img, int(h_img * 0.35)], [0, int(h_img * 0.35)]])
    boxes.append([[0, int(h_img * 0.3)], [w_img, int(h_img * 0.3)], [w_img, int(h_img * 0.7)], [0, int(h_img * 0.7)]])
    boxes.append([[0, int(h_img * 0.65)], [w_img, int(h_img * 0.65)], [w_img, h_img], [0, h_img]])
    
    boxes = sorted(boxes, key=poly_area, reverse=True)[:max_boxes]
    return boxes

def run_ocr_pipeline(img_cv2, detector, recognizer, max_boxes=16, min_crop=8):
    if recognizer is None:
        return "", []

    boxes = []
    # 1. Try Paddle Text Detection
    if detector is not None and detector != "cv2_fallback":
        try:
            if hasattr(detector, "predict"):
                res = list(detector.predict(img_cv2))
                if res and len(res) > 0:
                    boxes = res[0].get("dt_polys", res[0].get("dt_boxes", []))
            elif hasattr(detector, "ocr"):
                res = detector.ocr(img_cv2)
                if res and len(res) > 0 and res[0]:
                    boxes = [line[0] for line in res[0] if line]
        except Exception:
            boxes = []

    # 2. Fallback to OpenCV morphological detector if Paddle failed
    if not boxes:
        boxes = detect_regions_opencv(img_cv2, max_boxes=max_boxes)

    if not boxes:
        return "", []

    # Prune and sort in reading order
    if max_boxes and len(boxes) > max_boxes:
        boxes = sorted(boxes, key=poly_area, reverse=True)[:max_boxes]
    boxes = sort_boxes(boxes)

    texts, bdata = [], []
    for b in boxes:
        crop = crop_padding(img_cv2, b)
        hc, wc = crop.shape[:2]
        if hc < min_crop or wc < min_crop: continue
        try:
            text = recognizer.predict(Image.fromarray(crop)).strip()
            if len(text) > 1 and not text.isdigit():
                texts.append(text)
                bdata.append({"text": text, "area": wc * hc, "box": b})
        except Exception:
            continue

    return postprocess_ocr(" ".join(texts)), bdata

# ─── VLM Prompt Templates ───
PROMPT_GATE = "Ảnh này có chứa chữ đọc được, hoặc có nhãn hàng, logo, tên sản phẩm nào không? Chỉ trả lời đúng một từ: CÓ hoặc KHÔNG."

PROMPT_SUMMARY = """Bạn viết mô tả ngắn cho ảnh thu thập từ mạng xã hội Việt Nam.

## Nhiệm vụ
Viết một đoạn mô tả ngắn gọn bằng tiếng Việt CÓ DẤU về nội dung ảnh.

## Quy tắc
1. Nêu bối cảnh chính: ảnh sản phẩm, banner quảng cáo, ảnh livestream, ảnh review, ảnh chụp màn hình, ảnh phong cảnh, ảnh người...
2. TÊN NHÃN HÀNG và TÊN SẢN PHẨM: giữ nguyên chính tả và nguyên ký tự đúng như xuất hiện trong ảnh.
3. CÁC DÒNG CHỮ KHÁC: diễn đạt lại ý bằng lời của bạn. KHÔNG trích dẫn nguyên văn, KHÔNG dùng dấu ngoặc kép, KHÔNG chép lại từng dòng chữ trong ảnh.
4. Nếu một dòng OCR đọc ra vô nghĩa, sai chính tả nặng hoặc không thành câu: BỎ QUA hoàn toàn. Không nhắc đến, không đoán lại nội dung của nó.
5. Chỉ mô tả những gì CÓ trong ảnh. KHÔNG viết câu nói rằng ảnh không có hoặc thiếu thứ gì.
6. Viết khẳng định. KHÔNG dùng "có thể là", "có vẻ", "dường như", "không rõ", "không thể xác định". Không chắc thì bỏ hẳn chi tiết đó.
7. KHÔNG bịa nhãn hàng, sản phẩm, giá, con số hay chương trình khuyến mãi không nhìn thấy trong ảnh.
8. KHÔNG nhắc đến nhiệm vụ này và KHÔNG dùng các từ như FMCG, OCR, "hình ảnh này", "mô tả".
9. Nếu ảnh KHÔNG có chữ đọc được VÀ KHÔNG có nhãn hàng/sản phẩm nào: trả về chuỗi rỗng, không xuất bất kỳ ký tự nào.
10. Chỉ xuất đúng đoạn mô tả. Không lời dẫn, không ngoặc kép bao quanh, không markdown.

## Ví dụ đúng
Ảnh chụp hộp sữa Vinamilk Flex không đường, bao bì ghi thông tin sữa tươi tiệt trùng.
Banner quảng cáo chương trình mua 2 tặng 1 của Nestlé Milo trên nền xanh.
Ảnh cận cảnh chai nước giặt Comfort hương Ban Mai đặt trên nền trắng.
Ảnh chụp màn hình livestream bán mỹ phẩm, người bán giới thiệu kem dưỡng da.
Ảnh chụp một nhóm học sinh trước sân trường trong ngày khai giảng.

## Kết quả OCR của ảnh này (chỉ để tham khảo, có thể sai):
__OCR_CONTEXT__

Nhắc lại: nếu ảnh không có chữ đọc được và không có nhãn hàng/sản phẩm, trả về chuỗi rỗng.
Viết mô tả ngay:"""

def build_ocr_context(ocr_text, box_data=None, n_ctx=16):
    if not ocr_text or not ocr_text.strip():
        return "Không phát hiện chữ nào trong ảnh."
    if not box_data:
        return "Chữ phát hiện được: " + ocr_text
    ranked = sorted(box_data, key=lambda b: b["area"], reverse=True)[:n_ctx]
    lines = [f'{i}. {b["text"]}' for i, b in enumerate(ranked, 1)]
    return "Chữ phát hiện được (theo độ nổi bật):\n" + "\n".join(lines)

def encode_image(img_pil, max_side=1024):
    img = img_pil.copy()
    img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

def call_vlm_api(client, model_name, b64_img, prompt, max_tokens=384, temperature=0.0):
    resp = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=120,
        extra_body={"repetition_penalty": 1.05},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64_img}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    return resp.choices[0].message.content or ""

# ─── Helper for In-Memory Image File Abstraction ───
class MemoryImageFile:
    def __init__(self, filename, byte_data):
        self.name = filename
        self._data = byte_data

    def read(self):
        return self._data

    def seek(self, pos):
        pass

def extract_images_from_zip(zip_file_bytes):
    """Extracts all image files from a ZIP archive in-memory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    extracted_images = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_file_bytes)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                filename = os.path.basename(info.filename)
                # Ignore macOS metadata or hidden files
                if filename.startswith(".") or filename.startswith("__MACOSX"):
                    continue
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_exts:
                    data = zf.read(info.filename)
                    extracted_images.append(MemoryImageFile(filename, data))
    except Exception as e:
        st.error(f"Error reading ZIP file: {e}")
    
    # Sort files naturally by filename
    extracted_images.sort(key=lambda x: x.name)
    return extracted_images

# ─── Sidebar Configuration ───
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sun.png", width=64)
    st.title("Team ArrayOfSunshine")
    st.caption("The 2nd URA Hackathon 2026 — Phase 3")
    st.divider()

    st.subheader("⚙️ Pipeline Mode")
    run_mode = st.radio(
        "Select Execution Engine",
        ["🚀 Full Pipeline (OCR + Remote VLM Server)", "⚡ Standalone OCR Fallback (Offline Mode - No GPU Server)"],
        index=0,
        help="Use Standalone OCR if your remote GPU server is currently offline."
    )

    st.divider()
    st.subheader("🖥️ Local Hardware Info")
    if torch.cuda.is_available():
        st.success(f"NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("Local Engine: High-Performance CPU Mode (AMD Ryzen / Intel)")

    st.divider()
    st.subheader("🌐 Remote VLM Server")
    vllm_url = st.text_input(
        "API Base URL",
        value=st.session_state.get("vllm_url", "https://delay-buffer-unnerve.ngrok-free.dev/v1"),
        help="Enter the ngrok public URL or remote server IP (e.g. http://<IP>:8000/v1)"
    )
    model_name = st.text_input(
        "Model ID",
        value="Qwen/Qwen3-VL-8B-Instruct"
    )
    api_key = st.text_input("API Key (Optional)", value="EMPTY", type="password")

    if st.button("🔄 Test Backend Connection"):
        try:
            client_test = OpenAI(base_url=vllm_url, api_key=api_key)
            models = client_test.models.list()
            st.success(f"Connected! Available: {[m.id for m in models]}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            st.caption("Tip: Switch to 'Standalone OCR Fallback' mode to test locally without a GPU server!")

    st.divider()
    st.subheader("🎛️ Inference Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    max_tokens = st.slider("Max Output Tokens", min_value=64, max_value=1024, value=384, step=32)
    max_ocr_boxes = st.slider("Max OCR Boxes Context", min_value=4, max_value=32, value=16, step=2)

# Initialize OpenAI Client
vllm_client = OpenAI(base_url=vllm_url, api_key=api_key)

# ─── Main UI Tabs ───
st.markdown('<div class="main-header">☀️ FMCG Multimodal Image Summarization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated End-to-End OCR & Vision-Language Model Pipeline (VietOCR + Qwen3-VL-8B)</div>', unsafe_allow_html=True)

tab_single, tab_batch, tab_arch = st.tabs(["📸 Single Image Demo", "⚡ Batch Image & ZIP Processing", "ℹ️ System Architecture"])

detector, recognizer, has_gpu = load_ocr_engines()

def generate_ocr_fallback_summary(ocr_text, box_data, category):
    if not ocr_text or not ocr_text.strip():
        return ""
    
    unique_tokens = []
    if box_data:
        for b in box_data:
            t = b["text"].strip()
            if t and t.lower() not in [u.lower() for u in unique_tokens]:
                unique_tokens.append(t)
    else:
        unique_tokens = ocr_text.strip().split()

    lines = [
        f"📸 Bối cảnh: Ảnh sản phẩm (Độ sáng: {category.upper()})",
    ]
    
    if unique_tokens:
        lines.append(f"🏷️ Nhãn hiệu / Tiêu đề: {unique_tokens[0]}")
        if len(unique_tokens) > 1:
            lines.append("📝 Chi tiết văn bản trên bao bì:")
            for idx, token in enumerate(unique_tokens[1:8], 1):
                lines.append(f"   • {token}")
    else:
        lines.append(f"📝 Chữ nhận diện được: {ocr_text}")
        
    return "\n".join(lines)

# ─── Tab 1: Single Image Demo ───
with tab_single:
    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.subheader("1. Upload FMCG Product Image")
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)...", type=["jpg", "jpeg", "png"], key="single_uploader")
        
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")
            st.image(image_pil, caption=f"Uploaded: {uploaded_file.name} ({image_pil.width}x{image_pil.height})", use_container_width=True)
            
            run_btn = st.button("🚀 Generate Vietnamese Summary", type="primary", use_container_width=True)

    with col_res:
        st.subheader("2. Real-Time Pipeline Results")
        if uploaded_file is not None and "run_btn" in locals() and run_btn:
            start_time = time.time()

            with st.status("Processing Pipeline...", expanded=True) as status:
                # Step 1: Preprocessing
                st.write("🔧 Step 1: Adaptive Preprocessing (LAB CLAHE & Sharpen)...")
                img_cv2, category = preprocess(image_pil)
                st.markdown(f'Category Detected: <span class="status-badge badge-category">{category.upper()}</span>', unsafe_allow_html=True)

                # Step 2: OCR Detection & Recognition
                st.write("🔍 Step 2: OCR Text Detection & Recognition (VietOCR)...")
                ocr_time_start = time.time()
                ocr_text, box_data = run_ocr_pipeline(img_cv2, detector, recognizer, max_boxes=max_ocr_boxes)
                ocr_elapsed = time.time() - ocr_time_start
                st.write(f"✓ OCR finished in {ocr_elapsed:.2f}s (Found {len(box_data)} text regions)")

                # Step 3: Visual Bounding Boxes
                if box_data:
                    preview_img = image_pil.copy()
                    draw = ImageDraw.Draw(preview_img)
                    h_cv, w_cv = img_cv2.shape[:2]
                    sx, sy = image_pil.width / w_cv, image_pil.height / h_cv
                    for item in box_data:
                        poly = [(pt[0] * sx, pt[1] * sy) for pt in item["box"]]
                        draw.polygon(poly, outline="#2563eb", width=3)
                    with st.expander("👁️ View Detected OCR Bounding Boxes & Text", expanded=True):
                        st.image(preview_img, caption="Detected Text Bounding Boxes", use_container_width=True)
                        st.json([{"rank": idx+1, "text": b["text"], "area_px": b["area"]} for idx, b in enumerate(box_data)])

                # Step 4: VLM Inference / Fallback
                final_summary = ""
                used_fallback = False

                if "Standalone OCR" in run_mode:
                    st.write("⚡ Standalone OCR Mode: Generating description directly from OCR...")
                    final_summary = generate_ocr_fallback_summary(ocr_text, box_data, category)
                    used_fallback = True
                else:
                    st.write("🧠 Step 3: Calling Remote VLM Server...")
                    b64_img = encode_image(image_pil)
                    try:
                        # OCR Gated Empty Check
                        if not (ocr_text or "").strip():
                            st.write("⚠️ No text detected by OCR. Executing Gate Empty Check...")
                            gate_raw = call_vlm_api(vllm_client, model_name, b64_img, PROMPT_GATE, max_tokens=4, temperature=0.0).strip().upper()
                            is_empty = gate_raw.replace("Ô", "O").lstrip("*# ").startswith("KHONG")
                            if is_empty:
                                final_summary = ""
                                st.write("Gate check result: KHÔNG (Zero text/branding detected) -> Empty Output")
                            else:
                                context = build_ocr_context(ocr_text, box_data, n_ctx=max_ocr_boxes)
                                prompt = PROMPT_SUMMARY.replace("__OCR_CONTEXT__", context)
                                final_summary = call_vlm_api(vllm_client, model_name, b64_img, prompt, max_tokens=max_tokens, temperature=temperature)
                        else:
                            context = build_ocr_context(ocr_text, box_data, n_ctx=max_ocr_boxes)
                            prompt = PROMPT_SUMMARY.replace("__OCR_CONTEXT__", context)
                            final_summary = call_vlm_api(vllm_client, model_name, b64_img, prompt, max_tokens=max_tokens, temperature=temperature)
                    except Exception as e:
                        st.warning(f"⚠️ Remote VLM server unreachable ({e}).")
                        st.info("🔄 Auto-Fallback: Successfully generated description using local OCR text!")
                        final_summary = generate_ocr_fallback_summary(ocr_text, box_data, category)
                        used_fallback = True

                total_elapsed = time.time() - start_time
                status.update(label=f"✅ Pipeline Completed in {total_elapsed:.2f}s", state="complete", expanded=False)

            # Display Final Description
            st.markdown("### 📝 Generated Description:")
            if used_fallback:
                st.markdown('<span class="status-badge badge-warning">⚡ Standalone OCR Fallback Result</span>', unsafe_allow_html=True)
            
            if final_summary.strip() == "":
                st.info('"" (Exact empty string returned - No relevant text/branding detected)')
            else:
                formatted_html = "<br>".join([f"<div style='margin-bottom: 4px;'>{line}</div>" if line.strip().startswith(('📸', '🏷️', '📝')) else f"<div style='margin-left: 16px; margin-bottom: 2px;'>{line}</div>" for line in final_summary.split('\n')])
                st.markdown(f'<div class="result-box">{formatted_html}</div>', unsafe_allow_html=True)
            
            st.caption(f"⏱️ Total Latency: {total_elapsed:.2f}s (OCR: {ocr_elapsed:.2f}s | Processing: {total_elapsed-ocr_elapsed:.2f}s)")

# ─── Tab 2: Batch Image & ZIP Processing ───
with tab_batch:
    st.subheader("⚡ High-Throughput Batch & ZIP Processing")
    st.markdown("Upload multiple images or a **ZIP file** containing images to run parallel multi-threaded inference and export `submission.jsonl`.")

    uploaded_batch_files = st.file_uploader(
        "Upload images (JPG, PNG) or a ZIP archive containing images...",
        type=["jpg", "jpeg", "png", "zip"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    # Process uploaded items: direct images + unpacked zip contents
    all_batch_images = []
    if uploaded_batch_files:
        for f in uploaded_batch_files:
            if f.name.lower().endswith(".zip"):
                with st.spinner(f"📦 Extracting images from archive {f.name}..."):
                    extracted = extract_images_from_zip(f.getvalue())
                    all_batch_images.extend(extracted)
                    st.success(f"✓ Extracted {len(extracted)} images from **{f.name}**")
            else:
                all_batch_images.append(f)

        # Deduplicate by filename
        seen_names = set()
        deduped_images = []
        for img_file in all_batch_images:
            if img_file.name not in seen_names:
                seen_names.add(img_file.name)
                deduped_images.append(img_file)
        
        # Sort naturally by filename
        deduped_images.sort(key=lambda x: x.name)
        all_batch_images = deduped_images

    col_b_cfg, col_b_run = st.columns([1, 1])
    with col_b_cfg:
        n_workers = st.slider("Concurrent Worker Threads", min_value=1, max_value=16, value=6, help="Number of parallel worker threads")
    
    if all_batch_images:
        st.info(f"Loaded total **{len(all_batch_images)} images** ready for batch summarization.")
        
        with st.expander(f"👁️ View Image List ({len(all_batch_images)} items)", expanded=False):
            st.write([img.name for img in all_batch_images[:50]])
            if len(all_batch_images) > 50:
                st.caption(f"... and {len(all_batch_images) - 50} more images")

        if st.button(f"🚀 Process {len(all_batch_images)} Images Concurrently", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            batch_results = []
            
            t_batch_start = time.time()

            def process_single_batch_item(uploaded_img):
                img_id = os.path.splitext(uploaded_img.name)[0]
                try:
                    if hasattr(uploaded_img, "getvalue"):
                        raw_data = uploaded_img.getvalue()
                    elif hasattr(uploaded_img, "read"):
                        raw_data = uploaded_img.read()
                    else:
                        raw_data = uploaded_img._data

                    img_pil = Image.open(io.BytesIO(raw_data)).convert("RGB")
                    img_cv2, cat = preprocess(img_pil)
                    ocr_txt, bdata = run_ocr_pipeline(img_cv2, detector, recognizer, max_boxes=max_ocr_boxes)
                    
                    if "Standalone OCR" in run_mode:
                        summary = generate_ocr_fallback_summary(ocr_txt, bdata, cat)
                        return {"image_id": img_id, "summary": summary}

                    b64 = encode_image(img_pil)
                    try:
                        if not (ocr_txt or "").strip():
                            gate_raw = call_vlm_api(vllm_client, model_name, b64, PROMPT_GATE, max_tokens=4, temperature=0.0).strip().upper()
                            if gate_raw.replace("Ô", "O").lstrip("*# ").startswith("KHONG"):
                                return {"image_id": img_id, "summary": ""}

                        ctx = build_ocr_context(ocr_txt, bdata, n_ctx=max_ocr_boxes)
                        prompt = PROMPT_SUMMARY.replace("__OCR_CONTEXT__", ctx)
                        summary = call_vlm_api(vllm_client, model_name, b64, prompt, max_tokens=max_tokens, temperature=temperature)
                        return {"image_id": img_id, "summary": summary.strip()}
                    except Exception:
                        summary = generate_ocr_fallback_summary(ocr_txt, bdata, cat)
                        return {"image_id": img_id, "summary": summary}
                except Exception as e:
                    return {"image_id": img_id, "summary": "", "error": str(e)}

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_single_batch_item, f): f for f in all_batch_images}
                completed = 0
                for future in as_completed(futures):
                    res = future.result()
                    batch_results.append(res)
                    completed += 1
                    progress_bar.progress(completed / len(all_batch_images))
                    status_text.text(f"Processed {completed}/{len(all_batch_images)} images...")

            # Sort batch results back to alphabetical order by image_id
            batch_results.sort(key=lambda x: x["image_id"])

            t_batch_elapsed = time.time() - t_batch_start
            throughput = len(all_batch_images) / max(0.01, t_batch_elapsed)
            st.success(f"🎉 Batch Finished in {t_batch_elapsed:.2f}s! Throughput: {throughput:.2f} images/second")

            # Display Preview Table
            df_res = pd.DataFrame(batch_results)
            st.dataframe(df_res[["image_id", "summary"]], use_container_width=True)

            # Download buttons
            jsonl_str = "\n".join([json.dumps({"image_id": r["image_id"], "summary": r["summary"]}, ensure_ascii=False) for r in batch_results]) + "\n"
            st.download_button(
                label="📥 Download submission.jsonl",
                data=jsonl_str.encode("utf-8"),
                file_name="submission.jsonl",
                mime="application/jsonl",
                type="primary",
                use_container_width=True
            )

# ─── Tab 3: System Architecture & Documentation ───
with tab_arch:
    st.subheader("🏛️ Phase 3 Decoupled Client-Server Architecture")
    st.markdown("""
    The system is built on a **Decoupled Architecture**:
    
    1. **Local Client (Streamlit App):**
       - **Adaptive Preprocessing:** Classifies image into 6 illumination/contrast categories in LAB space.
       - **Text Detection:** PP-OCR / Hybrid Morphological Region Detector.
       - **Text Recognition:** VietOCR (`vgg_transformer` with greedy decoding).
       - **Box Sorting & Pruning:** Top-16 boxes ordered top-to-bottom, left-to-right.
       - **ZIP & Batch Support:** In-memory extraction and concurrent worker threads.
       - **Offline Fallback Engine:** Automatically converts OCR text into structured descriptions if the remote VLM GPU is unreachable.
    
    2. **Remote GPU Server (H100 MIG 20GB):**
       - **Inference Engine:** Serves `Qwen/Qwen3-VL-8B-Instruct` with the fine-tuned FMCG LoRA adapter.
       - **Reverse Tunneling via ngrok:** Secure HTTPS tunneling.
       - **Continuous Batching:** Concurrent requests handled seamlessly.
    
    3. **Anti-Hallucination Guardrails:**
       - **OCR-Gated Empty Check:** Binary check before summarization.
       - **10 Strict Prompt Rules:** Prohibiting brand fabrication, hallucination, and quotes.
    """)
    st.info("Developed by Team ArrayOfSunshine for The 2nd URA Hackathon 2026.")
