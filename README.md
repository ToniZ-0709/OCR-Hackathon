# Team-15 — ArrayOfSunshine

## 🥈 2nd Place Winner — The 2nd URA Hackathon 2026
### OCR & Product Name Extraction Pipeline

> 🏆 **Achievement:** This repository contains the official solution that secured **Rank 2 (Top 2)** on the final leaderboard for the **SMCE Challenge 2026** (RAISE Lab, HCMUT).

Pipeline designed for the **SMCE Challenge 2026** (RAISE Lab, HCMUT). Given social-media
thumbnail images (TikTok, Vietnamese FMCG — Phase 1: Pate & Milk; Phase 2 expands
into cosmetics/skincare, baby formula, and other categories), the system:

1. **OCR** — transcribes all visible text in the image (`ocr_text`).
2. **Product extraction** — identifies the primary brand and product (`brand_name`, `product_name`).

### 📊 Evaluation Metric & Performance
Submission is a CSV; the leaderboard score balances Brand F1, Product F1, and Character Error Rate (CER):

$$Score = 0.40 \times F1_{brand} + 0.35 \times (1 − CER) + 0.25 \times F1_{product}$$

- **Final Leaderboard Standing:** **2nd Place** 🥈

## Pipeline overview

    image
      └─ preprocess()            # smart rescale + CLAHE contrast + mild sharpen (OpenCV)
           └─ PaddleOCR (det)    # text DETECTION only (lang='en'), returns bounding boxes
                └─ Sort_Boxes()  # reading order: top→bottom, left→right
                     └─ Crop_Padding() per box
                          └─ VietOCR (vgg_transformer)   # text RECOGNITION (Vietnamese)
                               └─ postprocess_ocr()       # whitespace + de-dup -> ocr_text
                                    └─ predict_product(ocr_text, box_data) -> (brand_name, product_name)
                                         ├─ extract_product()                  # Layer 1: Regex dictionary (high precision)
                                         │    ├─ safe_ml_predict()             # ML fallback if product line is missing
                                         │    └─ extract_product_by_subtraction()  # heuristic subtraction fallback
                                         ├─ ner_extract_brand()                # Layer 2: Grammar-based NER + blacklist
                                         └─ extract_brand_from_boxes()         # Layer 3: Spatial dominance (largest box area)

**Product extraction** uses a 4-layer hybrid funnel designed to handle both known Phase 1 brands and completely unseen Phase 2 brands:
- **Layer 1 (Regex):** `BRAND_RULES` — a regex brand/product-line dictionary (high precision on known brands). When the regex matches a brand but no product line, the pipeline tries:
  - ML fallback (`ProductPredictor`) to predict the product line if the ML brand matches the regex brand
  - Text subtraction heuristic (`extract_product_by_subtraction`) to extract the product from remaining tokens
- **Layer 2 (NER):** Uses `underthesea` to extract organization entities, filtered through a strict Vietnamese/English blacklist (`_NER_BLACKLIST`). It requires the presence of a known category keyword (`PRODUCT_KEYWORDS`) to validate the extraction. If the NER finds a brand but no product, text subtraction is applied.
- **Layer 3 (Spatial):** A visual heuristic that identifies the brand by assuming it is the most dominant text box (largest area, >1.5× average area), and assigns the next most prominent text as the product name, provided a category keyword is present.
- **Layer 4 (Fallback):** If all layers fail, returns `(" ", " ")`.

---

## Requirements

- **Python** 3.10+
- **GPU**: optional but recommended (the code auto-detects via `torch.cuda.is_available()`).
  On CPU it still runs, just slower.
- **Internet**: required at runtime to `pip install` the OCR and NLP packages and to download
  the model weights on first use.

Python packages (installed by the first notebook cell):

| Package | Notes |
|---|---|
| `paddleocr==2.8.0`, `paddlepaddle` | text detection |
| `vietocr` | Vietnamese text recognition (`vgg_transformer`) |
| `opencv-python` | preprocessing (CLAHE, resize, sharpen) |
| `underthesea` | Vietnamese NLP (NER extraction for unknown brands) |
| `scikit-learn` | Used for fallback ML head |
| `numpy`, `pandas`, `Pillow`, `tqdm`, `torch`, `kagglehub` | core / data / IO |

> **NumPy note:** `paddleocr 2.8.0` expects `numpy < 2.0`. On environments that ship
> NumPy 2.x (e.g. current Kaggle), the imports cell monkeypatches the removed
> `np.sctypes` / `PIL._util.is_directory` symbols so the stack runs as-is. Do **not**
> downgrade NumPy in-session — that breaks `pandas` (ABI mismatch). If you must use
> `numpy < 2.0`, set it before any import and restart the kernel (or Factory Reset).

---

## Setup & run

1. Create a new Kaggle Notebook and **Add Input → Competition Data → The 2nd URA Hackathon**.
   The data mounts under `/kaggle/input/competitions/the-2nd-ura-hackathon/`.
2. **Settings → Internet: On**. Required for pip and
   model-weight downloads.
3. **Settings → Accelerator: GPU** (T4 recommended).
4. Open `hackathon-2nd-ura.ipynb` and **Run All** (top to bottom).
5. The run writes `submission.csv` to the working directory. 

Expected runtime: a few minutes of setup + model download, then ~30–60 min for OCR over
the 2,006 test images (depends on GPU). Progress is checkpointed to
`submission_checkpoint.csv` every 100 images.

### Cell order

| Section | Purpose |
|---|---|
| Installation | `pip install` OCR and NLP packages |
| Imports | libraries + NumPy/PIL compatibility shims |
| Brand Rules (Regex Dictionary) | `BRAND_RULES` definition |
| Load Dataset | `kagglehub` download + auto-discover CSV/image paths |
| Preprocessing and Postprocessing | `preprocess`, `postprocess_ocr` |
| Text Detection using PaddleOCR | build `Detector` |
| Text Recognition using VietOCR | build `recognizer` |
| Crop + Padding + Sort | `Crop_Padding`, `Sort_Boxes` |
| ML Fallback Model | `ProductPredictor` class |
| Core Extraction Pipeline | `predict_product` orchestrator + NER + Spatial Heuristics |
| Main Loop | Iterates over images → builds `results` |
| Postprocessing | format cleanup + exports `submission.csv` |

---

## Output format

`submission.csv` — exactly 4 columns, 2,006 rows, UTF-8, all fields quoted:

| Column | Description |
|---|---|
| `image_id` | Test image id, e.g. `img_2934` |
| `ocr_text` | All visible text, single-spaced (no `\n`/`\t`); `" "` if none |
| `brand_name` | The extracted brand; `" "` if none |
| `product_name` | The extracted product line/name; `" "` if none |

Empty fields are written as a single space `" "` (the metric strips whitespace, and
Kaggle rejects blank cells as null).

---

## Notes & limitations

- **`lang='en'` for detection** is intentional: PaddleOCR is used for box detection
  only; Vietnamese **recognition** is handled by VietOCR.
- **Diacritics** are preserved end-to-end (NFC). The `patê`/`pate` spelling of the input
  is carried through to the product name.
- **Zero-Shot Generalization:** Because Phase 2 introduces completely unseen brands with zero provided training labels, the extraction pipeline relies heavily on spatial dominance (automatically selecting the largest text bounding box) and NER grammar tagging to locate unknown logos.
- **Category Dependency:** To prevent wild hallucinations, the NER and Spatial layers require the presence of a known category keyword within the OCR text. If completely new product categories are introduced in the future, the `PRODUCT_KEYWORDS` list will need to be updated with those new terms.