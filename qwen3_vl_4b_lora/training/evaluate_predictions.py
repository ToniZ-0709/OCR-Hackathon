#!/usr/bin/env python3
"""Evaluate deterministic FMCG validation predictions.

This evaluator intentionally reports only metrics supported by the 50-sample
validation data. The dataset does not contain brand-span annotations, so it
does not claim a brand exact-match score.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
MARKDOWN_RE = re.compile(r"```|^\s{0,3}(?:#{1,6}|[-*+]\s|>\s)", flags=re.MULTILINE)
META_RE = re.compile(
    r"\b(?:the image|this image|in the image|hình ảnh này|ảnh này|"
    r"the task|yêu cầu|tôi không thể|as an ai)\b",
    flags=re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def image_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    if metadata.get("image_id"):
        return str(metadata["image_id"])

    images = row.get("images") or []
    first = images[0] if images else ""
    if isinstance(first, dict):
        first = first.get("path") or first.get("image") or ""
    if first:
        return Path(str(first)).stem

    messages = row.get("messages") or []
    for message in messages:
        for item in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if isinstance(item, dict) and item.get("image"):
                return Path(str(item["image"])).stem
    return ""


def reference_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            return content if isinstance(content, str) else ""
    return str(row.get("labels") or row.get("label") or "")


def prediction_text(row: dict[str, Any]) -> str:
    for key in ("response", "prediction", "predict", "output", "generated_text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
    raise ValueError("Prediction row has no supported response field")


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip().casefold()
    return " ".join(WORD_RE.findall(text))


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(unicodedata.normalize("NFC", text).casefold())


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = tokens(reference)
    pred_tokens = tokens(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    overlap = sum((Counter(ref_tokens) & Counter(pred_tokens)).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def is_format_compliant(raw: str) -> bool:
    stripped = raw.strip()
    if raw != stripped:
        return False
    if not stripped:
        return True
    if "\n" in stripped or "\r" in stripped:
        return False
    if MARKDOWN_RE.search(stripped) or META_RE.search(stripped):
        return False
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'", "`"}:
        return False
    return True


def evaluate(reference_path: Path, prediction_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    references = read_jsonl(reference_path)
    predictions = read_jsonl(prediction_path)

    reference_by_id = {image_id(row): row for row in references}
    prediction_by_id = {image_id(row): row for row in predictions if image_id(row)}
    if len(reference_by_id) != len(references) or "" in reference_by_id:
        raise ValueError("Every reference row must have a unique image_id")

    if len(prediction_by_id) == len(references) and set(prediction_by_id) == set(reference_by_id):
        aligned = [(image, reference_by_id[image], prediction_by_id[image]) for image in reference_by_id]
        alignment = "image_id"
    elif len(predictions) == len(references):
        aligned = [(image_id(ref), ref, pred) for ref, pred in zip(references, predictions)]
        alignment = "row_order"
    else:
        missing = sorted(set(reference_by_id) - set(prediction_by_id))
        raise ValueError(
            f"Cannot align {len(predictions)} predictions to {len(references)} references; "
            f"missing IDs include {missing[:5]}"
        )

    details: list[dict[str, Any]] = []
    tp = fp = fn = tn = 0
    negative_correct = 0
    negative_total = 0
    present_nonempty = 0
    present_total = 0

    for sample_id, reference_row, prediction_row in aligned:
        reference = reference_text(reference_row)
        prediction = prediction_text(prediction_row)
        expected_present = str((reference_row.get("metadata") or {}).get("gate", "")).upper() == "PRESENT"
        if not (reference_row.get("metadata") or {}).get("gate"):
            expected_present = bool(normalized(reference))
        predicted_present = bool(normalized(prediction))

        if expected_present and predicted_present:
            tp += 1
        elif expected_present:
            fn += 1
        elif predicted_present:
            fp += 1
        else:
            tn += 1

        if expected_present:
            present_total += 1
            present_nonempty += int(predicted_present)
        else:
            negative_total += 1
            negative_correct += int(not predicted_present)

        details.append(
            {
                "image_id": sample_id,
                "expected_gate": "PRESENT" if expected_present else "ABSENT",
                "predicted_gate": "PRESENT" if predicted_present else "ABSENT",
                "reference": reference,
                "prediction": prediction,
                "normalized_exact_match": normalized(reference) == normalized(prediction),
                "token_f1": token_f1(reference, prediction),
                "character_similarity": SequenceMatcher(None, normalized(reference), normalized(prediction)).ratio(),
                "format_compliant": is_format_compliant(prediction),
            }
        )

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    present_details = [item for item in details if item["expected_gate"] == "PRESENT"]
    metrics: dict[str, Any] = {
        "reference_path": str(reference_path),
        "prediction_path": str(prediction_path),
        "alignment": alignment,
        "sample_count": len(details),
        "class_counts": {"present": present_total, "absent": negative_total},
        "gate_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "gate_accuracy": safe_ratio(tp + tn, len(details)),
        "gate_precision": precision,
        "gate_recall": recall,
        "gate_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "negative_rejection_accuracy": safe_ratio(negative_correct, negative_total),
        "present_response_rate": safe_ratio(present_nonempty, present_total),
        "normalized_exact_match": statistics.fmean(item["normalized_exact_match"] for item in details),
        "macro_token_f1": statistics.fmean(item["token_f1"] for item in details),
        "macro_character_similarity": statistics.fmean(item["character_similarity"] for item in details),
        "present_normalized_exact_match": statistics.fmean(
            item["normalized_exact_match"] for item in present_details
        ),
        "present_macro_token_f1": statistics.fmean(item["token_f1"] for item in present_details),
        "present_macro_character_similarity": statistics.fmean(
            item["character_similarity"] for item in present_details
        ),
        "output_format_compliance": statistics.fmean(item["format_compliant"] for item in details),
        "metric_note": (
            "Brand Exact Match is not reported because the validation metadata has no brand-span annotations. "
            "Macro token F1 compares the complete generated description with the audited reference."
        ),
    }
    return metrics, details


def markdown(metrics: dict[str, Any]) -> str:
    pct = lambda value: f"{100 * float(value):.2f}%"
    return "\n".join(
        [
            "# Validation metrics",
            "",
            f"Predictions: `{metrics['prediction_path']}`",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Samples | {metrics['sample_count']} |",
            f"| Gate accuracy | {pct(metrics['gate_accuracy'])} |",
            f"| Gate precision | {pct(metrics['gate_precision'])} |",
            f"| Gate recall | {pct(metrics['gate_recall'])} |",
            f"| Gate F1 | {pct(metrics['gate_f1'])} |",
            f"| Negative rejection accuracy | {pct(metrics['negative_rejection_accuracy'])} |",
            f"| Present response rate | {pct(metrics['present_response_rate'])} |",
            f"| Normalized exact match | {pct(metrics['normalized_exact_match'])} |",
            f"| Macro token F1 | {pct(metrics['macro_token_f1'])} |",
            f"| Macro character similarity | {pct(metrics['macro_character_similarity'])} |",
            f"| PRESENT-only normalized exact match | {pct(metrics['present_normalized_exact_match'])} |",
            f"| PRESENT-only macro token F1 | {pct(metrics['present_macro_token_f1'])} |",
            f"| PRESENT-only macro character similarity | {pct(metrics['present_macro_character_similarity'])} |",
            f"| Output format compliance | {pct(metrics['output_format_compliance'])} |",
            "",
            f"Note: {metrics['metric_note']}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    metrics, details = evaluate(args.references, args.predictions)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_name(args.output_prefix.name + "_details").with_suffix(".jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in details), encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(markdown(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
