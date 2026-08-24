from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_counts_and_files() -> tuple[list[dict], list[dict]]:
    train = read_jsonl(PROJECT_ROOT / "data" / "final" / "train_350.jsonl")
    validation = read_jsonl(PROJECT_ROOT / "data" / "final" / "validation_50.jsonl")
    if len(train) != 350 or len(validation) != 50:
        raise RuntimeError(f"Expected 350/50 records, found {len(train)}/{len(validation)}")
    train_ids = {item["metadata"]["image_id"] for item in train}
    val_ids = {item["metadata"]["image_id"] for item in validation}
    if train_ids & val_ids:
        raise RuntimeError("Image IDs overlap between train and validation")
    for item in train + validation:
        image_path = (PROJECT_ROOT / item["images"][0]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != item["metadata"]["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {image_path}")
        target = item["messages"][-1]["content"]
        gate = item["metadata"]["gate"]
        if gate == "ABSENT" and target != "":
            raise RuntimeError(f"ABSENT target is not exact empty: {item['metadata']['image_id']}")
        if gate == "PRESENT" and not target.strip():
            raise RuntimeError(f"PRESENT target is empty: {item['metadata']['image_id']}")
    return train, validation


def check_gpu() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 2**30
    if total_gib < 18.0:
        raise RuntimeError(f"At least an 18 GiB visible slice is required, found {total_gib:.2f}")
    return {"name": properties.name, "total_vram_gib": round(total_gib, 3)}


def inspect_token_lengths(records: list[dict], max_length: int, image_tokens: int) -> dict:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if Qwen3VLForConditionalGeneration.__name__ != "Qwen3VLForConditionalGeneration":
        raise RuntimeError("Wrong Qwen model class")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    lengths: list[int] = []
    empty_supervised_lengths: list[int] = []
    max_pixels = image_tokens * 16 * 16

    for item in records:
        image_path = str((PROJECT_ROOT / item["images"][0]).resolve())
        user_text = item["messages"][0]["content"].replace("<image>", "").strip()
        target = item["messages"][1]["content"]
        user_content = [
            {"type": "image", "image": image_path, "max_pixels": max_pixels},
            {"type": "text", "text": user_text},
        ]
        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target},
        ]
        prompt_messages = [{"role": "user", "content": user_content}]
        full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(full_messages, image_patch_size=16)
        full = processor(
            text=[full_text], images=image_inputs, videos=video_inputs,
            padding=False, return_tensors="pt"
        )
        prompt = processor(
            text=[prompt_text], images=image_inputs, videos=video_inputs,
            padding=False, return_tensors="pt"
        )
        full_length = int(full.input_ids.shape[1])
        prompt_length = int(prompt.input_ids.shape[1])
        lengths.append(full_length)
        if target == "":
            supervised = full_length - prompt_length
            empty_supervised_lengths.append(supervised)
            if supervised < 1:
                raise RuntimeError(
                    f"Empty assistant target has no supervised EOS token: {item['metadata']['image_id']}"
                )
        if full_length > max_length:
            raise RuntimeError(
                f"{item['metadata']['image_id']} renders to {full_length} tokens, above max_length={max_length}. "
                "Increase MAX_LENGTH; do not truncate image tokens."
            )
    return {
        "minimum_tokens": min(lengths),
        "maximum_tokens": max(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths), 2),
        "empty_target_minimum_supervised_tokens": min(empty_supervised_lengths) if empty_supervised_lengths else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-token-scan", action="store_true")
    parser.add_argument("--smoke-train", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "configs" / "training.yaml").read_text(encoding="utf-8"))
    train, validation = verify_counts_and_files()
    report = {
        "gpu": check_gpu(),
        "train_count": len(train),
        "validation_count": len(validation),
        "model": config["model"],
    }
    if not args.skip_token_scan:
        report["token_scan"] = inspect_token_lengths(
            train + validation,
            max_length=int(config["max_length"]),
            image_tokens=int(config["image_max_token_num"]),
        )
    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))

    if args.smoke_train:
        environment = os.environ.copy()
        environment["MAX_STEPS"] = "1"
        environment["OUTPUT_DIR"] = str(artifact_dir / "smoke_train")
        subprocess.run(["bash", str(PROJECT_ROOT / "training" / "train_qlora.sh")], check=True, env=environment)


if __name__ == "__main__":
    main()
