#!/usr/bin/env python3
"""Create reproducible summary metrics from an ms-swift training run."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def step_of(row: dict[str, Any]) -> int:
    value = row.get("global_step/max_steps", "0/0")
    return int(str(value).split("/", maxsplit=1)[0])


def safetensors_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    parameter_count = sum(math.prod(tensor["shape"]) for tensor in tensors)
    return {
        "file": str(path),
        "file_size_bytes": path.stat().st_size,
        "tensor_count": len(tensors),
        "parameter_count": parameter_count,
        "dtypes": sorted({tensor["dtype"] for tensor in tensors}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    logging_rows = read_jsonl(args.run_dir / "logging.jsonl")
    training_rows = [row for row in logging_rows if "loss" in row]
    evaluation_by_step = {
        step_of(row): row for row in logging_rows if "eval_loss" in row
    }
    evaluation_rows = [evaluation_by_step[step] for step in sorted(evaluation_by_step)]
    final_rows = [row for row in logging_rows if "train_loss" in row]

    state_candidates = sorted(
        args.run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda path: int(path.parent.name.rsplit("-", maxsplit=1)[1]),
    )
    if not state_candidates:
        raise FileNotFoundError("No trainer_state.json found")
    state = json.loads(state_candidates[-1].read_text(encoding="utf-8"))
    best_path = Path(state["best_model_checkpoint"])
    checkpoint = args.checkpoint or best_path

    previous_step = 0
    weighted_loss_sum = 0.0
    weighted_steps = 0
    for row in training_rows:
        current_step = step_of(row)
        span = current_step - previous_step
        weighted_loss_sum += float(row["loss"]) * span
        weighted_steps += span
        previous_step = current_step

    result = {
        "run_dir": str(args.run_dir),
        "best_checkpoint": str(best_path),
        "best_eval_loss": state.get("best_metric"),
        "global_step": state.get("global_step"),
        "epoch": state.get("epoch"),
        "initial_logged_loss": training_rows[0]["loss"],
        "final_logged_loss": training_rows[-1]["loss"],
        "mean_logged_loss": sum(float(row["loss"]) for row in training_rows) / len(training_rows),
        "step_weighted_mean_logged_loss": weighted_loss_sum / weighted_steps,
        "peak_logged_token_accuracy": max(float(row["token_acc"]) for row in training_rows),
        "final_logged_token_accuracy": training_rows[-1]["token_acc"],
        "evaluation_history": evaluation_rows,
        "final_training_summary": final_rows[-1] if final_rows else None,
        "adapter": safetensors_info(checkpoint / "adapter_model.safetensors"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
