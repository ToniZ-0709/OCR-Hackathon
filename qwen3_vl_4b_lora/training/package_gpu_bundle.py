from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def rewrite_and_copy(records: list[dict], bundle_root: Path) -> list[dict]:
    output = []
    for item in records:
        item = json.loads(json.dumps(item))
        source = (PROJECT_ROOT / item["images"][0]).resolve()
        destination = bundle_root / "images" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        item["images"] = [f"images/{source.name}"]
        output.append(item)
    return output


def checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(output: Path, include_external_test: bool = False) -> Path:
    final_dir = PROJECT_ROOT / "data" / "final"
    train_path = final_dir / "train_350.jsonl"
    validation_path = final_dir / "validation_50.jsonl"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Run `python pipeline.py export` before packaging")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen_lora_bundle_") as temporary:
        bundle_root = Path(temporary) / "qwen3_vl_4b_lora_gpu_bundle"
        bundle_root.mkdir()
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
        shutil.copytree(PROJECT_ROOT / "training", bundle_root / "training", ignore=ignore)
        shutil.copytree(PROJECT_ROOT / "configs", bundle_root / "configs", ignore=ignore)
        shutil.copytree(PROJECT_ROOT / "prompts", bundle_root / "prompts", ignore=ignore)

        train = rewrite_and_copy(read_jsonl(train_path), bundle_root)
        validation = rewrite_and_copy(read_jsonl(validation_path), bundle_root)
        write_jsonl(bundle_root / "data" / "final" / "train_350.jsonl", train)
        write_jsonl(bundle_root / "data" / "final" / "validation_50.jsonl", validation)
        if include_external_test:
            external_path = final_dir / "external_test_200.jsonl"
            if not external_path.exists():
                raise FileNotFoundError("external_test_200.jsonl is missing")
            external = rewrite_and_copy(read_jsonl(external_path), bundle_root)
            write_jsonl(bundle_root / "data" / "final" / "external_test_200.jsonl", external)
        for name in ("all_400.jsonl", "DATASET_CARD.md", "provenance.json"):
            source = final_dir / name
            if source.exists():
                destination = bundle_root / "data" / "final" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        (bundle_root / "README_GPU.md").write_text(
            "# GPU quick start\n\n"
            "```bash\n"
            "bash training/setup_gpu.sh\n"
            ".venv/bin/python training/preflight.py\n"
            "bash training/train_qlora.sh\n"
            "```\n\n"
            "Gemini and OpenAI API keys are not needed in this bundle.\n",
            encoding="utf-8",
        )
        checksums(bundle_root)
        with tarfile.open(output, "w:gz") as archive:
            archive.add(bundle_root, arcname=bundle_root.name)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "qwen3_vl_4b_lora_gpu_bundle.tar.gz",
    )
    parser.add_argument(
        "--include-external-test",
        action="store_true",
        help="Also copy the 200 held-out images for base-versus-adapter evaluation",
    )
    args = parser.parse_args()
    print(build(args.output.resolve(), include_external_test=args.include_external_test))


if __name__ == "__main__":
    main()
