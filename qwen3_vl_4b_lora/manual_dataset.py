from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MANUAL_SOURCE = PROJECT_ROOT / "data" / "manual_source"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
REVIEW_DIR = PROJECT_ROOT / "data" / "manual_review"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
MASTER_PATH = MANUAL_SOURCE / "dataset_400_master.json"
SHEET_PATH = REVIEW_DIR / "manual_labels.csv"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
AGENT_REPORT_PATHS = (
    REPORT_DIR / "agent_0001_0133.json",
    REPORT_DIR / "agent_0134_0266.json",
    REPORT_DIR / "agent_0267_0400.json",
)
MULTI_AGENT_REVIEWER = "multi-agent-visual-audit"

# The Phase 3 task is FMCG image description. Branded electronics, travel,
# vehicles, appliances, games, agricultural inputs, and other non-FMCG items
# remain ABSENT even when they satisfy a broader generic-product contract.
NON_FMCG_SCOPE_IDS = {
    "priv_d_0135",
    "priv_d_0145",
    "priv_d_0180",
    "priv_d_0210",
    "priv_d_0226",
    "priv_d_0258",
    "priv_d_0259",
    "priv_d_0261",
    "priv_d_0264",
    "priv_d_0282",
    "priv_d_0287",
    "priv_d_0301",
    "priv_d_0320",
    "priv_d_0329",
    "priv_d_0352",
    "priv_d_0354",
    "priv_d_0355",
    "priv_d_0363",
    "priv_d_0380",
    "priv_d_0385",
    "priv_d_0390",
    "priv_d_0395",
}

# These are the 21 changes accepted under the FMCG-only policy. The actual
# corrected Vietnamese text is read from the agent reports, so the reports are
# the immutable audit evidence and this set is the explicit policy gate.
FMCG_CORRECTION_IDS = {
    "priv_d_0030",
    "priv_d_0082",
    "priv_d_0084",
    "priv_d_0088",
    "priv_d_0091",
    "priv_d_0097",
    "priv_d_0124",
    "priv_d_0139",
    "priv_d_0140",
    "priv_d_0144",
    "priv_d_0147",
    "priv_d_0151",
    "priv_d_0172",
    "priv_d_0175",
    "priv_d_0176",
    "priv_d_0191",
    "priv_d_0217",
    "priv_d_0227",
    "priv_d_0229",
    "priv_d_0238",
    "priv_d_0243",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source() -> tuple[list[dict], dict[str, dict]]:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing manual source: {MASTER_PATH}")
    master = read_json(MASTER_PATH)
    if not isinstance(master, list) or len(master) != 400:
        raise RuntimeError(f"Expected 400 source labels, found {len(master)}")
    source = {str(row["image_id"]): row for row in master}
    if len(source) != 400:
        raise RuntimeError("Source image IDs are not unique")
    manifest_rows = read_jsonl(MANIFEST_DIR / "label_pool_400.jsonl")
    if len(manifest_rows) != 400:
        raise RuntimeError(f"Expected 400 manifest rows, found {len(manifest_rows)}")
    manifest = {str(row["image_id"]): row for row in manifest_rows}
    if set(source) != set(manifest):
        missing = sorted(set(manifest) - set(source))
        extra = sorted(set(source) - set(manifest))
        raise RuntimeError(f"Source and manifest IDs differ. Missing={missing[:5]}, extra={extra[:5]}")
    return manifest_rows, source


def source_gate(row: dict) -> str:
    summary = str(row.get("ground_truth_label") or "")
    return "ABSENT" if bool(row.get("is_negative")) or not summary.strip() else "PRESENT"


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "checked", "x"}


def create_review_sheet(reset: bool = False) -> dict[str, int | str]:
    if SHEET_PATH.exists() and not reset:
        raise FileExistsError(f"{SHEET_PATH} already exists. Use --reset to recreate it.")
    manifest_rows, source = load_source()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id",
        "filename",
        "relative_image_path",
        "image_path",
        "source_status",
        "source_error_type",
        "source_visual_description",
        "source_gate",
        "source_summary",
        "final_gate",
        "final_summary",
        "reviewed",
        "reviewer",
        "note",
    ]
    rows: list[dict[str, str]] = []
    for manifest in manifest_rows:
        src = source[manifest["image_id"]]
        summary = str(src.get("ground_truth_label") or "").strip()
        gate = source_gate(src)
        image_path = (PROJECT_ROOT / manifest["relative_path"]).resolve()
        rows.append(
            {
                "image_id": manifest["image_id"],
                "filename": manifest["filename"],
                "relative_image_path": manifest["relative_path"],
                "image_path": str(image_path),
                "source_status": str(src.get("verification_status") or ""),
                "source_error_type": str(src.get("error_type") or ""),
                "source_visual_description": str(src.get("visual_description") or ""),
                "source_gate": gate,
                "source_summary": summary,
                "final_gate": gate,
                "final_summary": summary,
                "reviewed": "FALSE",
                "reviewer": "",
                "note": "",
            }
        )
    with SHEET_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "reviewed": 0, "path": str(SHEET_PATH)}


def load_review_rows() -> list[dict[str, str]]:
    if not SHEET_PATH.exists():
        raise FileNotFoundError(f"Run `python pipeline.py manual-sheet` first. Missing: {SHEET_PATH}")
    with SHEET_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 400:
        raise RuntimeError(f"Expected 400 manual review rows, found {len(rows)}")
    return rows


def load_agent_entries() -> dict[str, dict]:
    """Load and validate the three independent visual-audit reports."""
    entries: dict[str, dict] = {}
    for report_path in AGENT_REPORT_PATHS:
        if not report_path.exists():
            raise FileNotFoundError(f"Missing agent report: {report_path}")
        payload = read_json(report_path)
        report_entries = payload.get("entries") if isinstance(payload, dict) else payload
        if not isinstance(report_entries, list):
            raise RuntimeError(f"Invalid agent report shape: {report_path}")
        for entry in report_entries:
            image_id = str(entry.get("image_id") or "").strip()
            if not image_id:
                raise RuntimeError(f"Agent report has a row without image_id: {report_path}")
            if image_id in entries:
                raise RuntimeError(f"Duplicate agent verdict for {image_id}")
            gate = str(entry.get("gate_verdict") or "").strip().upper()
            summary = str(entry.get("summary_verdict") or "").strip().upper()
            if gate not in {"OK", "EDIT", "PRESENT", "ABSENT"}:
                raise RuntimeError(f"{image_id}: invalid gate_verdict={gate!r}")
            if summary not in {"OK", "EDIT"}:
                raise RuntimeError(f"{image_id}: invalid summary_verdict={summary!r}")
            entry = dict(entry)
            entry["_report"] = report_path.name
            entries[image_id] = entry
    if len(entries) != 400:
        raise RuntimeError(f"Expected 400 agent verdicts, found {len(entries)}")
    return entries


def apply_multi_agent_review() -> dict[str, int | str | list[str]]:
    """Approve the 400 rows from the independent visual-agent reports.

    This is deliberately policy-gated. The agents may identify a branded
    object, but only the explicitly listed FMCG corrections are applied. The
    non-FMCG findings are preserved as ABSENT and recorded in the audit output.
    """
    manifest_rows, source = load_source()
    rows = load_review_rows()
    agent_entries = load_agent_entries()
    row_ids = {row["image_id"].strip() for row in rows}
    if row_ids != set(agent_entries):
        missing = sorted(row_ids - set(agent_entries))
        extra = sorted(set(agent_entries) - row_ids)
        raise RuntimeError(f"Review/report IDs differ. Missing={missing[:5]}, extra={extra[:5]}")
    if FMCG_CORRECTION_IDS & NON_FMCG_SCOPE_IDS:
        raise AssertionError("Correction policy overlaps non-FMCG policy")

    changed: list[dict[str, str]] = []
    retained_scope_flags: list[str] = []
    for row in rows:
        image_id = row["image_id"].strip()
        entry = agent_entries[image_id]
        original_gate = row["final_gate"].strip().upper()
        original_summary = row["final_summary"].strip()
        row["reviewed"] = "TRUE"
        row["reviewer"] = MULTI_AGENT_REVIEWER
        row["note"] = (
            f"Automated three-agent visual audit; FMCG_ONLY policy; source report {entry['_report']}."
        )

        final_gate = original_gate
        final_summary = original_summary
        if image_id in FMCG_CORRECTION_IDS:
            verdict_gate = str(entry.get("gate_verdict") or "").strip().upper()
            corrected = entry.get("corrected_summary")
            if verdict_gate in {"PRESENT", "ABSENT"}:
                final_gate = verdict_gate
            elif verdict_gate == "EDIT":
                # The first agent report uses EDIT for a gate change and
                # communicates the target gate through the corrected text.
                final_gate = "PRESENT" if str(corrected or "").strip() else "ABSENT"
            if corrected is not None:
                final_summary = str(corrected).strip()
            if final_gate == "ABSENT":
                final_summary = ""
            if final_gate == "PRESENT" and not final_summary:
                raise RuntimeError(f"{image_id}: accepted PRESENT correction has empty text")
            changed.append(
                {
                    "image_id": image_id,
                    "from_gate": original_gate,
                    "to_gate": final_gate,
                    "from_summary": original_summary,
                    "to_summary": final_summary,
                    "report": entry["_report"],
                }
            )
        elif image_id in NON_FMCG_SCOPE_IDS:
            retained_scope_flags.append(image_id)
            row["note"] += " Branded non-FMCG finding intentionally retained as ABSENT."
            final_gate = "ABSENT"
            final_summary = ""

        row["final_gate"] = final_gate
        row["final_summary"] = final_summary

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with SHEET_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "review_mode": "multi-agent-visual-audit",
        "scope": "FMCG_ONLY",
        "agent_reports": [path.name for path in AGENT_REPORT_PATHS],
        "audited_rows": len(rows),
        "reviewed_rows": sum(truthy(row["reviewed"]) for row in rows),
        "applied_corrections": len(changed),
        "retained_non_fmcg_scope_flags": len(retained_scope_flags),
        "retained_non_fmcg_ids": retained_scope_flags,
        "present": sum(row["final_gate"] == "PRESENT" for row in rows),
        "absent": sum(row["final_gate"] == "ABSENT" for row in rows),
        "changes": changed,
        "api_keys_required": False,
        "human_in_the_loop": False,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "multi_agent_decision.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "rows": len(rows),
        "reviewed": report["reviewed_rows"],
        "applied_corrections": len(changed),
        "retained_non_fmcg_scope_flags": len(retained_scope_flags),
        "present": report["present"],
        "absent": report["absent"],
        "sheet": str(SHEET_PATH),
        "report": str(REPORT_DIR / "multi_agent_decision.json"),
    }


def manual_status() -> dict[str, int | str]:
    rows = load_review_rows()
    reviewed = sum(truthy(row.get("reviewed", "")) for row in rows)
    present = sum(row.get("final_gate", "").strip().upper() == "PRESENT" for row in rows if truthy(row.get("reviewed", "")))
    absent = reviewed - present
    return {
        "rows": len(rows),
        "reviewed": reviewed,
        "remaining": len(rows) - reviewed,
        "reviewed_present": present,
        "reviewed_absent": absent,
        "sheet": str(SHEET_PATH),
    }


def choose_validation(records: list[dict], target_size: int = 50, seed: int = 42) -> set[str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["duplicate_group"]].append(record)
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    target_present = round(sum(record["gate"] == "PRESENT" for record in records) * target_size / len(records))
    states: dict[tuple[int, int], tuple[str, ...]] = {(0, 0): ()}
    for group_id, group_records in items:
        size = len(group_records)
        present = sum(record["gate"] == "PRESENT" for record in group_records)
        additions: dict[tuple[int, int], tuple[str, ...]] = {}
        for (count, positive), selected in list(states.items()):
            if count + size <= target_size:
                key = (count + size, positive + present)
                additions.setdefault(key, selected + (group_id,))
        for key, selected in additions.items():
            states.setdefault(key, selected)
    candidates = [(key, selected) for key, selected in states.items() if key[0] == target_size]
    if not candidates:
        raise RuntimeError("Could not create an exact 50-image group-safe validation split")
    _, selected_groups = min(candidates, key=lambda item: (abs(item[0][1] - target_present), item[1]))
    return {record["image_id"] for record in records if record["duplicate_group"] in selected_groups}


def qwen_record(record: dict, prompt: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": record["summary"]},
        ],
        "images": [record["image"]],
        "metadata": {
            "image_id": record["image_id"],
            "gate": record["gate"],
            "brands": [],
            "products": [],
            "sha256": record["sha256"],
            "label_source": record.get("label_source", "manual-review"),
        },
    }


def write_checksums(folder: Path) -> None:
    lines = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            lines.append(f"{sha256(path)}  {path.name}")
    (folder / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_manual() -> dict[str, int | str]:
    manifest_rows, _ = load_source()
    manifest = {row["image_id"]: row for row in manifest_rows}
    rows = load_review_rows()
    review_mode = "multi-agent-visual-audit" if all(
        row.get("reviewer", "").strip() == MULTI_AGENT_REVIEWER for row in rows
    ) else "manual-review"
    unresolved = [row["image_id"] for row in rows if not truthy(row.get("reviewed", ""))]
    if unresolved:
        raise RuntimeError(
            f"Import blocked: {len(unresolved)} rows are not reviewed. "
            f"First IDs: {', '.join(unresolved[:10])}"
        )
    labels: list[dict] = []
    for row in rows:
        image_id = row["image_id"].strip()
        item = manifest.get(image_id)
        if item is None:
            raise RuntimeError(f"Unknown image ID in manual sheet: {image_id}")
        gate = row.get("final_gate", "").strip().upper()
        summary = row.get("final_summary", "")
        if gate not in {"PRESENT", "ABSENT"}:
            raise RuntimeError(f"{image_id}: final_gate must be PRESENT or ABSENT")
        if gate == "ABSENT":
            summary = ""
        elif not summary.strip():
            raise RuntimeError(f"{image_id}: PRESENT requires a nonempty final_summary")
        image_path = (PROJECT_ROOT / item["relative_path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if sha256(image_path) != item["sha256"]:
            raise RuntimeError(f"Image checksum changed: {image_id}")
        labels.append(
            {
                "image_id": image_id,
                "filename": item["filename"],
                "image": item["relative_path"],
                "sha256": item["sha256"],
                "duplicate_group": item["duplicate_group"],
                "gate": gate,
                "summary": summary.strip(),
                "brands": [],
                "products": [],
                "label_source": review_mode,
            }
        )

    validation_ids = choose_validation(labels)
    train = [row for row in labels if row["image_id"] not in validation_ids]
    validation = [row for row in labels if row["image_id"] in validation_ids]
    if len(train) != 350 or len(validation) != 50:
        raise AssertionError("Expected a 350/50 split")
    if {row["duplicate_group"] for row in train} & {row["duplicate_group"] for row in validation}:
        raise AssertionError("Duplicate group leaked between train and validation")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    prompt = (PROJECT_ROOT / "prompts" / "qwen_training_v1.txt").read_text(encoding="utf-8").strip()
    write_jsonl(FINAL_DIR / "all_400.jsonl", labels)
    write_jsonl(FINAL_DIR / "train_350.jsonl", [qwen_record(row, prompt) for row in train])
    write_jsonl(FINAL_DIR / "validation_50.jsonl", [qwen_record(row, prompt) for row in validation])
    external_manifest = MANIFEST_DIR / "external_test_200.jsonl"
    external = []
    for item in read_jsonl(external_manifest):
        external.append(
            {
                "image_id": item["image_id"],
                "messages": [{"role": "user", "content": prompt}],
                "images": [item["relative_path"]],
                "metadata": {"sha256": item["sha256"], "duplicate_group": item["duplicate_group"]},
            }
        )
    write_jsonl(FINAL_DIR / "external_test_200.jsonl", external)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_count": 400,
        "train_count": 350,
        "validation_count": 50,
        "external_test_count": len(external),
        "seed": 42,
        "label_source": "data/manual_source/dataset_400_master.json",
        "generator": "Existing Gemini-generated labels; original generation metadata unavailable",
        "verification": "Three independent visual-agent audits recorded in data/reports/",
        "review_mode": "multi-agent-visual-audit" if all(
            row.get("reviewer", "").strip() == MULTI_AGENT_REVIEWER for row in rows
        ) else "manual-review",
        "scope": "FMCG_ONLY",
        "human_reviewed": sum(
            1 for row in rows if row.get("reviewer", "").strip() not in {"", MULTI_AGENT_REVIEWER}
        ),
        "api_keys_required": False,
    }
    (FINAL_DIR / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (FINAL_DIR / "DATASET_CARD.md").write_text(
        "# Qwen3-VL FMCG 400 Dataset Card\n\n"
        "This dataset is imported from the existing 400-image label source and approved by the recorded review workflow.\n\n"
        f"- Total: {len(labels)}\n- Train: {len(train)}\n- Validation: {len(validation)}\n"
        f"- PRESENT: {sum(row['gate'] == 'PRESENT' for row in labels)}\n"
        f"- ABSENT: {sum(row['gate'] == 'ABSENT' for row in labels)}\n"
        f"- Verification: {provenance['review_mode']} under FMCG_ONLY scope\n"
        "- API keys: not required\n"
        "- Labels are model-assisted source labels and must not be described as independent gold annotations.\n",
        encoding="utf-8",
    )
    write_checksums(FINAL_DIR)
    return {
        "all": len(labels),
        "train": len(train),
        "validation": len(validation),
        "external_test": len(external),
        "present": sum(row["gate"] == "PRESENT" for row in labels),
        "absent": sum(row["gate"] == "ABSENT" for row in labels),
        "output": str(FINAL_DIR),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual-only Qwen3-VL dataset preparation")
    sub = parser.add_subparsers(dest="command", required=True)
    sheet = sub.add_parser("manual-sheet", help="Create the editable manual review CSV")
    sheet.add_argument("--reset", action="store_true", help="Recreate the sheet and discard edits")
    sub.add_parser("manual-status", help="Show manual review progress")
    sub.add_parser(
        "multi-agent-review",
        help="Apply the three independent visual-agent audits under the FMCG-only policy",
    )
    sub.add_parser("manual-import", help="Import only rows marked reviewed=true into train/validation files")
    sub.add_parser("export", help="Alias for manual-import")
    args = parser.parse_args()
    if args.command == "manual-sheet":
        result = create_review_sheet(reset=args.reset)
    elif args.command == "manual-status":
        result = manual_status()
    elif args.command == "multi-agent-review":
        result = apply_multi_agent_review()
    else:
        result = import_manual()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
