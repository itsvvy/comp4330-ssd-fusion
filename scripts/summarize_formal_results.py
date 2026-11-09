#!/usr/bin/env python3
"""Collect the latest formal run and evaluation metrics into one summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("m0_ft", "control", "m1_fusion")
ARM_LABELS = {
    "m0_ft": "M0-FT",
    "control": "C",
    "m1_fusion": "M1",
}


def latest_directory(root: Path, arm: str) -> Path:
    candidates = sorted(
        path for path in (root / arm).glob("*") if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"no run directory for {arm} under {root}")
    return candidates[-1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(run_root: Path, eval_root: Path) -> dict:
    result = {}
    for arm in ARMS:
        run_dir = latest_directory(run_root, arm)
        eval_dir = latest_directory(eval_root, arm)
        model_info = read_json(run_dir / "model_info.json")
        overall = read_json(eval_dir / "metrics.json")
        size_metrics = read_json(eval_dir / "size_metrics.json")
        result[arm] = {
            "run_dir": str(run_dir),
            "eval_dir": str(eval_dir),
            "checkpoint": str(run_dir / "last.pt"),
            "model": model_info,
            "overall_map_0.5": overall["map_0.5"],
            "fps": overall["fps"],
            "latency_ms_mean": overall["latency_ms_mean"],
            "input_space_overall_ap_0.5": size_metrics["overall"]["ap"],
            "input_space_overall_recall_0.5": size_metrics["overall"][
                "recall"
            ],
            "size_bins": size_metrics["size_bins"],
        }
    return result


def markdown(summary: dict) -> str:
    lines = [
        "# Formal 6k Results",
        "",
        "All arms use the same 6,000-iteration budget and seed. The differences",
        "below are descriptive single-seed course-project results, not evidence",
        "of statistical significance.",
        "",
        "| Arm | Overall mAP@0.5 | Small AP | Medium AP | Large AP | FPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = summary[arm]
        bins = item["size_bins"]
        lines.append(
            f"| {ARM_LABELS[arm]} | {item['overall_map_0.5']:.4f} | "
            f"{bins['small']['ap']:.4f} | "
            f"{bins['medium']['ap']:.4f} | "
            f"{bins['large']['ap']:.4f} | {item['fps']:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Arm | Small recall | Medium recall | Large recall | Parameters |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        item = summary[arm]
        bins = item["size_bins"]
        lines.append(
            f"| {ARM_LABELS[arm]} | "
            f"{bins['small']['recall']:.4f} | "
            f"{bins['medium']['recall']:.4f} | "
            f"{bins['large']['recall']:.4f} | "
            f"{item['model']['total_parameters']:,} |"
        )

    lines.extend(
        [
            "",
            "| Comparison | Δ overall mAP | Δ small AP | Δ medium AP | Δ large AP |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    comparisons = (
        ("C - M0-FT", "control", "m0_ft"),
        ("M1 - M0-FT", "m1_fusion", "m0_ft"),
        ("M1 - C", "m1_fusion", "control"),
    )
    for label, candidate, reference in comparisons:
        candidate_item = summary[candidate]
        reference_item = summary[reference]
        candidate_bins = candidate_item["size_bins"]
        reference_bins = reference_item["size_bins"]
        lines.append(
            f"| {label} | "
            f"{candidate_item['overall_map_0.5'] - reference_item['overall_map_0.5']:+.4f} | "
            f"{candidate_bins['small']['ap'] - reference_bins['small']['ap']:+.4f} | "
            f"{candidate_bins['medium']['ap'] - reference_bins['medium']['ap']:+.4f} | "
            f"{candidate_bins['large']['ap'] - reference_bins['large']['ap']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "All values come from the latest timestamped run under the supplied",
            "run and evaluation roots. The table does not select checkpoints by",
            "test performance.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize(args.run_root, args.eval_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
