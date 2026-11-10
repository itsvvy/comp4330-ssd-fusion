#!/usr/bin/env python3
"""Create the final experiment overview with standard Matplotlib figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "baseline": "#64748b",
    "selected": "#2563eb",
    "control": "#94a3b8",
    "fusion": "#0f9f74",
    "rejected": "#e07a1f",
}


def percent(value: float) -> float:
    return value * 100.0


def grouped_bars(ax, labels, first, second, first_label, second_label):
    positions = np.arange(len(labels))
    width = 0.36
    first_bars = ax.bar(
        positions - width / 2,
        first,
        width,
        color=COLORS["baseline"],
        label=first_label,
    )
    second_bars = ax.bar(
        positions + width / 2,
        second,
        width,
        color=COLORS["selected"],
        label=second_label,
    )
    ax.set_xticks(positions, labels)
    ax.bar_label(first_bars, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(second_bars, fmt="%.2f", padding=2, fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("reports/experiment_history.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
    )
    args = parser.parse_args()

    data = json.loads(args.history.read_text(encoding="utf-8"))
    baseline = data["published_baseline"]
    tiled = data["selected_tiled_inference"]
    formal = data["formal_6k"]
    screens = data["screens"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.1, 1.2, 1]}
    )
    fig.suptitle(
        "COMP4330 small-object detection experiment overview",
        fontsize=17,
        fontweight="bold",
    )

    labels = ["Overall mAP", "Small AP", "Small recall"]
    grouped_bars(
        axes[0],
        labels,
        [
            percent(baseline["overall_map_at_iou_0_5"]),
            percent(baseline["small_ap_at_iou_0_5"]),
            percent(baseline["small_recall_at_iou_0_5"]),
        ],
        [
            percent(tiled["overall_map_at_iou_0_5"]),
            percent(tiled["small_ap_at_iou_0_5"]),
            percent(tiled["small_recall_at_iou_0_5"]),
        ],
        "SSD300",
        "Selective tiled",
    )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Percent")
    axes[0].set_title("A. Final method - full VOC2007 test")
    axes[0].legend(frameon=False, loc="upper left")

    fusion_labels = ["M0-FT", "C", "M1", "M1 gatefix", "M1 active"]
    fusion_items = [
        formal["m0_ft"],
        formal["parameter_matched_control"],
        formal["m1_zero_gate"],
        formal["m1_gatefix"],
        formal["m1_active"],
    ]
    grouped_bars(
        axes[1],
        fusion_labels,
        [percent(item["overall_map_at_iou_0_5"]) for item in fusion_items],
        [percent(item["small_ap_at_iou_0_5"]) for item in fusion_items],
        "Overall mAP",
        "Small AP",
    )
    axes[1].set_ylim(0, 90)
    axes[1].set_ylabel("Percent")
    axes[1].set_title("B. Fusion experiments - full 6k test")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].legend(frameon=False, loc="upper right")

    screen_labels = ["Baseline", "Multiscale", "High-res head"]
    screen_items = [
        screens["baseline"],
        screens["multiscale_residual"],
        screens["high_resolution_head"],
    ]
    grouped_bars(
        axes[2],
        screen_labels,
        [percent(item["overall_map_at_iou_0_5"]) for item in screen_items],
        [percent(item["small_ap_at_iou_0_5"]) for item in screen_items],
        "Overall mAP",
        "Small AP",
    )
    axes[2].set_ylim(0, 90)
    axes[2].set_ylabel("Percent")
    axes[2].set_title("C. 1k screening")
    axes[2].tick_params(axis="x", rotation=15)

    fig.text(
        0.01,
        0.015,
        "Panels A-B: complete 4,952-image VOC2007 test. "
        "Panel C: first 1,000 test images, matched budget.",
        fontsize=9,
        color="#536176",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        output = args.output_dir / f"experiment_overview.{suffix}"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
