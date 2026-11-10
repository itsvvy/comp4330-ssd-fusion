#!/usr/bin/env python3
"""Evaluate SSD300 with full-image and overlapping tiled inference."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MEANS, VOCDetection
from eval import (
    evaluate_detections,
    infer_control_channels,
)
from layers.box_utils import nms
from ssd import build_ssd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model", required=True)
    parser.add_argument("--voc_root", required=True)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument(
        "--variant",
        choices=[
            "baseline",
            "control",
            "fusion",
            "fusion_multiscale",
        ],
        default="baseline",
    )
    parser.add_argument("--control_channels", type=int)
    parser.add_argument("--tile_grid", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--include_full", action="store_true")
    parser.add_argument("--tile_max_area", type=float, default=0.0)
    parser.add_argument("--tile_score_scale", type=float, default=1.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.01)
    parser.add_argument("--nms_threshold", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def tile_regions(width, height, grid, overlap):
    if grid < 1:
        raise ValueError("tile_grid must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    tile_width = min(
        width, math.ceil(width / (grid - (grid - 1) * overlap))
    )
    tile_height = min(
        height, math.ceil(height / (grid - (grid - 1) * overlap))
    )
    x_starts = np.linspace(0, width - tile_width, grid).round().astype(int)
    y_starts = np.linspace(0, height - tile_height, grid).round().astype(int)
    return [
        (int(x), int(y), int(x + tile_width), int(y + tile_height))
        for y in np.unique(y_starts)
        for x in np.unique(x_starts)
    ]


def resize_for_ssd(image):
    resized = cv2.resize(image, (300, 300)).astype(np.float32)
    resized -= np.asarray(MEANS, dtype=np.float32)
    resized = resized[:, :, (2, 1, 0)]
    tensor = torch.from_numpy(resized)
    return tensor.permute(2, 0, 1).unsqueeze(0)


def map_crop_detections(detections, region):
    left, top, right, bottom = region
    mapped = detections.clone()
    mapped[:, 1] = detections[:, 1] * (right - left) + left
    mapped[:, 3] = detections[:, 3] * (right - left) + left
    mapped[:, 2] = detections[:, 2] * (bottom - top) + top
    mapped[:, 4] = detections[:, 4] * (bottom - top) + top
    return mapped


def filter_by_resized_area(detections, width, height, max_area):
    if not max_area or not detections.numel():
        return detections
    scale_x = 300.0 / width
    scale_y = 300.0 / height
    box_width = (detections[:, 3] - detections[:, 1]) * scale_x
    box_height = (detections[:, 4] - detections[:, 2]) * scale_y
    return detections[box_width * box_height <= max_area]


def merge_class_detections(detections, nms_threshold, top_k=200):
    nonempty = [item for item in detections if item.numel()]
    if not nonempty:
        return torch.empty((0, 5))
    merged = torch.cat(nonempty, dim=0)
    keep, count = nms(
        merged[:, 1:5],
        merged[:, 0],
        overlap=nms_threshold,
        top_k=top_k,
    )
    selected = merged[keep[:count]]
    return selected[:, [1, 2, 3, 4, 0]]


def main():
    args = parse_args()
    if args.variant == "control" and args.control_channels is None:
        args.control_channels = infer_control_channels(args.trained_model)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = build_ssd(
        "test",
        variant=args.variant,
        control_channels=args.control_channels or 768,
    )
    model.load_weights(args.trained_model)
    model.to(device).eval()

    voc_root = Path(args.voc_root)
    dataset = VOCDetection(
        root=str(voc_root),
        image_sets=[("2007", "test")],
        transform=None,
    )
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    all_detections = [[[] for _ in range(count)] for _ in range(21)]
    durations = []

    for index in range(count):
        image = dataset.pull_image(index)
        height, width = image.shape[:2]
        regions = tile_regions(
            width, height, args.tile_grid, args.overlap
        )
        if args.include_full:
            regions.insert(0, (0, 0, width, height))

        started = time.perf_counter()
        per_class = [[] for _ in range(21)]
        for region_index, (left, top, right, bottom) in enumerate(regions):
            crop = image[top:bottom, left:right]
            tensor = resize_for_ssd(crop).to(device)
            with torch.inference_mode():
                output = model(tensor)[0].cpu()
            for class_index in range(1, 21):
                class_detections = output[class_index]
                keep = (
                    class_detections[:, 0]
                    >= args.confidence_threshold
                )
                class_detections = class_detections[keep]
                if class_detections.numel():
                    mapped = map_crop_detections(
                        class_detections,
                        (left, top, right, bottom),
                    )
                    is_full = args.include_full and region_index == 0
                    if not is_full:
                        mapped = filter_by_resized_area(
                            mapped,
                            width,
                            height,
                            args.tile_max_area,
                        )
                        mapped[:, 0] *= args.tile_score_scale
                    per_class[class_index].append(mapped)

        for class_index in range(1, 21):
            all_detections[class_index][index] = (
                merge_class_detections(
                    per_class[class_index], args.nms_threshold
                ).numpy()
            )
        durations.append(time.perf_counter() - started)
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"evaluated {index + 1}/{count}", flush=True)

    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    with (save_folder / "detections.pkl").open("wb") as stream:
        pickle.dump(
            all_detections, stream, protocol=pickle.HIGHEST_PROTOCOL
        )

    image_ids = [image_id for _, image_id in dataset.ids[:count]]
    metrics = evaluate_detections(
        all_detections, voc_root, "test", image_ids=image_ids
    )
    metrics["count"] = count
    metrics["latency_ms_mean"] = float(np.mean(durations) * 1000)
    metrics["fps"] = float(1.0 / np.mean(durations))
    metrics["tile_grid"] = args.tile_grid
    metrics["tile_overlap"] = args.overlap
    metrics["include_full"] = args.include_full
    metrics["tile_max_area"] = args.tile_max_area
    metrics["tile_score_scale"] = args.tile_score_scale
    (save_folder / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
