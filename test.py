#!/usr/bin/env python3
"""Run one image through a trained SSD model and save a visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data import MEANS, VOC_CLASSES, BaseTransform
from ssd import build_ssd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="demo.jpg")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--variant",
        choices=["baseline", "control", "fusion"],
        default="baseline",
    )
    parser.add_argument("--control_channels", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = build_ssd(
        "test",
        variant=args.variant,
        control_channels=args.control_channels,
    )
    model.load_weights(args.trained_model)
    model.to(device).eval()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")
    transformed, _, _ = BaseTransform(300, MEANS)(image)
    tensor = (
        torch.from_numpy(transformed)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    with torch.inference_mode():
        detections = model(tensor)[0]

    height, width = image.shape[:2]
    scale = np.array([width, height, width, height])
    for class_index in range(1, detections.size(0)):
        for detection in detections[class_index]:
            score = float(detection[0])
            if score < args.threshold:
                continue
            box = (detection[1:].cpu().numpy() * scale).astype(int)
            label = VOC_CLASSES[class_index - 1]
            cv2.rectangle(
                image,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"{label} {score:.2f}",
                (box[0], max(18, box[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    print("saved:", output)


if __name__ == "__main__":
    main()
