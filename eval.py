#!/usr/bin/env python3
"""Evaluate a trained model on VOC2007 and save raw detections."""

from __future__ import annotations

import argparse
import json
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from data import (
    MEANS,
    VOC_CLASSES,
    VOCDetection,
    VOCAnnotationTransform,
    BaseTransform,
)
from ssd import build_ssd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model", required=True)
    parser.add_argument("--voc_root", required=True)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument(
        "--variant",
        choices=["baseline", "control", "fusion"],
        default="baseline",
    )
    parser.add_argument("--control_channels", type=int)
    parser.add_argument("--confidence_threshold", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def infer_control_channels(checkpoint_path: str) -> int:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state_dict = checkpoint.get("model", checkpoint)
    key = "control.pointwise.weight"
    if key not in state_dict:
        raise ValueError(
            "cannot infer control channels from a checkpoint without "
            f"{key}"
        )
    return int(state_dict[key].shape[1])


def parse_gt(xml_path: Path) -> list[dict]:
    objects = []
    root = ET.parse(xml_path).getroot()
    for item in root.findall("object"):
        box = item.find("bndbox")
        objects.append(
            {
                "name": item.find("name").text.lower().strip(),
                "difficult": int(item.find("difficult").text),
                "bbox": [
                    int(box.find("xmin").text) - 1,
                    int(box.find("ymin").text) - 1,
                    int(box.find("xmax").text) - 1,
                    int(box.find("ymax").text) - 1,
                ],
            }
        )
    return objects


def voc_ap(recall, precision) -> float:
    ap = 0.0
    for threshold in np.arange(0.0, 1.1, 0.1):
        if np.sum(recall >= threshold) == 0:
            value = 0.0
        else:
            value = np.max(precision[recall >= threshold])
        ap += value / 11.0
    return float(ap)


def class_ap(
    detections,
    image_ids,
    gt_by_image,
    class_name,
    iou_threshold=0.5,
) -> float:
    positives = sum(
        1
        for image_id in image_ids
        for item in gt_by_image[image_id]
        if item["name"] == class_name and not item["difficult"]
    )
    if positives == 0:
        return float("nan")

    rows = []
    for image_id, image_detections in zip(image_ids, detections):
        for detection in image_detections:
            rows.append((float(detection[4]), image_id, detection[:4]))
    rows.sort(key=lambda item: item[0], reverse=True)

    used = {
        image_id: np.zeros(len(gt_by_image[image_id]), dtype=bool)
        for image_id in image_ids
    }
    true_positive = np.zeros(len(rows))
    false_positive = np.zeros(len(rows))

    for detection_index, (_, image_id, box) in enumerate(rows):
        candidates = [
            index
            for index, item in enumerate(gt_by_image[image_id])
            if item["name"] == class_name
        ]
        best_iou = 0.0
        best_index = -1
        for index in candidates:
            gt_box = gt_by_image[image_id][index]["bbox"]
            x1 = max(box[0], gt_box[0])
            y1 = max(box[1], gt_box[1])
            x2 = min(box[2], gt_box[2])
            y2 = min(box[3], gt_box[3])
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            intersection = width * height
            area_detection = max(
                0.0, (box[2] - box[0]) * (box[3] - box[1])
            )
            area_gt = max(
                0.0,
                (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]),
            )
            union = area_detection + area_gt - intersection
            iou = 0.0 if union <= 0 else intersection / union
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_iou >= iou_threshold and best_index >= 0:
            if gt_by_image[image_id][best_index]["difficult"]:
                continue
            if not used[image_id][best_index]:
                true_positive[detection_index] = 1.0
                used[image_id][best_index] = True
            else:
                false_positive[detection_index] = 1.0
        else:
            false_positive[detection_index] = 1.0

    tp = np.cumsum(true_positive)
    fp = np.cumsum(false_positive)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, np.finfo(float).eps)
    return voc_ap(recall, precision)


def evaluate_detections(
    detections, voc_root: Path, split: str, image_ids=None
):
    if image_ids is None:
        image_set = (
            voc_root / "VOC2007" / "ImageSets" / "Main" / f"{split}.txt"
        )
        image_ids = image_set.read_text().splitlines()
    gt_by_image = {
        image_id: parse_gt(
            voc_root / "VOC2007" / "Annotations" / f"{image_id}.xml"
        )
        for image_id in image_ids
    }
    class_aps = {}
    for class_index, class_name in enumerate(VOC_CLASSES, start=1):
        class_detections = detections[class_index]
        class_aps[class_name] = class_ap(
            class_detections, image_ids, gt_by_image, class_name
        )
    valid = [value for value in class_aps.values() if not np.isnan(value)]
    return {
        "map_0.5": float(np.mean(valid)),
        "per_class_ap_0.5": class_aps,
        "images": len(image_ids),
    }


def main():
    args = parse_args()
    voc_root = Path(args.voc_root)
    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    control_channels = args.control_channels
    if args.variant == "control" and control_channels is None:
        control_channels = infer_control_channels(args.trained_model)
    if control_channels is None:
        control_channels = 768

    model = build_ssd(
        "test",
        variant=args.variant,
        control_channels=control_channels,
    )
    model.load_weights(args.trained_model)
    model.to(device).eval()

    dataset = VOCDetection(
        root=str(voc_root),
        image_sets=[("2007", "test")],
        transform=BaseTransform(300, MEANS),
        target_transform=VOCAnnotationTransform(),
    )
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    all_detections = [[[] for _ in range(count)] for _ in range(21)]
    durations = []

    for index in range(count):
        image, _, height, width = dataset.pull_item(index)
        tensor = image.unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(tensor)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)

        for class_index in range(1, 21):
            class_detections = output[class_index]
            keep = class_detections[:, 0] >= args.confidence_threshold
            class_detections = class_detections[keep]
            if class_detections.numel() == 0:
                continue
            boxes = class_detections[:, 1:].cpu().numpy().copy()
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
            scores = class_detections[:, 0].cpu().numpy()[:, None]
            all_detections[class_index][index] = np.concatenate(
                (boxes, scores), axis=1
            ).astype(np.float32)
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"evaluated {index + 1}/{count}", flush=True)

    with (save_folder / "detections.pkl").open("wb") as stream:
        pickle.dump(all_detections, stream, protocol=pickle.HIGHEST_PROTOCOL)

    evaluation_ids = [image_id for _, image_id in dataset.ids[:count]]
    metrics = evaluate_detections(
        all_detections, voc_root, "test", image_ids=evaluation_ids
    )
    metrics["count"] = count
    metrics["latency_ms_mean"] = float(np.mean(durations) * 1000)
    metrics["fps"] = float(1.0 / np.mean(durations))
    (save_folder / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
