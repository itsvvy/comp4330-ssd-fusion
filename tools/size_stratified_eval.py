#!/usr/bin/env python3
"""VOC2007 detection evaluation with input-space object-size bins."""

from __future__ import annotations

import argparse
import json
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np


VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def iou(box, candidate) -> float:
    x1 = max(float(box[0]), float(candidate[0]))
    y1 = max(float(box[1]), float(candidate[1]))
    x2 = min(float(box[2]), float(candidate[2]))
    y2 = min(float(box[3]), float(candidate[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_box = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    area_candidate = max(
        0.0, float(candidate[2] - candidate[0])
    ) * max(0.0, float(candidate[3] - candidate[1]))
    union = area_box + area_candidate - inter
    return 0.0 if union <= 0 else inter / union


def voc_ap(recall, precision) -> float:
    return float(
        sum(
            (
                np.max(precision[recall >= threshold])
                if np.any(recall >= threshold)
                else 0.0
            )
            for threshold in np.arange(0.0, 1.1, 0.1)
        )
        / 11.0
    )


def size_bin(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def load_ground_truth(voc_root: Path, image_ids: list[str]):
    records = {}
    for image_id in image_ids:
        root = ET.parse(
            voc_root / "VOC2007" / "Annotations" / f"{image_id}.xml"
        ).getroot()
        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        objects = defaultdict(list)
        for item in root.findall("object"):
            class_name = item.find("name").text.lower().strip()
            box = item.find("bndbox")
            original = np.array(
                [
                    int(box.find("xmin").text) - 1,
                    int(box.find("ymin").text) - 1,
                    int(box.find("xmax").text) - 1,
                    int(box.find("ymax").text) - 1,
                ],
                dtype=np.float64,
            )
            resized = original * np.array(
                [300.0 / width, 300.0 / height, 300.0 / width, 300.0 / height]
            )
            area = max(0.0, resized[2] - resized[0]) * max(
                0.0, resized[3] - resized[1]
            )
            objects[class_name].append(
                {
                    "box": resized,
                    "difficult": int(item.find("difficult").text) == 1,
                    "bin": size_bin(area),
                    "area": area,
                }
            )
        records[image_id] = objects
    return records


def collect_detections(all_detections, image_ids, voc_root):
    records = load_ground_truth(voc_root, image_ids)
    detections = []
    for class_index, class_name in enumerate(VOC_CLASSES, start=1):
        for image_index, image_id in enumerate(image_ids):
            array = all_detections[class_index][image_index]
            if array is None or len(array) == 0:
                continue
            root = ET.parse(
                voc_root / "VOC2007" / "Annotations" / f"{image_id}.xml"
            ).getroot()
            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)
            for row in array:
                original = np.asarray(row[:4], dtype=np.float64)
                resized = original * np.array(
                    [300.0 / width, 300.0 / height, 300.0 / width, 300.0 / height]
                )
                detections.append(
                    {
                        "score": float(row[4]),
                        "image_id": image_id,
                        "class_name": class_name,
                        "box": resized,
                    }
                )
    detections.sort(key=lambda item: item["score"], reverse=True)
    return records, detections


def scores_for_subset(records, detections, classes, bins=None):
    gt_used = {}
    positives = 0
    for image_id, image_records in records.items():
        for class_name in classes:
            for index, item in enumerate(image_records[class_name]):
                if item["difficult"]:
                    continue
                if bins is not None and item["bin"] not in bins:
                    continue
                positives += 1
                gt_used[(image_id, class_name, index)] = False

    if positives == 0:
        return None

    tp = []
    fp = []
    ignored = 0
    for detection in detections:
        if detection["class_name"] not in classes:
            continue
        image_id = detection["image_id"]
        class_name = detection["class_name"]
        best_iou = 0.0
        best_index = None
        for index, target in enumerate(records[image_id][class_name]):
            overlap = iou(detection["box"], target["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_iou < 0.5 or best_index is None:
            tp.append(0.0)
            fp.append(1.0)
            continue
        target = records[image_id][class_name][best_index]
        if target["difficult"]:
            ignored += 1
            continue
        if bins is not None and target["bin"] not in bins:
            ignored += 1
            continue
        key = (image_id, class_name, best_index)
        if gt_used[key]:
            tp.append(0.0)
            fp.append(1.0)
        else:
            tp.append(1.0)
            fp.append(0.0)
            gt_used[key] = True

    tp_array = np.cumsum(np.asarray(tp))
    fp_array = np.cumsum(np.asarray(fp))
    recall = tp_array / positives
    precision = tp_array / np.maximum(
        tp_array + fp_array, np.finfo(float).eps
    )
    return {
        "ap": voc_ap(recall, precision),
        "recall": float(recall[-1]) if len(recall) else 0.0,
        "gt_count": positives,
        "ignored_detections": ignored,
    }


def summarize(records, detections):
    overall = scores_for_subset(records, detections, set(VOC_CLASSES))
    bins = {}
    for name in ("small", "medium", "large"):
        bins[name] = scores_for_subset(
            records, detections, set(VOC_CLASSES), {name}
        )

    per_class = {}
    for class_name in VOC_CLASSES:
        per_class[class_name] = {
            "overall": scores_for_subset(
                records, detections, {class_name}
            ),
            **{
                name: scores_for_subset(
                    records, detections, {class_name}, {name}
                )
                for name in ("small", "medium", "large")
            },
        }

    return {
        "protocol": "VOC07 11-point AP; 300x300 input-space area bins",
        "overall": overall,
        "size_bins": bins,
        "per_class": per_class,
        "image_count": len(records),
        "detection_count": len(detections),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    image_ids = (
        args.voc_root
        / "VOC2007"
        / "ImageSets"
        / "Main"
        / f"{args.split}.txt"
    ).read_text().splitlines()
    with args.detections.open("rb") as stream:
        detections = pickle.load(stream)
    if args.limit:
        image_ids = image_ids[: args.limit]
    if len(detections[0]) != len(image_ids):
        raise SystemExit(
            f"detection/image mismatch: {len(detections[0])} != {len(image_ids)}"
        )

    records, rows = collect_detections(detections, image_ids, args.voc_root)
    metrics = summarize(records, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["size_bins"], indent=2))


if __name__ == "__main__":
    main()
