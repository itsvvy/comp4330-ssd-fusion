from collections import defaultdict

import numpy as np
import pytest

from tools.size_stratified_eval import size_bin, scores_for_subset


def records_with(box, difficult=False):
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    area = width * height
    target = {
        "box": np.asarray(box, dtype=float),
        "difficult": difficult,
        "area": area,
        "bin": size_bin(area),
    }
    return {"img": defaultdict(list, {"car": [target]})}


def detection(box, score=1.0, class_name="car"):
    return {
        "image_id": "img",
        "class_name": class_name,
        "box": np.asarray(box, dtype=float),
        "score": score,
    }


@pytest.mark.parametrize(
    ("area", "expected"),
    [
        (32**2 - 1, "small"),
        (32**2, "medium"),
        (96**2 - 1, "medium"),
        (96**2, "large"),
    ],
)
def test_size_boundaries(area, expected):
    assert size_bin(area) == expected


def test_perfect_detection():
    box = [10, 10, 30, 30]
    result = scores_for_subset(
        records_with(box), [detection(box)], {"car"}, {"small"}
    )
    assert result["ap"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["gt_count"] == 1


def test_detection_matched_to_other_bin_is_ignored():
    large = [10, 10, 120, 120]
    small = [150, 150, 170, 170]
    records = records_with(large)
    records["img"]["person"].append(
        {
            "box": np.asarray(small, dtype=float),
            "difficult": False,
            "area": 400.0,
            "bin": "small",
        }
    )
    result = scores_for_subset(
        records,
        [
            detection(large, score=0.9),
            detection(small, score=0.8, class_name="person"),
        ],
        {"car", "person"},
        {"small"},
    )
    assert result["gt_count"] == 1
    assert result["ignored_detections"] == 1
    assert result["ap"] == pytest.approx(1.0)


def test_unmatched_detection_is_false_positive():
    target = [10, 10, 30, 30]
    wrong = [100, 100, 120, 120]
    result = scores_for_subset(
        records_with(target), [detection(wrong)], {"car"}, {"small"}
    )
    assert result["ap"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)


def test_difficult_target_is_ignored():
    box = [10, 10, 30, 30]
    result = scores_for_subset(
        records_with(box, difficult=True),
        [detection(box)],
        {"car"},
        {"small"},
    )
    assert result is None
