import torch

from tools.tiled_eval import (
    map_crop_detections,
    merge_class_detections,
    tile_regions,
)


def test_tile_regions_cover_image_and_overlap():
    regions = tile_regions(1000, 600, grid=2, overlap=0.2)
    assert len(regions) == 4
    assert regions[0][0] == 0
    assert regions[0][1] == 0
    assert regions[-1][2] == 1000
    assert regions[-1][3] == 600
    assert regions[0][2] > regions[1][0]
    assert regions[0][3] > regions[2][1]


def test_map_crop_detections_to_full_image():
    detections = torch.tensor(
        [[0.9, 0.1, 0.2, 0.3, 0.4]], dtype=torch.float32
    )
    mapped = map_crop_detections(detections, (100, 50, 300, 250))
    expected = torch.tensor(
        [[0.9, 120.0, 70.0, 140.0, 130.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(mapped, expected)


def test_merge_class_detections_suppresses_duplicate():
    rows = [
        torch.tensor(
            [
                [0.9, 10.0, 10.0, 50.0, 50.0],
                [0.8, 11.0, 11.0, 51.0, 51.0],
            ]
        ),
        torch.tensor(
            [[0.7, 100.0, 100.0, 140.0, 140.0]]
        ),
    ]
    merged = merge_class_detections(rows, nms_threshold=0.45)
    assert merged.shape == (2, 5)
    assert torch.allclose(merged[0], rows[0][0])
