# Selective Tiled Inference for Small Objects

## Result

The frozen released SSD300 checkpoint is evaluated with a full-image pass plus
four overlapping image tiles. Tile detections are retained only when their
boxes fall in the small-object area range after mapping back to the 300 x 300
evaluation frame. All detections are combined with per-class NMS.

| Metric | SSD300 baseline | Selective tiled | Delta |
|---|---:|---:|---:|
| Overall mAP@0.5 | 77.66% | **78.33%** | **+0.67 pp** |
| Small AP@0.5 | 10.78% | **30.65%** | **+19.87 pp** |
| Small recall@0.5 | 77.05% | **90.99%** | **+13.94 pp** |
| Medium AP@0.5 | 63.07% | 60.54% | -2.54 pp |
| Large AP@0.5 | 85.97% | 84.97% | -1.00 pp |
| FPS | 19.06 | 5.45 | -13.62 |
| Latency | 52.45 ms | 183.59 ms | +131.14 ms |

The result uses the complete VOC2007 test set (4,952 images). No model weights
were changed and no new training was performed.

## Procedure

- Run the original full image at 300 x 300.
- Run four overlapping tiles at 300 x 300.
- Use 20% overlap between adjacent tiles.
- Map every tile detection back to full-image coordinates.
- Keep only tile detections whose resized area is at most `32 x 32`.
- Merge full and tile detections with per-class NMS at IoU 0.45.

Keeping all tile detections was tested first and was harmful: overall mAP fell
to 63.32% because tiles produced too many duplicated and high-scoring false
detections. Restricting tile detections to the small-object area range made
the method effective.

## Interpretation

The gain is substantive and concentrated where intended. Magnifying each
overlapping crop lets the fixed SSD300 detector see small objects at a more
usable effective scale. Restricting tile detections prevents them from
overwriting the already stronger medium/large detections from the full-image
pass.

This is an accuracy-first inference enhancement, not a free improvement. The
method performs five detector passes per image and therefore reduces measured
FPS by roughly 3.5 times. It is suitable when small-object recall matters more
than real-time throughput.

## Reproduction

```bash
python tools/tiled_eval.py \
  --trained_model /content/assets/weights/ssd300_mAP_77.43_v2.pth \
  --voc_root /content/data/VOCdevkit \
  --save_folder /content/results/full_tiled_small \
  --variant baseline \
  --tile_grid 2 \
  --overlap 0.2 \
  --include_full \
  --tile_max_area 1024

python tools/size_stratified_eval.py \
  --detections /content/results/full_tiled_small/detections.pkl \
  --voc-root /content/data/VOCdevkit \
  --output /content/results/full_tiled_small/size_metrics.json
```

The 500-image pilot used the same parameters and improved overall mAP from
80.29% to 80.68% and small AP from 14.07% to 36.83%.
