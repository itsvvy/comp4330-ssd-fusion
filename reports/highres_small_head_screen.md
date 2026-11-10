# High-Resolution Small-Object Head Screen

## Design

This experiment adds a dedicated 75 x 75 detection head built from a fusion of
conv3_3, conv4_3 and conv7. The head uses four priors per location with
12-24 pixel anchors, adding 22,500 priors to the original 8,732.

VGG16 is frozen and only the new high-resolution branch and detection heads are
trained for 1,000 iterations.

## Screening result

The screen uses the first 1,000 VOC2007 test images.

| Metric | M0-FT 1k | High-resolution head 1k |
|---|---:|---:|
| Overall mAP@0.5 | **78.36%** | 68.44% |
| Small AP@0.5 | **13.00%** | 0.10% |
| Small recall@0.5 | **76.81%** | 72.24% |
| Medium AP@0.5 | **63.87%** | 0.61% |
| Large AP@0.5 | **85.95%** | 1.06% |
| Latency | about 45 ms | about 989 ms |
| FPS | about 22.4 | 1.01 |

## Interpretation

The added head does not improve recall and produces many high-scoring false
positives from the 22,500 new priors. This collapses AP even though some true
objects are still recalled.

The dedicated head also needs substantially more calibration than 1,000
iterations, and its global NMS cost makes evaluation about 22 times slower.
Increasing training would cost much more without evidence that the final
accuracy-speed trade-off would beat selective tiled inference.

## Decision

Rejected. Do not merge into `main` or continue toward a full VOC2007 run.
