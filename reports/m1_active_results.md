# Forced-Active M1 Fusion

## Purpose

This experiment distinguishes two explanations for the weak M1 result:

1. the fusion branch was too weak and effectively disabled;
2. the fusion idea and training conditions are insufficient.

The branch output is normalized, the pointwise convolution uses a normal
non-zero initialization, the scale is fixed at `0.05`, and VGG16 remains
frozen for all 6,000 iterations.

## Result

| Metric | Published baseline | M0-FT | M1 active |
|---|---:|---:|---:|
| Overall mAP@0.5 | 77.66% | 77.26% | **77.74%** |
| Small AP@0.5 | 10.78% | **11.11%** | 10.93% |
| Small recall@0.5 | 77.05% | 76.25% | **77.12%** |
| Medium AP@0.5 | 63.07% | 62.95% | **63.62%** |
| Large AP@0.5 | 85.97% | 85.90% | 85.93% |
| FPS | 19.06 | 23.56 | 27.92 |

The active branch was verified at the final checkpoint:

- pointwise L1: `5700.51`
- branch delta RMS: `0.0442`
- shallow feature RMS: `0.5748`
- relative branch contribution: approximately `7.69%`

Therefore this run did not self-disable. It produced a small overall gain over
the published baseline and a medium-object gain, but no meaningful small-object
gain.

## Combined with tiled inference

On the 500-image pilot:

| Method | Overall mAP | Small AP |
|---|---:|---:|
| Baseline + tiled | **80.68%** | 36.83% |
| Active M1 + tiled | 80.22% | **39.07%** |

The combination improves the already targeted small-object AP, but lowers
overall mAP relative to the selected baseline + tiled method.

## Conclusion

The original failure is partly an optimization design problem: an additive
residual branch can be shut off, and the repaired zero-gate variant did
eventually close itself.

However, forcing the branch to remain active does not recover a meaningful
small-object gain at 6k. This indicates that the remaining limitation is the
narrow conv7-to-conv4_3 fusion hypothesis and the limited training regime, not
just the gate implementation.

The selected final method remains selective tiled inference.
