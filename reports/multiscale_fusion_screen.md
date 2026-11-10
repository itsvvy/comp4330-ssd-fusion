# Multiscale Fusion 1k Screen

This experiment projects and fuses conv4_3, conv7, conv8_2 and conv9_2 into
the 38 x 38 conv4_3 detector space, followed by a 3 x 3 fusion convolution.
The branch uses a fixed residual scale and VGG16 is frozen.

## Paired result

Both arms use 1,000 training iterations and the first 1,000 VOC2007 test
images for the screening comparison.

| Metric | Baseline 1k | Multiscale fusion 1k |
|---|---:|---:|
| Overall mAP@0.5 | **78.36%** | 78.30% |
| Small AP@0.5 | 13.00% | **13.10%** |
| Small recall@0.5 | **76.81%** | 76.05% |
| Medium AP@0.5 | **63.87%** | 63.78% |
| Large AP@0.5 | 85.95% | **85.99%** |

The apparent small-AP gain is only 0.10 percentage points while small recall
decreases by 0.76 points. This is below a useful or defensible effect size.

## Decision

Rejected. Do not continue to a longer run.
