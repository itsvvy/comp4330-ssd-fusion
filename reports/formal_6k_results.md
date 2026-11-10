# Formal 6k Results

All arms use the same 6,000-iteration budget and seed. The differences below
are descriptive single-seed course-project results, not evidence of
statistical significance.

| Arm | Overall mAP at IoU 0.5 | Small AP | Medium AP | Large AP | FPS |
|---|---:|---:|---:|---:|---:|
| M0-FT | 0.7726 | 0.1111 | 0.6295 | 0.8590 | 23.56 |
| C | 0.7727 | 0.1110 | 0.6291 | 0.8587 | 22.83 |
| M1 | 0.7726 | 0.1107 | 0.6292 | 0.8593 | 23.35 |

| Arm | Small recall | Medium recall | Large recall | Parameters |
|---|---:|---:|---:|---:|
| M0-FT | 0.7625 | 0.9120 | 0.9644 | 26,285,486 |
| C | 0.7633 | 0.9118 | 0.9650 | 27,076,765 |
| M1 | 0.7640 | 0.9115 | 0.9644 | 27,076,527 |

| Comparison | Δ overall mAP | Δ small AP | Δ medium AP | Δ large AP |
|---|---:|---:|---:|---:|
| C - M0-FT | +0.0001 | -0.0001 | -0.0004 | -0.0003 |
| M1 - M0-FT | +0.0001 | -0.0004 | -0.0003 | +0.0004 |
| M1 - C | -0.0000 | -0.0003 | +0.0001 | +0.0006 |

## Interpretation

The released SSD300 baseline reproduces 77.66% overall VOC2007 mAP at IoU 0.5 before
the controlled 6k fine-tuning runs. Fine-tuning M0 for 6k iterations preserves
overall mAP at 77.26%, so the shared comparison budget is usable.

C and M1 are essentially indistinguishable from M0-FT at this budget. The
parameter-matched C result shows that adding branch capacity alone does not
produce a measurable small-object gain. M1 versus C is the cleaner test of the
cross-scale path, and it likewise shows no meaningful advantage: overall mAP
is unchanged to four decimal places, small/medium AP differ by less than 0.06
percentage points, and the small sweep difference is not evidence of an
improvement under one seed.

M1 has slightly higher small-object recall than M0-FT (76.40% versus 76.25%)
but slightly lower small AP. This should not be over-interpreted; at the
observed effect size it is consistent with seed/optimization variation rather
than a reliable ranking or localization gain.

FPS values come from the evaluation loop and vary with runtime conditions.
They are useful as rough deployment-cost context, not a precise benchmark.

## College-project conclusion

The defensible conclusion is a negative but useful result: a lightweight
top-down fusion branch did not measurably improve small-object AP under the
available 6k budget and a single seed. The 6k reduction was appropriate for
the time and compute constraint, while the matched C ablation prevents
attributing a gain to extra parameters.
