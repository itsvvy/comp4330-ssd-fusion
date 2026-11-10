# M1 Fusion Repair Attempt

## Motivation

The original M1 fusion branch had two optimization problems:

- its output was numerically much smaller than the L2-normalized conv4_3 map;
- with a zero scalar gate, branch parameter gradients were exactly zero on the
  first optimizer step.

The repair experiment normalizes the branch output, initializes the final
pointwise convolution to zero, and starts the scale at one. This preserves
identity at initialization while allowing gradients to reach the final fusion
convolution immediately.

## Result

The repaired model was trained for the same 6,000 iterations as M0-FT.

| Metric | M0-FT | M1 gate-fix | Delta |
|---|---:|---:|---:|
| Overall mAP@0.5 | 77.26% | 76.99% | -0.26 pp |
| Small AP@0.5 | 11.11% | 14.88% | +3.77 pp |
| Small recall@0.5 | 76.25% | 81.41% | +5.16 pp |
| Medium AP@0.5 | 62.95% | 61.52% | -1.43 pp |
| Large AP@0.5 | 85.90% | 84.45% | -1.45 pp |
| FPS | 23.56 | 15.47 | -8.09 |

At 500 iterations the repair also improved small AP from 14.07% to 17.03% on
the same 500-image pilot, but overall mAP fell from 80.29% to 79.72%.

## Why it still does not beat the selected method

The repair successfully lets fusion influence training and improves the
targeted small-object metric. However, the gain is accompanied by a larger
medium/large regression, so overall mAP decreases.

The final checkpoint reveals another issue: the fusion branch delta RMS is
exactly zero. Changing the final scale from 0.5 to 0.75 produces identical
outputs. The optimization eventually shuts the branch off and retains only
side effects learned while the branch was active.

Therefore this is a partial mechanism fix, not a successful final model.

## Decision

Do not replace selective tiled inference with M1 gate-fix. The tiled method:

- improves overall mAP from 77.66% to 78.33%;
- improves small AP from 10.78% to 30.65%;
- remains an inference-only change to the released checkpoint.

The fusion repair remains available on `experiment/m1-gatefix` as a documented
negative/partial-result study.
