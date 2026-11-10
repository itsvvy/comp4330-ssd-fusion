# Experiment History

This file records the full project investigation, not only the final selected
method.

## 1. Baseline diagnosis

The released VGG16-SSD300 VOC0712 checkpoint reproduces 77.66% overall mAP at
IoU 0.5 on VOC2007 test. Its input-space size profile exposes the project's
problem:

- small AP: 10.78%
- medium AP: 63.07%
- large AP: 85.97%

The original hypothesis was that the 38 x 38 conv4_3 map lacks the semantic
information available in conv7 and deeper maps.

## 2. Controlled 6k fusion experiment

All three formal arms used the same VOC0712 training data, released checkpoint,
6,000-iteration schedule, seed, augmentation, and evaluation code.

| Arm | Overall mAP | Small AP | Medium AP | Large AP |
|---|---:|---:|---:|---:|
| M0-FT | 77.26% | 11.11% | 62.95% | 85.90% |
| C control, parameter-matched | 77.27% | 11.10% | 62.91% | 85.87% |
| M1 zero-gate | 77.26% | 11.07% | 62.92% | 85.93% |

The result was neutral. The parameter-only control behaved the same as M0-FT,
and the cross-scale branch did not improve small-object AP. `C` matched the
parameter count of `M1` but not its FLOPs: its branch used about `1,143M` MACs
versus about `574M` for `M1`. The controlled comparison therefore separates
additional capacity from the cross-scale path without claiming compute
matching.

## 3. Why the original M1 branch was weak

Targeted diagnostics showed:

- the branch feature RMS was only about 4.6% of the normalized conv4_3 RMS
  with the released backbone;
- at gate zero, fusion convolution gradients were exactly zero on the first
  optimizer step;
- the gate moved very slowly under the 6k training schedule.

The branch was correct at initialization, but its optimization design made it
much harder to activate than expected.

## 4. Fusion repair attempts

### Normalized gate repair

The branch was normalized, the pointwise convolution was zero-initialized, and
the scale started at one. Full-test result:

- overall mAP: 76.99%
- small AP: 14.88%
- small recall: 81.41%

Small-object accuracy improved, but medium/large AP decreased enough that
overall mAP fell below M0-FT. The final pointwise output collapsed back to
zero, so the final inference path no longer used the fusion branch.

### Forced-active fusion

The final convolution used a non-zero initialization, the branch stayed
normalized, the scale was fixed at 0.05, and VGG16 remained frozen. Final audit
showed the branch was active with about 7.7% relative feature contribution.

Full-test result:

- overall mAP: 77.74%
- small AP: 10.93%
- small recall: 77.12%

The branch stayed active, but it did not materially improve the targeted small
object metric.

## 5. Alternative fusion screens

Two additional structures were tested under a matched 1,000-iteration screen on
the first 1,000 test images.

### Multiscale residual fusion

This fused conv4_3, conv7, conv8_2, and conv9_2 into the conv4_3 space.

- baseline overall / small AP: 78.36% / 13.00%
- fusion overall / small AP: 78.30% / 13.10%
- baseline / fusion small recall: 76.81% / 76.05%

The difference was noise-level, so it was rejected.

### High-resolution auxiliary head

This added a 75 x 75 conv3_3-based detection head with 22,500 additional
priors.

- overall mAP: 68.44%
- small AP: 0.10%
- small recall: 72.24%
- latency: 988.55 ms

The additional priors produced severe false-positive ranking and latency
regressions. The screen was rejected.

## 6. Inference-time alternative

Small recall was `77.05%` despite `10.78%` small AP. At the saved-detection
level, this means many small objects had an IoU 0.5 match in the low-confidence
list, but false positives and low-confidence ranking kept AP low. The result
did not show that every small object was represented well, but it showed that
the main limitation was not simply a missing multi-scale prediction path.
Feature fusion could add semantics, while the 300 x 300 input still discarded
pixels needed to improve tiny-object precision.

Selective tiled inference keeps the model unchanged and runs one full-image
pass plus four overlapping tiles. Only tile detections in the small-object
area range are merged with the full-image detections.

The fixed configuration is:

- one full-image pass;
- a `2 x 2` tile grid with `0.2` overlap;
- tile detections retained only at a maximum resized area of `32 x 32 = 1024`;
- per-class NMS at IoU `0.45`;
- tile score scale `1.0`.

Full VOC2007 test result on all 4,952 images:

| Metric | SSD300 | Selective tiled | Delta |
|---|---:|---:|---:|
| Overall mAP at IoU 0.5 | 77.66% | 78.33% | +0.67 pp |
| Small AP at IoU 0.5 | 10.78% | 30.65% | +19.87 pp |
| Small recall at IoU 0.5 | 77.05% | 90.99% | +13.94 pp |
| Medium AP at IoU 0.5 | 63.07% | 60.54% | -2.54 pp |
| Large AP at IoU 0.5 | 85.97% | 84.97% | -1.00 pp |
| Latency | 52.45 ms/image | 183.59 ms/image | +131.14 ms |
| FPS | 19.06 | 5.45 | -13.62 |

This became the selected method because it produces a large, directly measured
small-object improvement on the complete test set. The 500-image pilot used
the same inference parameters; the table above uses the complete test result.

The main cost is five detector passes per image. Small-object accuracy and
recall increase substantially, while medium/large AP decrease slightly and
throughput falls by about 3.5 times. The final report is
[`tiled_inference_results.md`](tiled_inference_results.md).
