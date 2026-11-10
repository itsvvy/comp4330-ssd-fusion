# Final Project Summary

## Selected method

`Selective Tiled Inference` is the default and final method. It performs one
full-image SSD300 pass and four overlapping 300 x 300 tile passes. Tile
detections are mapped to full-image coordinates, restricted to the
small-object area range, and merged with the full-image detections.

The model checkpoint is unchanged.

## Frozen full-test result

VOC2007 test, 4,952 images:

| Metric | SSD300 | Selective tiled |
|---|---:|---:|
| Overall mAP at IoU 0.5 | 77.66% | 78.33% |
| Small AP at IoU 0.5 | 10.78% | 30.65% |
| Small recall at IoU 0.5 | 77.05% | 90.99% |
| Medium AP at IoU 0.5 | 63.07% | 60.54% |
| Large AP at IoU 0.5 | 85.97% | 84.97% |
| FPS | 19.06 | 5.45 |

## Fusion experiments

The original 6k comparison of M0-FT, C, and M1 is recorded in
`formal_6k_results.json`. M1 did not improve small-object AP relative to the
matched fine-tuned baseline.

Two follow-up fusion structures were screened and rejected:

- multiscale residual fusion: no material gain under the matched 1k budget;
- 75 x 75 high-resolution auxiliary head: severe false-positive and latency
  regressions.

The fusion work is retained as ablation and failure analysis. It does not
change the selected final method.

## Reproducibility

The final tiled result needs only:

- the released `ssd300_mAP_77.43_v2.pth` checkpoint;
- VOC2007 test;
- `tools/tiled_eval.py`;
- `tools/size_stratified_eval.py`;
- the frozen parameters listed in `tiled_inference_results.json`.

No training or test-set tuning is required. The exact commands are in the
top-level README and final Colab notebook.

## Known limitations

- The selected method is about 3.5 times slower than single-pass SSD300.
- Medium and large AP decrease slightly.
- Fusion experiments use one seed and do not establish statistical
  significance.
- Raw detections and checkpoints are generated artifacts and are not committed.
