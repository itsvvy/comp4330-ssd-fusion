# Experiment Records

`baseline_m0.json` records the released SSD300 VOC0712 checkpoint on VOC2007
test. The standard evaluator reproduces 77.66% mAP at IoU 0.5, close to the
repository's 77.43% reference. The size-stratified result uses the same saved
detections after mapping annotations and detections into 300 x 300 input
space.

`experiment_history.json` and `experiment_history.md` consolidate the baseline
diagnosis, 6k three-arm comparison, gate diagnostics, repair attempts, rejected
alternative structures, and the final tiled result.

`t4_profile_m1.json` records a 30-iteration compatibility/throughput profile
followed by a five-iteration checkpoint resume. It is not an accuracy result.
The profile establishes that batch size 8, gradient accumulation 2, and FP16
AMP fit comfortably on the available T4.

`pilot_m1_300.json` records the longer 300-iteration stability pilot. It also
completed without OOM or non-finite loss, but it is not used to select a model.

`formal_6k_results.json` and `formal_6k_results.md` contain the final formal
M0-FT, parameter-matched control C, and fusion M1 results. All three arms use
the same 6,000-iteration protocol and one seed. The final test differences are
very small; the Markdown report states the limitations explicitly.

`fusion_gate_validation.md` records the diagnostic checks that explain the
inactive and slow-starting behavior of the original fusion gate.

`tiled_inference_results.json` and `tiled_inference_results.md` record the
accuracy-first selective tiled inference experiment. It keeps the released
model weights unchanged and improves full VOC2007 test mAP and small-object AP
while reducing throughput.

`final_project_summary.md` is the concise final-state summary and points to the
frozen selected result.

`figures/experiment_overview.png` and `figures/experiment_overview.svg` are
generated from `experiment_history.json` by `scripts/plot_results.py`.

The detailed alternative-fusion records remain on their experiment branches:
`m1_gatefix_results.*` and `m1_active_results.*` on
`experiment/m1-gatefix`, `multiscale_fusion_screen.*` on
`experiment/multiscale-fusion`, and `highres_small_head_screen.*` on
`experiment/highres-small-head`.

These files are small summaries. Raw `detections.pkl`, metrics logs, and
checkpoints remain outside Git and should be stored in Drive or another
artifact store.
