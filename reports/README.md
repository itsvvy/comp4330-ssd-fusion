# Experiment Records

`baseline_m0.json` records the released SSD300 VOC0712 checkpoint on VOC2007
test. The standard evaluator reproduces 77.66% mAP@0.5, close to the
repository's 77.43% reference. The size-stratified result uses the same saved
detections after mapping annotations and detections into 300 x 300 input
space.

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

`tiled_inference_results.json` and `tiled_inference_results.md` record the
accuracy-first selective tiled inference experiment. It keeps the released
model weights unchanged and improves full VOC2007 test mAP and small-object AP
while reducing throughput.

`highres_small_head_screen.json` and `highres_small_head_screen.md` record the
rejected 75 x 75 auxiliary-head screening experiment. The larger prior set
produced severe false-positive and latency regressions.

These files are small summaries. Raw `detections.pkl`, metrics logs, and
checkpoints remain outside Git and should be stored in Drive or another
artifact store.
