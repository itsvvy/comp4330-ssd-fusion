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

These files are small summaries. Raw `detections.pkl`, metrics logs, and
checkpoints remain outside Git and should be stored in Drive or another
artifact store.
