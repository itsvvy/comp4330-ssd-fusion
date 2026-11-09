# Course Project Experiment Protocol

All formal arms start from `ssd300_mAP_77.43_v2.pth` and use the same data,
augmentation, batch size, gradient accumulation, iteration budget, seed, and
learning-rate schedule.

The original 10k-iteration design was reduced to 6k iterations to fit the
course-project compute budget:

- 2,400 frozen-backbone warm-up iterations
- 3,600 full-network fine-tuning iterations
- learning-rate steps at 3,600, 5,100, and 5,700 iterations
- batch size 8 with gradient accumulation 2
- FP16 mixed precision
- one seed

The 300-iteration M1 pilot and 35-iteration resume check completed without OOM
or non-finite loss. T4 throughput indicates that one 6k arm should take roughly
25-35 minutes after startup.

The formal comparison remains `M0-FT`, `C`, and `M1`. VOC2007 test is evaluated
only after each configuration and checkpoint are fixed.
