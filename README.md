# COMP4330 SSD300 Fusion

Course project code for diagnosing and improving small-object detection in
VGG16-SSD300 with a lightweight top-down feature-fusion branch.

The project compares four pre-registered configurations on PASCAL VOC:

| Config | Purpose |
|---|---|
| `M0` | Released SSD300 VOC0712 checkpoint |
| `M0-FT` | Same-budget fine-tuned baseline |
| `C` | Parameter-matched branch driven only by `conv4_3` |
| `M1` | `conv7 -> conv4_3` zero-gated top-down fusion |

The source is based on the MIT-licensed
[`amdegroot/ssd.pytorch`](https://github.com/amdegroot/ssd.pytorch), pinned at
upstream commit `5b0b77faa955c1917b0c710d770739ba8fbff9b7`, with compatibility
updates required by current PyTorch.

## Quick start in Colab

1. Open `notebooks/COMP4330_sef_runner.ipynb` in Google Colab.
2. Select a GPU runtime: **Runtime > Change runtime type > T4 GPU**.
3. Keep the notebook's project and asset directories in Google Drive.
4. Run the cells in order. The notebook downloads the repository if it is not
   already present, prepares VOC0712, checks the checkpoint, and runs a short
   profiling job before a long training job.

The DataLoader should read images from `/content/data`, not directly from the
mounted Drive. Drive is used for source archives, checkpoints, logs, and small
result files.

## Local setup

On Windows, use WSL2 Ubuntu 22.04. Keep the repository and environments in the
Linux filesystem, for example `~/work/comp4330-ssd-fusion`. The Windows NVIDIA
driver provides CUDA to WSL; do not install another Linux display driver.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build recommended by pytorch.org for the local CUDA.
python -m pip install torch torchvision
python -m pip install -r requirements.txt
pytest -q
```

For the 4 GB RTX 3050, start with batch size 2 and gradient accumulation 4.
Fall back to batch size 1 and accumulation 8 if an epoch does not fit.

## Assets

```bash
bash scripts/download_assets.sh \
  /path/to/persistent/assets \
  /content/data/VOCdevkit
```

The script fetches VOC2007 trainval/test, VOC2012 trainval, the released SSD300
checkpoint, and the reduced VGG16 base. It writes `SHA256SUMS.json` on the first
complete run and rejects changed or truncated files on later runs.

Expected locations:

```text
assets/VOCtrainval_06-Nov-2007.tar
assets/VOCtest_06-Nov-2007.tar
assets/VOCtrainval_11-May-2012.tar
assets/weights/ssd300_mAP_77.43_v2.pth
assets/weights/vgg16_reducedfc.pth
VOCdevkit/VOC2007
VOCdevkit/VOC2012
```

## Training

Experiment settings live in `configs/`. All formal arms use the same iteration
budget, augmentation, optimizer schedule, and evaluation code.

```bash
python scripts/run_experiment.py --config configs/m1_fusion.yaml
```

Resume from the latest complete checkpoint:

```bash
python scripts/run_experiment.py \
  --config configs/m1_fusion.yaml \
  --resume /path/to/last.pt
```

Checkpoints contain model, optimizer, scheduler, iteration, AMP scaler, seed,
and RNG state. A resumed Colab session should lose at most the interval since
the latest checkpoint.

## Evaluation

Overall mAP follows the repository's VOC2007 11-point AP convention. The
size-stratified result is a separate SSD-input-space metric using areas after
resizing to 300 x 300:

- small: `< 32^2`
- medium: `32^2 <= area < 96^2`
- large: `>= 96^2`

The primary size result is class-aware pooled AP; per-class AP and macro mAP
are reported as secondary results. VOC2007 test must not be used to tune the
model or select an iteration.

Save all VOC2007 detections:

```bash
python eval.py \
  --trained_model /path/to/ssd300_mAP_77.43_v2.pth \
  --voc_root /content/data/VOCdevkit \
  --save_folder /path/to/run/eval \
  --variant baseline
```

Then compute overall and size-stratified metrics:

```bash
python tools/size_stratified_eval.py \
  --detections /path/to/run/eval/detections.pkl \
  --voc-root /content/data/VOCdevkit \
  --output /path/to/run/eval/metrics.json
```

Use `--limit 5` on `eval.py` for a quick end-to-end smoke test. Do not use a
limited result as a project metric.

## Repository layout

```text
configs/       experiment definitions
data/          VOC and COCO loaders from upstream
layers/        SSD losses and box utilities
notebooks/     Colab runner
scripts/       asset, training, and evaluation entry points
tests/         CPU tests and synthetic evaluator checks
```

## Compute

Use Colab Pro/T4 for the formal `M0-FT`, `C`, and `M1` runs, with checkpoints
written every 500-1000 iterations. The local 4 GB GPU is appropriate for
baseline evaluation, demo inference, and short compatibility tests. Avoid
making it the critical path for three long training runs.

## Upstream license

The upstream code is distributed under the MIT license in `LICENSE`. The
project modifications retain that license.
