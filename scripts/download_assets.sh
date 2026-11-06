#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_assets.sh [asset_root] [data_root]
#
# Keep asset_root on a persistent volume in Colab, for example:
#   /content/drive/MyDrive/COMP4330/assets
# Keep data_root on the fast local runtime disk, for example:
#   /content/data/VOCdevkit

ASSET_ROOT="${1:-${PWD}/data_cache/assets}"
DATASET_ROOT="${2:-${PWD}/data_cache/VOCdevkit}"
DATA_PARENT="$(dirname "${DATASET_ROOT}")"
WEIGHT_ROOT="${ASSET_ROOT}/weights"

mkdir -p "${ASSET_ROOT}" "${WEIGHT_ROOT}" "${DATA_PARENT}"

download() {
  local url="$1"
  local target="$2"
  if [[ -s "${target}" ]]; then
    echo "exists: ${target}"
    return
  fi

  echo "downloading: ${url}"
  curl --fail --location --retry 3 --retry-delay 2 \
    --continue-at - "${url}" --output "${target}.part"
  mv "${target}.part" "${target}"
}

download \
  "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" \
  "${ASSET_ROOT}/VOCtrainval_06-Nov-2007.tar"
download \
  "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar" \
  "${ASSET_ROOT}/VOCtest_06-Nov-2007.tar"
download \
  "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" \
  "${ASSET_ROOT}/VOCtrainval_11-May-2012.tar"

download \
  "https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth" \
  "${WEIGHT_ROOT}/ssd300_mAP_77.43_v2.pth"
download \
  "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth" \
  "${WEIGHT_ROOT}/vgg16_reducedfc.pth"

extract_if_missing() {
  local archive="$1"
  local marker="$2"
  if [[ -f "${marker}" ]]; then
    echo "already extracted: ${marker}"
    return
  fi
  echo "extracting ${archive} -> ${DATA_PARENT}"
  tar -xf "${archive}" -C "${DATA_PARENT}"
}

extract_if_missing \
  "${ASSET_ROOT}/VOCtrainval_06-Nov-2007.tar" \
  "${DATASET_ROOT}/VOC2007/ImageSets/Main/trainval.txt"
extract_if_missing \
  "${ASSET_ROOT}/VOCtest_06-Nov-2007.tar" \
  "${DATASET_ROOT}/VOC2007/ImageSets/Main/test.txt"
extract_if_missing \
  "${ASSET_ROOT}/VOCtrainval_11-May-2012.tar" \
  "${DATASET_ROOT}/VOC2012/ImageSets/Main/trainval.txt"

python scripts/verify_assets.py \
  --asset-root "${ASSET_ROOT}" \
  --data-root "${DATASET_ROOT}" \
  --write-manifest

echo "assets ready"
echo "dataset root: ${DATASET_ROOT}"
echo "weights root: ${WEIGHT_ROOT}"

