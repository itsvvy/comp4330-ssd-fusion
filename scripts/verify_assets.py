#!/usr/bin/env python3
"""Verify the VOC archives and checkpoints used by the project."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_BYTES = {
    "VOCtrainval_06-Nov-2007.tar": 460_032_000,
    "VOCtest_06-Nov-2007.tar": 451_020_800,
    "VOCtrainval_11-May-2012.tar": 1_999_639_040,
    "ssd300_mAP_77.43_v2.pth": 105_151_288,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_paths(asset_root: Path) -> dict[str, Path]:
    return {
        "VOCtrainval_06-Nov-2007.tar": asset_root / "VOCtrainval_06-Nov-2007.tar",
        "VOCtest_06-Nov-2007.tar": asset_root / "VOCtest_06-Nov-2007.tar",
        "VOCtrainval_11-May-2012.tar": asset_root / "VOCtrainval_11-May-2012.tar",
        "ssd300_mAP_77.43_v2.pth": asset_root / "weights" / "ssd300_mAP_77.43_v2.pth",
        "vgg16_reducedfc.pth": asset_root / "weights" / "vgg16_reducedfc.pth",
    }


def verify(asset_root: Path, data_root: Path, write_manifest: bool) -> None:
    paths = asset_paths(asset_root)
    manifest_path = asset_root / "SHA256SUMS.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    current: dict[str, dict[str, int | str]] = {}
    errors: list[str] = []

    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        size = path.stat().st_size
        expected_size = EXPECTED_BYTES.get(name)
        if expected_size is not None and size != expected_size:
            errors.append(
                f"wrong size: {path} expected {expected_size}, got {size}"
            )
        digest = sha256(path)
        if name in previous and previous[name]["sha256"] != digest:
            errors.append(f"checksum changed: {path}")
        current[name] = {"bytes": size, "sha256": digest}

    required_dirs = [
        data_root / "VOC2007" / "Annotations",
        data_root / "VOC2007" / "JPEGImages",
        data_root / "VOC2007" / "ImageSets" / "Main",
        data_root / "VOC2012" / "Annotations",
        data_root / "VOC2012" / "JPEGImages",
        data_root / "VOC2012" / "ImageSets" / "Main",
    ]
    for path in required_dirs:
        if not path.is_dir():
            errors.append(f"missing dataset directory: {path}")

    test_ids = data_root / "VOC2007" / "ImageSets" / "Main" / "test.txt"
    if test_ids.is_file() and len(test_ids.read_text().splitlines()) != 4952:
        errors.append(f"VOC2007 test image count is not 4952: {test_ids}")

    if errors:
        raise SystemExit("\n".join(errors))

    if write_manifest:
        manifest_path.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {manifest_path}")

    print(f"verified {len(paths)} assets and {len(required_dirs)} dataset folders")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    verify(args.asset_root, args.data_root, args.write_manifest)


if __name__ == "__main__":
    main()
