#!/usr/bin/env python3
"""Run the three pre-registered 6k SSD comparison arms and VOC evaluation."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "m0_ft": ("configs/m0_ft.yaml", "baseline"),
    "control": ("configs/control.yaml", "control"),
    "m1_fusion": ("configs/m1_fusion.yaml", "fusion"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/content/data/VOCdevkit")
    parser.add_argument(
        "--init-checkpoint",
        default="/content/assets/weights/ssd300_mAP_77.43_v2.pth",
    )
    parser.add_argument("--save-root", default="/content/results/formal")
    parser.add_argument("--eval-root", default="/content/results/formal_eval")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=sorted(ARMS),
        default=["m0_ft", "control", "m1_fusion"],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool, log_path: Path) -> None:
    rendered = shlex.join(command)
    print(f"\n$ {rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)


def main():
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_root = Path(args.save_root)
    eval_root = Path(args.eval_root)

    for arm in args.arms:
        config_path, variant = ARMS[arm]
        run_dir = save_root / arm / stamp
        eval_dir = eval_root / arm / stamp
        run_log = eval_root / arm / f"{stamp}.train.log"
        eval_log = eval_root / arm / f"{stamp}.eval.log"

        train_command = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            "--config",
            str(ROOT / config_path),
            "--dataset-root",
            args.data_root,
            "--init-checkpoint",
            args.init_checkpoint,
            "--save-root",
            str(save_root),
            "--run-dir",
            str(run_dir),
        ]
        run(train_command, args.dry_run, run_log)

        checkpoint = run_dir / "last.pt"
        eval_command = [
            sys.executable,
            str(ROOT / "eval.py"),
            "--trained_model",
            str(checkpoint),
            "--voc_root",
            args.data_root,
            "--save_folder",
            str(eval_dir),
            "--variant",
            variant,
        ]
        run(eval_command, args.dry_run, eval_log)

        size_command = [
            sys.executable,
            str(ROOT / "tools" / "size_stratified_eval.py"),
            "--detections",
            str(eval_dir / "detections.pkl"),
            "--voc-root",
            args.data_root,
            "--output",
            str(eval_dir / "size_metrics.json"),
        ]
        run(size_command, args.dry_run, eval_log)


if __name__ == "__main__":
    main()
