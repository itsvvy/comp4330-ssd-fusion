#!/usr/bin/env python3
"""Resolve a YAML experiment and launch the common SSD training entry point."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    config = load_config((path.parent / parent).resolve())
    config.update(raw)
    return config


def add_override(config: dict, key: str, value: object) -> None:
    if value is not None:
        config[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--save-root")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume")
    parser.add_argument("--profile-iters", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    add_override(config, "dataset_root", args.dataset_root)
    add_override(config, "init_checkpoint", args.init_checkpoint)
    add_override(config, "save_root", args.save_root)
    add_override(config, "batch_size", args.batch_size)
    add_override(config, "max_iter", args.max_iter)
    add_override(config, "seed", args.seed)

    experiment = str(config.get("name", args.config.stem))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else Path(config["save_root"]) / experiment / stamp
    )

    resolved = dict(config)
    resolved["source_config"] = str(args.config.resolve())
    resolved["run_dir"] = str(run_dir)
    resolved["resume"] = args.resume

    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--dataset_root",
        str(config["dataset_root"]),
        "--init_checkpoint",
        str(config["init_checkpoint"]),
        "--save_folder",
        str(run_dir),
        "--batch_size",
        str(config["batch_size"]),
        "--grad_accum_steps",
        str(config.get("grad_accum_steps", 1)),
        "--num_workers",
        str(config.get("num_workers", 2)),
        "--max_iter",
        str(config["max_iter"]),
        "--freeze_iter",
        str(config.get("freeze_iter", 0)),
        "--lr",
        str(config["learning_rate"]),
        "--unfreeze_lr",
        str(config.get("unfreeze_learning_rate", config["learning_rate"])),
        "--momentum",
        str(config.get("momentum", 0.9)),
        "--weight_decay",
        str(config.get("weight_decay", 5e-4)),
        "--lr_steps",
        ",".join(str(value) for value in config.get("lr_steps", [])),
        "--checkpoint_every",
        str(config.get("checkpoint_every", 500)),
        "--fusion_mode",
        str(config.get("model_variant", "baseline")),
        "--seed",
        str(config.get("seed", 1337)),
    ]

    if config.get("model_variant") == "control":
        command.extend(["--control_channels", str(config["control_channels"])])
    if args.resume:
        command.extend(["--resume", str(Path(args.resume).resolve())])
    if args.profile_iters:
        command.extend(["--profile_iters", str(args.profile_iters)])
    if config.get("amp", True):
        command.append("--amp")

    print("run directory:", run_dir)
    print("command:", shlex.join(command))
    if args.dry_run:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
