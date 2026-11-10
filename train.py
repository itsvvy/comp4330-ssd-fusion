#!/usr/bin/env python3
"""Train one pre-registered SSD300 experiment arm."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data import MEANS, VOCDetection, detection_collate, voc
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from utils.augmentations import SSDAugmentation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=10_000)
    parser.add_argument("--freeze_iter", type=int, default=4_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze_lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--lr_steps", default="6000,8500,9500"
    )
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument(
        "--fusion_mode",
        choices=["baseline", "control", "fusion", "small_head_fusion"],
        default="baseline",
    )
    parser.add_argument("--control_channels", type=int, default=768)
    parser.add_argument("--profile_iters", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    args.lr_steps = tuple(
        int(value) for value in args.lr_steps.split(",") if value.strip()
    )
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_order(length: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed * 100_003 + epoch)
    return torch.randperm(length, generator=generator).tolist()


def make_loader(
    dataset,
    order: list[int],
    start: int,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    subset = Subset(dataset, order[start:])
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def set_backbone_trainable(model, enabled: bool) -> None:
    for parameter in model.vgg.parameters():
        parameter.requires_grad_(enabled)


def set_learning_rates(
    optimizer,
    iteration: int,
    max_iter: int,
    head_lr: float,
    backbone_lr: float,
    lr_steps: tuple[int, ...],
    frozen: bool,
) -> None:
    step_index = sum(iteration >= step for step in lr_steps)
    factor = 0.1**step_index
    for group in optimizer.param_groups:
        base = group["base_lr"]
        base = 0.0 if group["name"] == "backbone" and frozen else base
        group["lr"] = base * factor


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    iteration: int,
    epoch: int,
    epoch_position: int,
    args,
    losses: dict[str, float],
) -> None:
    state = {
        "format_version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": iteration,
        "epoch": epoch,
        "epoch_position": epoch_position,
        "seed": args.seed,
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "losses": losses,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_checkpoint(path: Path, model, optimizer, scaler):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return (
        int(checkpoint["iteration"]),
        int(checkpoint.get("epoch", 0)),
        int(checkpoint.get("epoch_position", 0)),
    )


def train(args) -> None:
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    metrics_path = save_folder / "metrics.jsonl"

    dataset = VOCDetection(
        root=args.dataset_root,
        transform=SSDAugmentation(voc["min_dim"], MEANS),
        filter_empty=True,
    )
    print(f"training images: {len(dataset)}")

    model = build_ssd(
        "train",
        voc["min_dim"],
        voc["num_classes"],
        variant=args.fusion_mode,
        control_channels=args.control_channels,
    ).to(device)
    model_info = {
        "variant": args.fusion_mode,
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "control_channels": (
            args.control_channels
            if args.fusion_mode == "control"
            else None
        ),
    }
    (save_folder / "model_info.json").write_text(
        json.dumps(model_info, indent=2) + "\n", encoding="utf-8"
    )

    frozen = args.freeze_iter > 0
    set_backbone_trainable(model, not frozen)

    backbone_parameters = [
        parameter
        for parameter in model.vgg.parameters()
        if parameter.requires_grad or args.freeze_iter > 0
    ]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids and parameter.requires_grad
    ]
    optimizer = optim.SGD(
        [
            {
                "params": backbone_parameters,
                "lr": args.unfreeze_lr,
                "base_lr": args.unfreeze_lr,
                "name": "backbone",
            },
            {
                "params": other_parameters,
                "lr": args.lr,
                "base_lr": args.lr,
                "name": "head_branch",
            },
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = MultiBoxLoss(
        voc["num_classes"],
        0.5,
        True,
        0,
        True,
        3,
        0.5,
        False,
        use_gpu=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and device.type == "cuda"
    )

    iteration = 0
    epoch = 0
    epoch_position = 0
    if args.resume:
        iteration, epoch, epoch_position = restore_checkpoint(
            Path(args.resume), model, optimizer, scaler
        )
        print(
            "resumed:",
            args.resume,
            "iteration",
            iteration,
            "epoch",
            epoch,
            "position",
            epoch_position,
        )
    elif args.init_checkpoint:
        incompatible = model.load_weights(args.init_checkpoint)
        print(
            "initialized:",
            args.init_checkpoint,
            "missing:",
            list(incompatible.missing_keys),
            "unexpected:",
            list(incompatible.unexpected_keys),
        )

    if iteration >= args.freeze_iter:
        frozen = False
    set_backbone_trainable(model, not frozen)

    stop_iteration = args.max_iter
    if args.profile_iters:
        stop_iteration = min(
            args.max_iter, iteration + args.profile_iters
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loc = 0.0
    running_conf = 0.0
    running_steps = 0
    started = time.perf_counter()

    order = epoch_order(len(dataset), args.seed, epoch)
    loader = make_loader(
        dataset,
        order,
        epoch_position,
        args.batch_size,
        args.num_workers,
        args.seed + epoch,
    )
    batch_iterator = iter(loader)

    while iteration < stop_iteration:
        if frozen and iteration >= args.freeze_iter:
            frozen = False
            set_backbone_trainable(model, True)
            print(f"iteration {iteration}: unfreezing VGG16")
        set_learning_rates(
            optimizer,
            iteration,
            args.max_iter,
            args.lr,
            args.unfreeze_lr,
            args.lr_steps,
            frozen,
        )

        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accum_loc = 0.0
        accum_conf = 0.0

        for _ in range(args.grad_accum_steps):
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                epoch += 1
                epoch_position = 0
                order = epoch_order(len(dataset), args.seed, epoch)
                loader = make_loader(
                    dataset,
                    order,
                    0,
                    args.batch_size,
                    args.num_workers,
                    args.seed + epoch,
                )
                batch_iterator = iter(loader)
                images, targets = next(batch_iterator)

            epoch_position += images.size(0)
            images = images.to(
                device, non_blocking=device.type == "cuda"
            )
            targets = [
                target.to(device, non_blocking=device.type == "cuda")
                for target in targets
            ]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                predictions = model(images)
                loss_loc, loss_conf = criterion(predictions, targets)
                loss = (loss_loc + loss_conf) / args.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loc += float(loss_loc.detach())
            accum_conf += float(loss_conf.detach())

        scaler.step(optimizer)
        scaler.update()
        iteration += 1
        running_loc += accum_loc
        running_conf += accum_conf
        running_steps += 1

        if iteration % 10 == 0 or iteration == 1:
            elapsed = time.perf_counter() - started
            speed = running_steps / max(elapsed, 1e-9)
            record = {
                "iteration": iteration,
                "loss_loc": running_loc / running_steps,
                "loss_conf": running_conf / running_steps,
                "iterations_per_second": speed,
                "head_lr": optimizer.param_groups[1]["lr"],
                "backbone_lr": optimizer.param_groups[0]["lr"],
                "frozen": frozen,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                f"iter {iteration} "
                f"loc {record['loss_loc']:.4f} "
                f"conf {record['loss_conf']:.4f} "
                f"iter/s {speed:.2f} "
                f"step_s {time.perf_counter() - step_started:.2f}",
                flush=True,
            )
            running_loc = 0.0
            running_conf = 0.0
            running_steps = 0
            started = time.perf_counter()

        if iteration % args.checkpoint_every == 0:
            save_checkpoint(
                save_folder / "last.pt",
                model,
                optimizer,
                scaler,
                iteration,
                epoch,
                epoch_position,
                args,
                {
                    "loss_loc": accum_loc,
                    "loss_conf": accum_conf,
                },
            )

    save_checkpoint(
        save_folder / "last.pt",
        model,
        optimizer,
        scaler,
        iteration,
        epoch,
        epoch_position,
        args,
        {
            "loss_loc": accum_loc,
            "loss_conf": accum_conf,
        },
    )
    if args.profile_iters:
        (save_folder / "profile.json").write_text(
            json.dumps(
                {
                    "gpu": (
                        torch.cuda.get_device_name(0)
                        if device.type == "cuda"
                        else "cpu"
                    ),
                    "peak_memory_gb": (
                        torch.cuda.max_memory_allocated() / 1024**3
                        if device.type == "cuda"
                        else 0.0
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print("training complete:", save_folder / "last.pt")


if __name__ == "__main__":
    train(parse_args())
