#!/usr/bin/env python3
"""Behavior Cloning training for NS-2026-07 dexterous hand policy.

Usage:
    python training/train.py --demos ../09d3c1e8-ceeb-438c-a155-2fd865c65e2e/demonstrations/weak_train_rollouts.jsonl --output ../submission/model/best_model.pt --min-score01 0.1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import DexConfig
from model.policy import DexPolicy
from training.data_loader import DexDemoDataset, collate_fn
from training.losses import MultiHeadLoss


def compute_class_weights(dataset: DexDemoDataset, num_classes: int, field: str) -> torch.Tensor:
    counts = [0] * num_classes
    for i in range(len(dataset)):
        idx = dataset.samples[i][field]
        if 0 <= idx < num_classes:
            counts[idx] += 1
    total = sum(counts)
    weights = []
    for c in counts:
        if c > 0:
            weights.append(total / (num_classes * c))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: DexPolicy,
    dataloader: DataLoader,
    loss_fn: MultiHeadLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    total_metrics: dict[str, float] = {}
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        targets = {
            "primitive": batch.pop("primitive_target"),
            "finger": batch.pop("finger_target"),
            "force": batch.pop("force_target"),
            "direction": batch.pop("direction_target"),
            "task_type": batch.pop("task_type"),
            "sample_weight": batch.pop("sample_weight"),
        }

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                predictions = model(batch)
                loss, metrics = loss_fn(predictions, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(batch)
            loss, metrics = loss_fn(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}


@torch.no_grad()
def validate(
    model: DexPolicy,
    dataloader: DataLoader,
    loss_fn: MultiHeadLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_metrics: dict[str, float] = {}
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        targets = {
            "primitive": batch.pop("primitive_target"),
            "finger": batch.pop("finger_target"),
            "force": batch.pop("force_target"),
            "direction": batch.pop("direction_target"),
            "task_type": batch.pop("task_type"),
            "sample_weight": batch.pop("sample_weight"),
        }

        predictions = model(batch)
        _, metrics = loss_fn(predictions, targets)

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

    return {k: v / n_batches for k, v in total_metrics.items()}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train DexPolicy via BC")
    parser.add_argument("--demos", required=True, help="Path to weak_train_rollouts.jsonl")
    parser.add_argument("--output", default="best_model.pt", help="Output checkpoint path")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-score01", type=float, default=0.1, help="Minimum episode score to include")
    parser.add_argument("--aux-weight", type=float, default=0.1, help="Weight for auxiliary task type loss")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    print("Loading demonstrations...")
    train_dataset = DexDemoDataset(
        args.demos, split="train", augment=True, min_score01=args.min_score01,
    )
    val_dataset = DexDemoDataset(
        args.demos, split="val", augment=False, min_score01=args.min_score01,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    primitive_weights = compute_class_weights(train_dataset, 10, "primitive")
    finger_weights = compute_class_weights(train_dataset, 7, "finger")
    print(f"Primitive class weights: {[f'{w:.4f}' for w in primitive_weights.tolist()]}")
    print(f"Finger class weights: {[f'{w:.4f}' for w in finger_weights.tolist()]}")

    cfg = DexConfig()
    cfg.batch_size = args.batch_size
    model = DexPolicy(cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    loss_fn = MultiHeadLoss(
        class_weights_primitive=primitive_weights.to(device),
        class_weights_finger=finger_weights.to(device),
        aux_weight=args.aux_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and not args.no_amp) else None

    best_val_loss = float("inf")
    best_epoch = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, args.grad_clip
        )
        scheduler.step()
        train_time = time.perf_counter() - t0

        log = (
            f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_metrics['total_loss']:.4f}"
            f" | prim_acc={train_metrics['primitive_acc']:.3f}"
            f" | finger_acc={train_metrics['finger_acc']:.3f}"
            f" | force_mae={train_metrics['force_mae']:.4f}"
            f" | dir_cos={train_metrics['direction_cos_sim']:.3f}"
            f" | aux_acc={train_metrics['aux_acc']:.3f}"
            f" | time={train_time:.1f}s"
        )

        if epoch % 2 == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            log += (
                f" || val_loss={val_metrics['total_loss']:.4f}"
                f" | val_prim_acc={val_metrics['primitive_acc']:.3f}"
                f" | val_finger_acc={val_metrics['finger_acc']:.3f}"
                f" | val_aux_acc={val_metrics['aux_acc']:.3f}"
            )

            if val_metrics["total_loss"] < best_val_loss:
                best_val_loss = val_metrics["total_loss"]
                best_epoch = epoch
                torch.save(model.state_dict(), output_path)
                log += " *"

        print(log)

    print(f"\nBest val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Saved checkpoint to: {output_path}")


if __name__ == "__main__":
    main()
