#!/usr/bin/env python3
"""Command-line training tool for Face Studio.

Trains all stages sequentially (or a specific stage) without the GUI.
Usage:
    python train_cli.py              # Train all stages that have enough data
    python train_cli.py --stage 1    # Train only stage 1
    python train_cli.py --stage 1 2  # Train stages 1 and 2
    python train_cli.py --epochs 800 # Override epoch count
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

from config import (
    GRID_SIZE, STAGE_FILES, STAGE_MIN_SAMPLES, STAGE_NAMES,
    KL_WARMUP_START, KL_WARMUP_END, KL_FINAL_BETA,
    TRAINING_EPOCHS, TRAINING_LR, NOISE_FACTOR, SHARPNESS_WEIGHT,
    CONNECTIVITY_WEIGHT, CONNECTIVITY_WARMUP_START, CONNECTIVITY_WARMUP_END,
    BOUNDARY_WEIGHT, AUGMENT_SYMMETRY_STAGES,
)
from model import (
    HeadVAE, ConditionalVAE,
    staged_loss, kl_beta_schedule, add_noise, augment_batch,
)


def load_stage_data(stage):
    """Load and return (targets, bases) for a stage."""
    base_path, target_path, _ = STAGE_FILES[stage]

    targets = []
    bases = []

    if not os.path.exists(target_path):
        return targets, bases

    with open(target_path, "r") as f:
        for row in csv.reader(f):
            if len(row) == 256:
                targets.append([float(v) for v in row])

    if stage > 1 and base_path and os.path.exists(base_path):
        with open(base_path, "r") as f:
            for row in csv.reader(f):
                if len(row) == 256:
                    bases.append([float(v) for v in row])

    return targets, bases


def curate_data(targets, bases, stage):
    """Filter out bad training samples."""
    if stage == 1:
        keep = []
        for i, row in enumerate(targets):
            arr = np.array(row).reshape(GRID_SIZE, GRID_SIZE)
            filled = (arr > 0.5).sum()
            if filled < 15 or filled > 180:
                continue
            padded = np.pad(arr > 0.5, 1, mode='constant').astype(float)
            neighbor_count = sum(
                np.roll(np.roll(padded, dy, 0), dx, 1)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy, dx) != (0, 0)
            )[1:-1, 1:-1]
            filled_mask = arr > 0.5
            if filled_mask.sum() > 0:
                stray_frac = ((neighbor_count < 2) & filled_mask).sum() / filled_mask.sum()
                if stray_frac > 0.3:
                    continue
            keep.append(i)
    else:
        keep = []
        for i, row in enumerate(targets):
            if bases and i < len(bases):
                new_pixels = sum(1 for t, b in zip(row, bases[i]) if t > 0.5 and b < 0.5)
                if new_pixels < 1:
                    continue
            keep.append(i)

    if len(keep) < len(targets) * 0.7:
        return targets, bases, 0

    n_removed = len(targets) - len(keep)
    filtered_targets = [targets[i] for i in keep]
    filtered_bases = [bases[i] for i in keep] if bases else []
    return filtered_targets, filtered_bases, n_removed


def train_stage(stage, epochs=None):
    """Train a single stage's model."""
    if epochs is None:
        epochs = TRAINING_EPOCHS

    name = STAGE_NAMES.get(stage, f"Stage {stage}")
    _, _, model_path = STAGE_FILES[stage]

    # Load data
    targets, bases = load_stage_data(stage)
    count = len(targets)
    minimum = STAGE_MIN_SAMPLES.get(stage, 30)

    if count < minimum:
        print(f"  SKIP: Only {count}/{minimum} samples. Need more data.")
        return False

    # Curate
    targets, bases, n_removed = curate_data(targets, bases, stage)
    if n_removed > 0:
        print(f"  Curated: removed {n_removed} bad sample(s), {len(targets)} remaining")

    # Augment
    aug_targets, aug_bases = augment_batch(
        targets, bases, stage,
        symmetry_stages=AUGMENT_SYMMETRY_STAGES,
    )
    print(f"  Data: {len(targets)} raw -> {len(aug_targets)} augmented samples")

    target_tensor = torch.tensor(aug_targets, dtype=torch.float32)
    base_tensor = torch.tensor(aug_bases, dtype=torch.float32)

    # Create model
    if stage == 1:
        model = HeadVAE()
    else:
        model = ConditionalVAE(stage_name=f"stage{stage}")

    # Backup existing model
    if os.path.exists(model_path):
        import shutil
        shutil.copy2(model_path, model_path + ".bak")
        print(f"  Backed up existing model to {model_path}.bak")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model.__class__.__name__} ({param_count:,} params)")

    optimizer = optim.Adam(model.parameters(), lr=TRAINING_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=TRAINING_LR * 0.05,
    )

    n_samples = target_tensor.shape[0]
    batch_size = min(32, n_samples)

    start_time = time.time()
    last_print = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            batch_target = target_tensor[idx]
            batch_base = base_tensor[idx]

            optimizer.zero_grad()

            noisy_target = add_noise(batch_target, noise_factor=NOISE_FACTOR)

            if stage == 1:
                recon, mu, logvar = model(noisy_target)
            else:
                recon, mu, logvar = model(noisy_target, batch_base)

            beta = kl_beta_schedule(
                epoch, KL_WARMUP_START, KL_WARMUP_END, KL_FINAL_BETA,
            )
            npw = 1.0 if stage == 1 else 2.0

            if stage == 1 and epoch > CONNECTIVITY_WARMUP_START:
                conn_progress = min(1.0, (epoch - CONNECTIVITY_WARMUP_START)
                                    / (CONNECTIVITY_WARMUP_END - CONNECTIVITY_WARMUP_START))
                conn_w = CONNECTIVITY_WEIGHT * conn_progress
            else:
                conn_w = 0.0

            bnd_w = BOUNDARY_WEIGHT if stage > 1 else 0.0

            loss = staged_loss(
                recon, batch_target, batch_base, mu, logvar, beta,
                new_pixel_weight=npw,
                sharpness_weight=SHARPNESS_WEIGHT,
                connectivity_weight=conn_w,
                boundary_weight=bnd_w,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Update progress bar
        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - start_time
        eta = (elapsed / epoch) * (epochs - epoch) if epoch > 0 else 0
        bar_len = 30
        filled = int(bar_len * epoch / epochs)
        bar = "=" * filled + "-" * (bar_len - filled)
        lr = scheduler.get_last_lr()[0]
        pct = int(100 * epoch / epochs)
        line = (f"\r  [{bar}] {pct:3d}%  epoch {epoch:4d}/{epochs}  "
                f"loss={avg_loss:.4f}  beta={beta:.3f}  "
                f"lr={lr:.1e}  {elapsed:.0f}s elapsed, ~{eta:.0f}s left")
        sys.stdout.write(line)
        sys.stdout.flush()

    # Save
    print()  # newline after \r progress bar
    torch.save(model.state_dict(), model_path)
    total_time = time.time() - start_time
    print(f"  Saved to {model_path} ({total_time:.1f}s total)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train Face Studio models from the command line",
    )
    parser.add_argument(
        "--stage", type=int, nargs="+", default=None,
        help="Stage(s) to train (1-4). Default: all stages with enough data.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Number of training epochs (default: {TRAINING_EPOCHS})",
    )
    args = parser.parse_args()

    stages = args.stage if args.stage else [1, 2, 3, 4]

    print("=" * 60)
    print("Face Studio — CLI Trainer")
    print("=" * 60)

    # Show data status for all stages
    print("\nData status:")
    for s in range(1, 5):
        targets, _ = load_stage_data(s)
        minimum = STAGE_MIN_SAMPLES.get(s, 30)
        name = STAGE_NAMES.get(s, f"Stage {s}")
        status = "READY" if len(targets) >= minimum else f"need {minimum - len(targets)} more"
        _, _, model_path = STAGE_FILES[s]
        has_model = " [trained]" if os.path.exists(model_path) else ""
        print(f"  Stage {s} ({name}): {len(targets)}/{minimum} samples — {status}{has_model}")

    print()

    trained = 0
    skipped = 0

    for stage in stages:
        if stage < 1 or stage > 4:
            print(f"Invalid stage {stage}, skipping")
            continue

        name = STAGE_NAMES.get(stage, f"Stage {stage}")
        print(f"{'─' * 60}")
        print(f"Stage {stage}: {name}")
        print(f"{'─' * 60}")

        # Check prerequisites for stages 2+
        if stage > 1:
            _, _, prev_model = STAGE_FILES[stage - 1]
            if not os.path.exists(prev_model):
                print(f"  SKIP: Stage {stage - 1} model not trained yet.")
                skipped += 1
                continue

        if train_stage(stage, epochs=args.epochs):
            trained += 1
        else:
            skipped += 1

        print()

    print("=" * 60)
    print(f"Done! Trained {trained} stage(s), skipped {skipped}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
