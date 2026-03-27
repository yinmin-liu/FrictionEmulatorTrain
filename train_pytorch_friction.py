#!/usr/bin/env python3
"""
Train a Weertman friction emulator in PyTorch.

Data format:
  folder/weertman_rank_0.csv
  folder/weertman_rank_1.csv
  ...
Each CSV row has 3 columns:
  C2, vmag, alpha2

This script does three things:
1. trains an MLP in PyTorch
2. saves a native PyTorch checkpoint (.pt)
3. saves a plain-text model file compatible with the existing C++ FrictionEmulator::Load()

Example:
  python train_pytorch_friction.py \
      --n-ranks 5 \
      --epochs 5000 \
      --lr 1e-3 \
      --batch-size 4096 \
      --n-seeds 5 \
      --folder ./data \
      --model-file ./friction_emulator.txt \
      --checkpoint ./friction_emulator.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


FIXED_X_MEAN = np.array([9.05e6, 2.08e-5], dtype=np.float64)
FIXED_X_STD = np.array([6.61e6, 4.67e-5], dtype=np.float64)
FIXED_Y_MEAN = np.array([2.09e11], dtype=np.float64)
FIXED_Y_STD = np.array([1.18e12], dtype=np.float64)


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class FrictionSample:
    x: List[float]
    y: List[float]


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fmt_sci(value: float) -> str:
    return f"{float(value):.6e}"


def fmt_seq(values: List[float]) -> str:
    return "[" + ", ".join(fmt_sci(v) for v in values) + "]"


def fmt_percent(value: float) -> str:
    return f"{float(value):.6f}"


# -----------------------------------------------------------------------------
# CSV loading
# -----------------------------------------------------------------------------
def _is_header_line(row: List[str]) -> bool:
    """Match the C++ logic loosely: skip rows with letters other than e/E."""
    joined = ",".join(row)
    for ch in joined:
        if ch.isalpha() and ch not in ("e", "E"):
            return True
    return False



def parse_csv_row(row: List[str], in_dim: int, out_dim: int) -> FrictionSample | None:
    if not row:
        return None
    if _is_header_line(row):
        return None

    items = []
    for item in row:
        item = item.strip()
        if item == "":
            continue
        try:
            items.append(float(item))
        except ValueError:
            return None

    if len(items) != in_dim + out_dim:
        return None

    x = items[:in_dim]
    y = items[in_dim:]

    if not all(math.isfinite(v) for v in x + y):
        return None

    return FrictionSample(x=x, y=y)



def load_data(folder: str, n_ranks: int, in_dim: int, out_dim: int) -> List[FrictionSample]:
    all_data: List[FrictionSample] = []

    for rank in range(n_ranks):
        path = Path(folder) / f"weertman_rank_{rank}.csv"
        if not path.exists():
            print(f"Warning: could not open {path}, skipping.")
            continue

        count = 0
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                sample = parse_csv_row(row, in_dim, out_dim)
                if sample is not None:
                    all_data.append(sample)
                    count += 1

        print(f"Loaded {count} rows from {path}")

    print(f"Total loaded: {len(all_data)} samples")
    return all_data


# -----------------------------------------------------------------------------
# Split / stats
# -----------------------------------------------------------------------------
def split_data(
    all_data: List[FrictionSample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    split_seed: int = 42,
) -> Tuple[List[FrictionSample], List[FrictionSample], List[FrictionSample]]:
    idx = list(range(len(all_data)))
    rng = random.Random(split_seed)
    rng.shuffle(idx)

    n_total = len(all_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_data = [all_data[i] for i in idx[:n_train]]
    val_data = [all_data[i] for i in idx[n_train:n_train + n_val]]
    test_data = [all_data[i] for i in idx[n_train + n_val:]]

    print(
        f"Split -> Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}"
    )
    return train_data, val_data, test_data



def compute_stats(data: List[FrictionSample], in_dim: int, out_dim: int):
    x = np.array([s.x for s in data], dtype=np.float64)
    y = np.array([s.y for s in data], dtype=np.float64)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)

    x_std[x_std < 1e-12] = 1.0
    y_std[y_std < 1e-12] = 1.0

    assert x_mean.shape == (in_dim,)
    assert x_std.shape == (in_dim,)
    assert y_mean.shape == (out_dim,)
    assert y_std.shape == (out_dim,)

    return x_mean, x_std, y_mean, y_std



def to_numpy(data: List[FrictionSample]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([s.x for s in data], dtype=np.float32)
    y = np.array([s.y for s in data], dtype=np.float32)
    return x, y


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class FrictionMLP(nn.Module):
    def __init__(self, in_dim: int = 2, h1: int = 64, h2: int = 64, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# Eval helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_rmse(
    model: nn.Module,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total_sse = 0.0
    total_count = 0

    x_mean_t = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.as_tensor(x_std, dtype=torch.float32, device=device)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    for start in range(0, len(x_raw), batch_size):
        end = min(start + batch_size, len(x_raw))
        xb_raw = torch.as_tensor(x_raw[start:end], dtype=torch.float32, device=device)
        yb_raw = torch.as_tensor(y_raw[start:end], dtype=torch.float32, device=device)

        xb = (xb_raw - x_mean_t) / x_std_t
        pred_norm = model(xb)
        pred_raw = pred_norm * y_std_t + y_mean_t

        err = pred_raw - yb_raw
        total_sse += torch.sum(err * err).item()
        total_count += err.numel()

    return math.sqrt(total_sse / max(total_count, 1))


@torch.no_grad()
def print_samples(
    model: nn.Module,
    data: List[FrictionSample],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    n_samples: int = 5,
) -> None:
    model.eval()

    print("\nSample predictions:")
    for i, samp in enumerate(data[:n_samples]):
        x_input = np.asarray([samp.x], dtype=np.float32)
        x_raw = torch.as_tensor(x_input, dtype=torch.float32, device=device)
        x_norm = (x_raw - torch.tensor(x_mean, dtype=torch.float32, device=device)) / torch.tensor(
            x_std, dtype=torch.float32, device=device
        )
        pred = (
            model(x_norm) * torch.tensor(y_std, dtype=torch.float32, device=device)
            + torch.tensor(y_mean, dtype=torch.float32, device=device)
        ).cpu().numpy()[0]

        msg = f"  Sample {i}  inputs: {fmt_seq(samp.x)}"
        for j, true_val in enumerate(samp.y):
            pred_val = float(pred[j])
            rel_err = abs(pred_val - true_val) / (abs(true_val) + 1e-10) * 100.0
            msg += (
                f"  true={fmt_sci(true_val)}"
                f"  pred={fmt_sci(pred_val)}"
                f"  rel_err={fmt_percent(rel_err)}%"
            )
        print(msg)


# -----------------------------------------------------------------------------
# Save text model compatible with existing C++ loader
# -----------------------------------------------------------------------------
def export_to_cpp_text(
    model: FrictionMLP,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    out_file: str,
) -> None:
    layers = [m for m in model.net if isinstance(m, nn.Linear)]
    assert len(layers) == 3, "Expected exactly 3 Linear layers"

    l1, l2, l3 = layers
    in_dim = l1.in_features
    h1 = l1.out_features
    h2 = l2.out_features
    out_dim = l3.out_features

    with open(out_file, "w") as f:
        f.write(f"IN {in_dim}\n")
        f.write(f"H1 {h1}\n")
        f.write(f"H2 {h2}\n")
        f.write(f"OUT {out_dim}\n")

        for j in range(in_dim):
            f.write(f"x_mean {float(x_mean[j]):.17g}\n")
            f.write(f"x_std {float(x_std[j]):.17g}\n")
        for j in range(out_dim):
            f.write(f"y_mean {float(y_mean[j]):.17g}\n")
            f.write(f"y_std {float(y_std[j]):.17g}\n")

        def write_linear(tag_w: str, tag_b: str, layer: nn.Linear) -> None:
            w = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            f.write(f"{tag_w}\n")
            for i in range(w.shape[0]):
                f.write(" ".join(f"{float(v):.17g}" for v in w[i]) + "\n")
            f.write(f"{tag_b}\n")
            f.write(" ".join(f"{float(v):.17g}" for v in b) + "\n")

        write_linear("W1", "b1", l1)
        write_linear("W2", "b2", l2)
        write_linear("W3", "b3", l3)

    print(f"Saved C++ text model to {out_file}")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_one_seed(
    seed: int,
    train_x_raw: np.ndarray,
    train_y_raw: np.ndarray,
    val_x_raw: np.ndarray,
    val_y_raw: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    in_dim: int,
    h1: int,
    h2: int,
    out_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    print_every: int,
    device: torch.device,
) -> Tuple[FrictionMLP, float]:
    set_seed(seed)

    model = FrictionMLP(in_dim=in_dim, h1=h1, h2=h2, out_dim=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_x = (train_x_raw - x_mean) / x_std
    train_y = (train_y_raw - y_mean) / y_std

    train_ds = TensorDataset(
        torch.from_numpy(train_x.astype(np.float32)),
        torch.from_numpy(train_y.astype(np.float32)),
    )

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=g,
    )

    best_val_rmse = float("inf")
    best_state = None

    print("\n==============================")
    print(f"Seed {seed}")
    print("==============================")

    y_std_scalar = float(y_std[0]) if len(y_std) == 1 else None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.shape[0]
            total_count += xb.shape[0]

        if epoch % print_every == 0 or epoch == epochs - 1:
            train_rmse_norm = math.sqrt(total_loss / max(total_count, 1))
            if y_std_scalar is not None:
                train_rmse = train_rmse_norm * y_std_scalar
            else:
                train_rmse = train_rmse_norm

            val_rmse = compute_rmse(
                model,
                val_x_raw,
                val_y_raw,
                x_mean,
                x_std,
                y_mean,
                y_std,
                device,
                batch_size=batch_size,
            )
            print(
                f"Epoch {epoch}  Train RMSE = {fmt_sci(train_rmse)}"
                f"  Val RMSE = {fmt_sci(val_rmse)}"
            )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_rmse = compute_rmse(
        model,
        val_x_raw,
        val_y_raw,
        x_mean,
        x_std,
        y_mean,
        y_std,
        device,
        batch_size=batch_size,
    )
    print(f"Seed {seed} final Val RMSE = {fmt_sci(final_val_rmse)}")
    return model, final_val_rmse


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train friction emulator with PyTorch")
    parser.add_argument("--folder", type=str, default="./data")
    parser.add_argument("--model-file", type=str, default="./friction_emulator.txt")
    parser.add_argument("--checkpoint", type=str, default="./friction_emulator.pt")
    parser.add_argument("--n-ranks", type=int, default=1)
    parser.add_argument("--in-dim", type=int, default=2)
    parser.add_argument("--out-dim", type=int, default=1)
    parser.add_argument("--h1", type=int, default=64)
    parser.add_argument("--h2", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")

    print("========================================")
    print("FrictionEmulator Training (PyTorch)")
    print("========================================")
    print(f"Folder:       {args.folder}")
    print(f"N ranks:      {args.n_ranks}")
    print(f"Epochs:       {args.epochs}")
    print(f"LR:           {fmt_sci(args.lr)}")
    print(f"Batch size:   {args.batch_size}")
    print(f"N seeds:      {args.n_seeds}")
    print(f"Model file:   {args.model_file}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(
        f"Architecture: {args.in_dim} -> {args.h1} ->"
        f" {args.h2} -> {args.out_dim}"
    )
    print(f"Device:       {device}")
    print("========================================")

    all_data = load_data(args.folder, args.n_ranks, args.in_dim, args.out_dim)
    if not all_data:
        raise RuntimeError("No data loaded. Exiting.")

    train_data, val_data, test_data = split_data(all_data, 0.70, 0.15, split_seed=42)
    if args.in_dim != len(FIXED_X_MEAN):
        raise RuntimeError(
            f"Expected in_dim={len(FIXED_X_MEAN)} for hard-coded normalization, got {args.in_dim}"
        )
    if args.out_dim != len(FIXED_Y_MEAN):
        raise RuntimeError(
            f"Expected out_dim={len(FIXED_Y_MEAN)} for hard-coded normalization, got {args.out_dim}"
        )

    x_mean = FIXED_X_MEAN.copy()
    x_std = FIXED_X_STD.copy()
    y_mean = FIXED_Y_MEAN.copy()
    y_std = FIXED_Y_STD.copy()

    print(f"x_mean:       {fmt_seq(x_mean.tolist())}")
    print(f"x_std:        {fmt_seq(x_std.tolist())}")
    print(f"y_mean:       {fmt_seq(y_mean.tolist())}")
    print(f"y_std:        {fmt_seq(y_std.tolist())}")

    train_x_raw, train_y_raw = to_numpy(train_data)
    val_x_raw, val_y_raw = to_numpy(val_data)
    test_x_raw, test_y_raw = to_numpy(test_data)

    best_model = None
    best_val_rmse = float("inf")

    for seed in range(args.n_seeds):
        model, val_rmse = train_one_seed(
            seed=seed,
            train_x_raw=train_x_raw,
            train_y_raw=train_y_raw,
            val_x_raw=val_x_raw,
            val_y_raw=val_y_raw,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            in_dim=args.in_dim,
            h1=args.h1,
            h2=args.h2,
            out_dim=args.out_dim,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            print_every=args.print_every,
            device=device,
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model.cpu()
            print(f"--> New best model at seed {seed}")

    assert best_model is not None

    print("\n========================================")
    print(f"Best model (Val RMSE = {fmt_sci(best_val_rmse)})")
    print("========================================")

    train_rmse = compute_rmse(best_model, train_x_raw, train_y_raw, x_mean, x_std, y_mean, y_std, torch.device("cpu"), args.batch_size)
    val_rmse = compute_rmse(best_model, val_x_raw, val_y_raw, x_mean, x_std, y_mean, y_std, torch.device("cpu"), args.batch_size)
    test_rmse = compute_rmse(best_model, test_x_raw, test_y_raw, x_mean, x_std, y_mean, y_std, torch.device("cpu"), args.batch_size)

    print(f"Train RMSE      = {fmt_sci(train_rmse)}")
    print(f"Validation RMSE = {fmt_sci(val_rmse)}")
    print(f"Test RMSE       = {fmt_sci(test_rmse)}")

    print_samples(best_model, test_data, x_mean, x_std, y_mean, y_std, torch.device("cpu"), n_samples=5)

    checkpoint = {
        "state_dict": best_model.state_dict(),
        "in_dim": args.in_dim,
        "h1": args.h1,
        "h2": args.h2,
        "out_dim": args.out_dim,
        "x_mean": x_mean.astype(np.float64),
        "x_std": x_std.astype(np.float64),
        "y_mean": y_mean.astype(np.float64),
        "y_std": y_std.astype(np.float64),
    }
    torch.save(checkpoint, args.checkpoint)
    print(f"Saved PyTorch checkpoint to {args.checkpoint}")

    export_to_cpp_text(best_model, x_mean, x_std, y_mean, y_std, args.model_file)
    print("\nDone.")


if __name__ == "__main__":
    main()
