#!/usr/bin/env python3
"""
Train a Weertman friction emulator in PyTorch.

Data format:
  folder/weertman_rank_0.csv
  folder/weertman_rank_1.csv
  ...
Each CSV row has 3 columns:
  C2, vmag, alpha2

This script does four things:
1. trains an MLP in PyTorch
2. saves a native PyTorch checkpoint (.pt)
3. saves a plain-text model file compatible with the existing C++ FrictionEmulator::Load()
4. saves numerical report data for a separate plotting process

Example:
  python train.py \
      --n-ranks 5 \
      --epochs 5000 \
      --lr 1e-3 \
      --batch-size 4096 \
      --n-seeds 5 \
      --folder ./data \
      --model-file ./friction_emulator.txt \
      --checkpoint ./friction_emulator.pt \
      --report-data ./training_report_data.npz

Generate plots afterward in a separate process:
  python plot_training_results.py \
      --report-data ./training_report_data.npz \
      --plots-dir ./plots
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from accuracy_reports import (
    compute_accuracy_metrics,
    print_accuracy_metrics,
    save_accuracy_report_data,
)
from preprocessing import (
    RAW_X_SCALE,
    RAW_Y_SCALE,
    NormalizationConfig,
    build_normalization_config,
    inverse_transform_y_torch,
    transform_x_np,
    transform_x_torch,
    transform_y_np,
)


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


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.get_device_name(index)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return "CPU"


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


def filter_min_vmag_samples(
    data: List[FrictionSample],
    min_vmag: float,
) -> Tuple[List[FrictionSample], int]:
    filtered = [sample for sample in data if sample.x[1] >= min_vmag]
    removed = len(data) - len(filtered)
    if not filtered:
        raise RuntimeError(f"Velocity filter removed all samples with vmag < {min_vmag:.6e}.")
    print(
        f"Velocity filter kept {len(filtered)} of {len(data)} samples; "
        f"removed {removed} with vmag < {min_vmag:.6e}."
    )
    return filtered, removed


def sqrt_joint_sample_indices(
    x_raw: np.ndarray,
    n_samples: int,
    c2_bins: int,
    vmag_bins: int,
    seed: int,
) -> np.ndarray:
    if n_samples <= 0:
        raise RuntimeError("--train-samples must be > 0")

    x = np.asarray(x_raw, dtype=np.float64).reshape(len(x_raw), -1)
    c2 = np.sqrt(np.maximum(x[:, 0], 0.0))
    vmag = np.sqrt(np.maximum(x[:, 1], 0.0))
    finite_mask = np.isfinite(c2) & np.isfinite(vmag)
    finite_indices = np.flatnonzero(finite_mask)
    if len(finite_indices) == 0:
        raise RuntimeError("No finite sqrt(C2), sqrt(vmag) pairs for training sampling.")

    rng = np.random.default_rng(seed)
    if n_samples >= len(finite_indices):
        return rng.permutation(finite_indices)

    c2_finite = c2[finite_mask]
    vmag_finite = vmag[finite_mask]
    c2_edges = np.linspace(float(c2_finite.min()), float(c2_finite.max()), c2_bins + 1)
    vmag_edges = np.linspace(float(vmag_finite.min()), float(vmag_finite.max()), vmag_bins + 1)

    c2_ids = np.digitize(c2, c2_edges[1:-1], right=False)
    vmag_ids = np.digitize(vmag, vmag_edges[1:-1], right=False)
    joint_ids = c2_ids * vmag_bins + vmag_ids
    n_joint_bins = c2_bins * vmag_bins

    counts = np.bincount(joint_ids[finite_mask], minlength=n_joint_bins).astype(np.float64)
    occupied = counts > 0
    if not np.any(occupied):
        raise RuntimeError("No occupied sqrt(C2), sqrt(vmag) bins were found for training sampling.")

    bin_weights = np.zeros(n_joint_bins, dtype=np.float64)
    bin_weights[occupied] = 1.0 / counts[occupied]
    sample_weights = np.where(finite_mask, bin_weights[joint_ids], 0.0)
    if sample_weights.sum() <= 0.0:
        raise RuntimeError("Joint training sampling weights sum to zero.")

    return rng.choice(
        len(x),
        size=n_samples,
        replace=False,
        p=sample_weights / sample_weights.sum(),
    )


def sqrt_joint_sample_training_data(
    data: List[FrictionSample],
    n_samples: int,
    c2_bins: int,
    vmag_bins: int,
    seed: int,
) -> List[FrictionSample]:
    x_raw, _ = to_numpy(data)
    indices = sqrt_joint_sample_indices(
        x_raw,
        n_samples=min(n_samples, len(data)),
        c2_bins=c2_bins,
        vmag_bins=vmag_bins,
        seed=seed,
    )
    sampled = [data[int(i)] for i in indices]
    print(
        f"Joint train sampler kept {len(sampled)} of {len(data)} training samples "
        f"using {c2_bins} sqrt(C2) bins x {vmag_bins} sqrt(vmag) bins."
    )
    return sampled


def print_regime_metrics(
    split_name: str,
    x_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slow_vmag: float,
    fast_vmag: float,
) -> None:
    c2 = np.asarray(x_raw, dtype=np.float64)[:, 0]
    vmag = np.asarray(x_raw, dtype=np.float64)[:, 1]
    c2_q25 = float(np.percentile(c2, 25.0))
    c2_q75 = float(np.percentile(c2, 75.0))
    regimes = {
        f"slow_v<{slow_vmag:.1e}": vmag < slow_vmag,
        f"medium_{slow_vmag:.1e}-{fast_vmag:.1e}": (vmag >= slow_vmag) & (vmag < fast_vmag),
        f"fast_v>={fast_vmag:.1e}": vmag >= fast_vmag,
        "low_C2_q25": c2 <= c2_q25,
        "high_C2_q75": c2 >= c2_q75,
        "fast_low_C2": (vmag >= fast_vmag) & (c2 <= c2_q25),
    }

    print(f"\n{split_name} regime metrics")
    print("-" * (len(split_name) + 15))
    for name, mask in regimes.items():
        count = int(np.count_nonzero(mask))
        if count == 0:
            print(f"{name:<28} n=0")
            continue
        metrics = compute_accuracy_metrics(y_true[mask], y_pred[mask])
        print_accuracy_metrics(f"{name} n={count}", metrics)


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
    norm: NormalizationConfig,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total_sse = 0.0
    total_count = 0

    x_mean_t = torch.as_tensor(norm.x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.as_tensor(norm.x_std, dtype=torch.float32, device=device)
    y_mean_t = torch.as_tensor(norm.y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(norm.y_std, dtype=torch.float32, device=device)

    for start in range(0, len(x_raw), batch_size):
        end = min(start + batch_size, len(x_raw))
        xb_raw = torch.as_tensor(x_raw[start:end], dtype=torch.float32, device=device)
        yb_raw = torch.as_tensor(y_raw[start:end], dtype=torch.float32, device=device)

        xb_trans = transform_x_torch(xb_raw, norm, device)
        xb = (xb_trans - x_mean_t) / x_std_t
        pred_norm = model(xb)
        pred_trans = pred_norm * y_std_t + y_mean_t
        pred_raw = inverse_transform_y_torch(pred_trans, norm)

        err = pred_raw - yb_raw
        total_sse += torch.sum(err * err).item()
        total_count += err.numel()

    return math.sqrt(total_sse / max(total_count, 1))


@torch.no_grad()
def predict_raw(
    model: nn.Module,
    x_raw: np.ndarray,
    norm: NormalizationConfig,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds = []

    x_mean_t = torch.as_tensor(norm.x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.as_tensor(norm.x_std, dtype=torch.float32, device=device)
    y_mean_t = torch.as_tensor(norm.y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(norm.y_std, dtype=torch.float32, device=device)

    for start in range(0, len(x_raw), batch_size):
        end = min(start + batch_size, len(x_raw))
        xb_raw = torch.as_tensor(x_raw[start:end], dtype=torch.float32, device=device)
        xb_trans = transform_x_torch(xb_raw, norm, device)
        xb = (xb_trans - x_mean_t) / x_std_t
        pred_trans = model(xb) * y_std_t + y_mean_t
        pred_raw = inverse_transform_y_torch(pred_trans, norm)
        preds.append(pred_raw.detach().cpu().numpy())

    return np.concatenate(preds, axis=0)


@torch.no_grad()
def print_mlp_samples(
    model: FrictionMLP,
    data: List[FrictionSample],
    norm: NormalizationConfig,
    device: torch.device,
    n_samples: int = 5,
) -> None:
    model.eval()

    print("\nSample predictions:")
    for i, samp in enumerate(data[:n_samples]):
        x_input = np.asarray([samp.x], dtype=np.float32)
        x_raw = torch.as_tensor(x_input, dtype=torch.float32, device=device)
        x_mean_t = torch.as_tensor(norm.x_mean, dtype=torch.float32, device=device)
        x_std_t = torch.as_tensor(norm.x_std, dtype=torch.float32, device=device)
        y_mean_t = torch.as_tensor(norm.y_mean, dtype=torch.float32, device=device)
        y_std_t = torch.as_tensor(norm.y_std, dtype=torch.float32, device=device)
        x_trans = transform_x_torch(x_raw, norm, device)
        x_norm = (x_trans - x_mean_t) / x_std_t
        pred_trans = model(x_norm) * y_std_t + y_mean_t
        pred = inverse_transform_y_torch(pred_trans, norm).cpu().numpy()[0]

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


def print_sample_predictions(data: List[FrictionSample], predictions: np.ndarray, n_samples: int = 5) -> None:
    print("\nSample predictions:")
    for i, samp in enumerate(data[:n_samples]):
        msg = f"  Sample {i}  inputs: {fmt_seq(samp.x)}"
        for j, true_val in enumerate(samp.y):
            pred_val = float(predictions[i, j])
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
    norm: NormalizationConfig,
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
        if norm.mode != "raw":
            f.write(f"NORMALIZATION {norm.mode}\n")
            if norm.mode in ("log", "mixed"):
                f.write("LOG_BASE e\n")
            if norm.mode == "mixed":
                f.write("x_transform raw\n")
                f.write("x_transform log\n")
                f.write("y_transform log\n")
            elif norm.mode == "sqrt":
                for _ in range(in_dim):
                    f.write("x_transform sqrt\n")
                for _ in range(out_dim):
                    f.write("y_transform sqrt\n")
            for j in range(in_dim):
                f.write(f"x_floor {float(norm.x_floor[j]):.17g}\n")
            for j in range(out_dim):
                f.write(f"y_floor {float(norm.y_floor[j]):.17g}\n")

        for j in range(in_dim):
            f.write(f"x_mean {float(norm.x_mean[j]):.17g}\n")
            f.write(f"x_std {float(norm.x_std[j]):.17g}\n")
        for j in range(out_dim):
            f.write(f"y_mean {float(norm.y_mean[j]):.17g}\n")
            f.write(f"y_std {float(norm.y_std[j]):.17g}\n")

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
    norm: NormalizationConfig,
    in_dim: int,
    h1: int,
    h2: int,
    out_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    print_every: int,
    lr_scheduler: str,
    lr_patience: int,
    lr_factor: float,
    lr_min: float,
    lr_threshold: float,
    device: torch.device,
) -> Tuple[FrictionMLP, float, List[Dict[str, float]]]:
    set_seed(seed)

    model = FrictionMLP(in_dim=in_dim, h1=h1, h2=h2, out_dim=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=lr_threshold,
            threshold_mode="rel",
            min_lr=lr_min,
        )
    criterion = nn.MSELoss()

    train_x = (transform_x_np(train_x_raw, norm.mode, norm.x_floor) - norm.x_mean) / norm.x_std
    train_y = (transform_y_np(train_y_raw, norm.mode, norm.y_floor) - norm.y_mean) / norm.y_std

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
    history = []

    print("\n==============================")
    print(f"Seed {seed}")
    print("==============================")

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
            train_rmse = compute_rmse(
                model,
                train_x_raw,
                train_y_raw,
                norm,
                device,
                batch_size=batch_size,
            )

            val_rmse = compute_rmse(
                model,
                val_x_raw,
                val_y_raw,
                norm,
                device,
                batch_size=batch_size,
            )
            old_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step(val_rmse)
            current_lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"Epoch {epoch}  Train RMSE = {fmt_sci(train_rmse)}"
                f"  Val RMSE = {fmt_sci(val_rmse)}"
                f"  LR = {fmt_sci(current_lr)}"
            )
            if current_lr < old_lr:
                print(f"  LR reduced: {fmt_sci(old_lr)} -> {fmt_sci(current_lr)}")
            history.append(
                {
                    "epoch": float(epoch),
                    "train_rmse": float(train_rmse),
                    "val_rmse": float(val_rmse),
                    "train_mse_norm": float(total_loss / max(total_count, 1)),
                    "lr": current_lr,
                }
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
        norm,
        device,
        batch_size=batch_size,
    )
    print(f"Seed {seed} final Val RMSE = {fmt_sci(final_val_rmse)}")
    return model, final_val_rmse, history


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
    parser.add_argument("--lr-scheduler", type=str, default="none", choices=["none", "plateau"])
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-threshold", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--report-data", type=str, default="./training_report_data.npz")
    parser.add_argument(
        "--normalization",
        type=str,
        default="sqrt",
        choices=["raw", "sqrt", "log", "mixed"],
        help=(
            "Input/output preprocessing before scaling. "
            "'raw' applies only the hard-coded raw scaling constants."
        ),
    )
    parser.add_argument("--min-vmag", type=float, default=5e-8)
    parser.add_argument(
        "--disable-vmag-filter",
        action="store_true",
        help="Keep all rows instead of removing rows with vmag below --min-vmag.",
    )
    parser.add_argument("--train-samples", type=int, default=30000)
    parser.add_argument(
        "--disable-joint-sampling",
        action="store_true",
        help="Use the full training split instead of sqrt(C2),sqrt(vmag) joint sampling.",
    )
    parser.add_argument("--joint-c2-bins", type=int, default=30)
    parser.add_argument("--joint-vmag-bins", type=int, default=30)
    parser.add_argument("--joint-sampling-seed", type=int, default=42)
    parser.add_argument("--slow-vmag-threshold", type=float, default=1e-6)
    parser.add_argument("--fast-vmag-threshold", type=float, default=1e-5)
    args = parser.parse_args()

    if not (0.0 < args.lr_factor < 1.0):
        raise RuntimeError("--lr-factor must be between 0 and 1")
    if args.min_vmag <= 0.0:
        raise RuntimeError("--min-vmag must be > 0")
    if args.train_samples <= 0:
        raise RuntimeError("--train-samples must be > 0")
    if args.joint_c2_bins < 1 or args.joint_vmag_bins < 1:
        raise RuntimeError("--joint-c2-bins and --joint-vmag-bins must be >= 1")
    if not args.disable_vmag_filter and args.slow_vmag_threshold <= args.min_vmag:
        raise RuntimeError("--slow-vmag-threshold must be greater than --min-vmag")
    if args.slow_vmag_threshold <= 0.0:
        raise RuntimeError("--slow-vmag-threshold must be > 0")
    if args.fast_vmag_threshold <= args.slow_vmag_threshold:
        raise RuntimeError("--fast-vmag-threshold must be greater than --slow-vmag-threshold")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps was requested, but MPS is not available")

    print("========================================")
    print("FrictionEmulator Training (PyTorch)")
    print("========================================")
    print(f"Folder:       {args.folder}")
    print(f"N ranks:      {args.n_ranks}")
    print(f"Epochs:       {args.epochs}")
    print(f"LR:           {fmt_sci(args.lr)}")
    print(f"LR scheduler: {args.lr_scheduler}")
    if args.lr_scheduler == "plateau":
        print(
            "LR cfg:       "
            f"patience={args.lr_patience}, "
            f"factor={args.lr_factor:g}, "
            f"min_lr={fmt_sci(args.lr_min)}, "
            f"threshold={args.lr_threshold:g}"
        )
    print(f"Batch size:   {args.batch_size}")
    print(f"N seeds:      {args.n_seeds}")
    print(f"Model file:   {args.model_file}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Report data:  {args.report_data}")
    print(f"Normalization:{args.normalization}")
    if args.disable_vmag_filter:
        print("Min vmag:     disabled")
    else:
        print(f"Min vmag:     {fmt_sci(args.min_vmag)}")
    if args.disable_joint_sampling:
        print("Train sample: disabled; using full training split")
    else:
        print(
            "Train sample: "
            f"{args.train_samples} samples, "
            f"{args.joint_c2_bins} sqrt(C2) bins x {args.joint_vmag_bins} sqrt(vmag) bins"
        )
    print(
        "Regimes:      "
        f"slow < {fmt_sci(args.slow_vmag_threshold)}, "
        f"fast >= {fmt_sci(args.fast_vmag_threshold)}"
    )
    print(
        f"Architecture: {args.in_dim} -> {args.h1} ->"
        f" {args.h2} -> {args.out_dim}"
    )
    print(f"Device:       {device} ({describe_device(device)})")
    print("========================================")

    all_data = load_data(args.folder, args.n_ranks, args.in_dim, args.out_dim)
    if not all_data:
        raise RuntimeError("No data loaded. Exiting.")

    x_floor = np.array([1e-30, 1e-12], dtype=np.float64)
    y_floor = np.array([1e-30], dtype=np.float64)

    removed_low_vmag = 0
    if not args.disable_vmag_filter:
        all_data, removed_low_vmag = filter_min_vmag_samples(all_data, args.min_vmag)

    train_data, val_data, test_data = split_data(all_data, 0.70, 0.15, split_seed=42)
    full_train_count = len(train_data)
    if not args.disable_joint_sampling:
        train_data = sqrt_joint_sample_training_data(
            train_data,
            n_samples=args.train_samples,
            c2_bins=args.joint_c2_bins,
            vmag_bins=args.joint_vmag_bins,
            seed=args.joint_sampling_seed,
        )
    else:
        print(f"Joint train sampler disabled; using all {len(train_data)} training samples.")
    if args.in_dim != len(RAW_X_SCALE):
        raise RuntimeError(
            f"Expected in_dim={len(RAW_X_SCALE)} for hard-coded normalization, got {args.in_dim}"
        )
    if args.out_dim != len(RAW_Y_SCALE):
        raise RuntimeError(
            f"Expected out_dim={len(RAW_Y_SCALE)} for hard-coded normalization, got {args.out_dim}"
        )

    train_x_raw, train_y_raw = to_numpy(train_data)
    val_x_raw, val_y_raw = to_numpy(val_data)
    test_x_raw, test_y_raw = to_numpy(test_data)
    norm = build_normalization_config(args.normalization, train_x_raw, train_y_raw, x_floor, y_floor)

    print(f"x_mean:       {fmt_seq(norm.x_mean.tolist())}")
    print(f"x_std:        {fmt_seq(norm.x_std.tolist())}")
    print(f"y_mean:       {fmt_seq(norm.y_mean.tolist())}")
    print(f"y_std:        {fmt_seq(norm.y_std.tolist())}")
    if norm.mode in ("log", "mixed"):
        print(f"x_floor:      {fmt_seq(norm.x_floor.tolist())}")
        print(f"y_floor:      {fmt_seq(norm.y_floor.tolist())}")

    best_model = None
    best_val_rmse = float("inf")
    best_seed = None
    history_by_seed: Dict[int, List[Dict[str, float]]] = {}

    for seed in range(args.n_seeds):
        model, val_rmse, history = train_one_seed(
            seed=seed,
            train_x_raw=train_x_raw,
            train_y_raw=train_y_raw,
            val_x_raw=val_x_raw,
            val_y_raw=val_y_raw,
            norm=norm,
            in_dim=args.in_dim,
            h1=args.h1,
            h2=args.h2,
            out_dim=args.out_dim,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            print_every=args.print_every,
            lr_scheduler=args.lr_scheduler,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            lr_min=args.lr_min,
            lr_threshold=args.lr_threshold,
            device=device,
        )
        history_by_seed[seed] = history

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model.cpu()
            best_seed = seed
            print(f"--> New best model at seed {seed}")

    assert best_model is not None
    assert best_seed is not None

    print("\n========================================")
    print(f"Best model (seed {best_seed}, Val RMSE = {fmt_sci(best_val_rmse)})")
    print("========================================")

    eval_device = torch.device("cpu")
    split_predictions = {
        "train": (
            train_y_raw,
            predict_raw(best_model, train_x_raw, norm, eval_device, args.batch_size),
        ),
        "validation": (
            val_y_raw,
            predict_raw(best_model, val_x_raw, norm, eval_device, args.batch_size),
        ),
        "test": (
            test_y_raw,
            predict_raw(best_model, test_x_raw, norm, eval_device, args.batch_size),
        ),
    }

    print_accuracy_metrics("Train", compute_accuracy_metrics(*split_predictions["train"]))
    print_accuracy_metrics("Validation", compute_accuracy_metrics(*split_predictions["validation"]))
    print_accuracy_metrics("Test", compute_accuracy_metrics(*split_predictions["test"]))
    print_regime_metrics(
        "Validation",
        val_x_raw,
        split_predictions["validation"][0],
        split_predictions["validation"][1],
        slow_vmag=args.slow_vmag_threshold,
        fast_vmag=args.fast_vmag_threshold,
    )
    print_regime_metrics(
        "Test",
        test_x_raw,
        split_predictions["test"][0],
        split_predictions["test"][1],
        slow_vmag=args.slow_vmag_threshold,
        fast_vmag=args.fast_vmag_threshold,
    )

    print_sample_predictions(test_data, split_predictions["test"][1], n_samples=5)
    save_accuracy_report_data(
        args.report_data,
        split_predictions,
        history_by_seed,
        best_seed,
        test_x_raw,
    )

    common_metadata = {
        "in_dim": args.in_dim,
        "out_dim": args.out_dim,
        "normalization": norm.mode,
        "x_mean": norm.x_mean.astype(np.float64),
        "x_std": norm.x_std.astype(np.float64),
        "y_mean": norm.y_mean.astype(np.float64),
        "y_std": norm.y_std.astype(np.float64),
        "x_floor": norm.x_floor.astype(np.float64),
        "y_floor": norm.y_floor.astype(np.float64),
        "min_vmag": args.min_vmag,
        "vmag_filter_enabled": not args.disable_vmag_filter,
        "removed_low_vmag": removed_low_vmag,
        "full_train_samples_before_joint_sampling": full_train_count,
        "train_samples": len(train_data),
        "joint_sampling_enabled": not args.disable_joint_sampling,
        "joint_c2_bins": args.joint_c2_bins,
        "joint_vmag_bins": args.joint_vmag_bins,
        "joint_sampling_seed": args.joint_sampling_seed,
        "slow_vmag_threshold": args.slow_vmag_threshold,
        "fast_vmag_threshold": args.fast_vmag_threshold,
        "lr_scheduler": args.lr_scheduler,
        "lr_patience": args.lr_patience,
        "lr_factor": args.lr_factor,
        "lr_min": args.lr_min,
        "lr_threshold": args.lr_threshold,
    }
    checkpoint = {
        **common_metadata,
        "state_dict": best_model.state_dict(),
        "h1": args.h1,
        "h2": args.h2,
    }
    torch.save(checkpoint, args.checkpoint)
    print(f"Saved PyTorch checkpoint to {args.checkpoint}")
    export_to_cpp_text(best_model, norm, args.model_file)
    print("\nDone.")


if __name__ == "__main__":
    main()
