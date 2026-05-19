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
  python train.py \
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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from accuracy_reports import (
    compute_accuracy_metrics,
    print_accuracy_metrics,
    save_accuracy_outputs,
)


RAW_X_OFFSET = np.array([0.0, 0.0], dtype=np.float64)
RAW_X_SCALE = np.array([9.05e6, 2.08e-5], dtype=np.float64)
RAW_Y_OFFSET = np.array([0.0], dtype=np.float64)
RAW_Y_SCALE = np.array([2.09e11], dtype=np.float64)


@dataclass
class NormalizationConfig:
    mode: str
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    x_floor: np.ndarray
    y_floor: np.ndarray


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


def build_normalization_config(
    mode: str,
    train_x_raw: np.ndarray,
    train_y_raw: np.ndarray,
    x_floor: np.ndarray,
    y_floor: np.ndarray,
) -> NormalizationConfig:
    if mode == "raw":
        return NormalizationConfig(
            mode=mode,
            x_mean=RAW_X_OFFSET.copy(),
            x_std=RAW_X_SCALE.copy(),
            y_mean=RAW_Y_OFFSET.copy(),
            y_std=RAW_Y_SCALE.copy(),
            x_floor=x_floor.astype(np.float64),
            y_floor=y_floor.astype(np.float64),
        )

    if mode == "mixed":
        x_trans = transform_x_np(train_x_raw, mode, x_floor)
        y_trans = transform_y_np(train_y_raw, mode, y_floor)
        x_mean = np.array([0.0, x_trans[:, 1].mean()], dtype=np.float64)
        x_std = np.array([RAW_X_SCALE[0], x_trans[:, 1].std()], dtype=np.float64)
        y_mean = y_trans.mean(axis=0)
        y_std = y_trans.std(axis=0)
        x_std[x_std < 1e-12] = 1.0
        y_std[y_std < 1e-12] = 1.0
        return NormalizationConfig(
            mode=mode,
            x_mean=x_mean.astype(np.float64),
            x_std=x_std.astype(np.float64),
            y_mean=y_mean.astype(np.float64),
            y_std=y_std.astype(np.float64),
            x_floor=x_floor.astype(np.float64),
            y_floor=y_floor.astype(np.float64),
        )

    if mode != "log":
        raise RuntimeError(f"Unsupported normalization mode: {mode}")

    x_trans = transform_x_np(train_x_raw, mode, x_floor)
    y_trans = transform_y_np(train_y_raw, mode, y_floor)
    x_mean = x_trans.mean(axis=0)
    x_std = x_trans.std(axis=0)
    y_mean = y_trans.mean(axis=0)
    y_std = y_trans.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_std[y_std < 1e-12] = 1.0
    return NormalizationConfig(
        mode=mode,
        x_mean=x_mean.astype(np.float64),
        x_std=x_std.astype(np.float64),
        y_mean=y_mean.astype(np.float64),
        y_std=y_std.astype(np.float64),
        x_floor=x_floor.astype(np.float64),
        y_floor=y_floor.astype(np.float64),
    )


def transform_x_np(x_raw: np.ndarray, mode: str, x_floor: np.ndarray) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    if mode == "raw":
        return x
    if mode == "mixed":
        x_trans = x.copy()
        x_trans[:, 1] = np.log(np.maximum(x[:, 1], x_floor[1]))
        return x_trans
    return np.log(np.maximum(x, x_floor))


def transform_y_np(y_raw: np.ndarray, mode: str, y_floor: np.ndarray) -> np.ndarray:
    y = np.asarray(y_raw, dtype=np.float64)
    if mode == "raw":
        return y
    return np.log(np.maximum(y, y_floor))


def normalize_x_np(x_raw: np.ndarray, norm: NormalizationConfig) -> np.ndarray:
    return (transform_x_np(x_raw, norm.mode, norm.x_floor) - norm.x_mean) / norm.x_std


def normalize_y_np(y_raw: np.ndarray, norm: NormalizationConfig) -> np.ndarray:
    return (transform_y_np(y_raw, norm.mode, norm.y_floor) - norm.y_mean) / norm.y_std


def inverse_transform_y_np(y_trans: np.ndarray, norm: NormalizationConfig) -> np.ndarray:
    if norm.mode == "raw":
        return y_trans
    return np.exp(y_trans)


def denormalize_y_np(y_norm: np.ndarray, norm: NormalizationConfig) -> np.ndarray:
    return inverse_transform_y_np(y_norm * norm.y_std + norm.y_mean, norm)


def transform_x_torch(x_raw: torch.Tensor, norm: NormalizationConfig, device: torch.device) -> torch.Tensor:
    if norm.mode == "raw":
        return x_raw
    x_floor = torch.as_tensor(norm.x_floor, dtype=torch.float32, device=device)
    if norm.mode == "mixed":
        c2 = x_raw[:, 0:1]
        vmag = torch.log(torch.maximum(x_raw[:, 1:2], x_floor[1]))
        return torch.cat((c2, vmag), dim=1)
    return torch.log(torch.maximum(x_raw, x_floor))


def inverse_transform_y_torch(y_trans: torch.Tensor, norm: NormalizationConfig) -> torch.Tensor:
    if norm.mode == "raw":
        return y_trans
    return torch.exp(y_trans)


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


def predict_xgboost_raw(model, x_raw: np.ndarray, norm: NormalizationConfig, iteration: int | None = None) -> np.ndarray:
    import xgboost as xgb

    x_norm = normalize_x_np(x_raw, norm).astype(np.float32)
    dmat = xgb.DMatrix(x_norm)
    kwargs = {}
    if iteration is not None:
        kwargs["iteration_range"] = (0, iteration)
    pred_norm = np.asarray(model.predict(dmat, **kwargs), dtype=np.float64).reshape(-1, len(norm.y_mean))
    return denormalize_y_np(pred_norm, norm)


def compute_rmse_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    return float(np.sqrt(np.mean(err * err)))


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
            f.write("LOG_BASE e\n")
            if norm.mode == "mixed":
                f.write("x_transform raw\n")
                f.write("x_transform log\n")
                f.write("y_transform log\n")
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
    device: torch.device,
) -> Tuple[FrictionMLP, float, List[Dict[str, float]]]:
    set_seed(seed)

    model = FrictionMLP(in_dim=in_dim, h1=h1, h2=h2, out_dim=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            print(
                f"Epoch {epoch}  Train RMSE = {fmt_sci(train_rmse)}"
                f"  Val RMSE = {fmt_sci(val_rmse)}"
            )
            history.append(
                {
                    "epoch": float(epoch),
                    "train_rmse": float(train_rmse),
                    "val_rmse": float(val_rmse),
                    "train_mse_norm": float(total_loss / max(total_count, 1)),
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


def train_xgboost(
    train_x_raw: np.ndarray,
    train_y_raw: np.ndarray,
    val_x_raw: np.ndarray,
    val_y_raw: np.ndarray,
    norm: NormalizationConfig,
    n_estimators: int,
    max_depth: int,
    lr: float,
    print_every: int,
    seed: int,
    device: torch.device,
):
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "XGBoost is not installed. Install it or run with --model-type mlp."
        ) from exc

    train_x = normalize_x_np(train_x_raw, norm).astype(np.float32)
    train_y = normalize_y_np(train_y_raw, norm).astype(np.float32).reshape(-1)
    val_x = normalize_x_np(val_x_raw, norm).astype(np.float32)
    val_y = normalize_y_np(val_y_raw, norm).astype(np.float32).reshape(-1)

    xgb_device = "cuda" if device.type == "cuda" else "cpu"
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": max_depth,
        "eta": lr,
        "tree_method": "hist",
        "device": xgb_device,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "seed": seed,
    }
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dval = xgb.DMatrix(val_x, label=val_y)

    print("\n==============================")
    print("XGBoost")
    print("==============================")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "validation")],
        verbose_eval=False,
    )

    history = []
    checkpoints = sorted(set(list(range(1, n_estimators + 1, print_every)) + [n_estimators]))
    best_val_rmse = float("inf")
    best_iteration = n_estimators
    for iteration in checkpoints:
        train_pred = predict_xgboost_raw(model, train_x_raw, norm, iteration=iteration)
        val_pred = predict_xgboost_raw(model, val_x_raw, norm, iteration=iteration)
        train_rmse = compute_rmse_from_predictions(train_y_raw, train_pred)
        val_rmse = compute_rmse_from_predictions(val_y_raw, val_pred)
        epoch = iteration - 1
        print(
            f"Round {epoch}  Train RMSE = {fmt_sci(train_rmse)}"
            f"  Val RMSE = {fmt_sci(val_rmse)}"
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
                "train_mse_norm": float("nan"),
            }
        )
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_iteration = iteration

    print(f"XGBoost final Val RMSE = {fmt_sci(best_val_rmse)}")
    return model, best_val_rmse, history, best_iteration


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train friction emulator with PyTorch")
    parser.add_argument("--folder", type=str, default="./data")
    parser.add_argument("--model-file", type=str, default="./friction_emulator.txt")
    parser.add_argument("--checkpoint", type=str, default="./friction_emulator.pt")
    parser.add_argument("--xgb-model-file", type=str, default="./friction_emulator_xgboost.json")
    parser.add_argument("--model-type", type=str, default="mlp", choices=["mlp", "xgboost"])
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
    parser.add_argument("--xgb-n-estimators", type=int, default=500)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--plots-dir", type=str, default="./plots")
    parser.add_argument("--normalization", type=str, default="raw", choices=["raw", "log", "mixed"])
    parser.add_argument("--log-c2-floor", type=float, default=1e-30)
    parser.add_argument("--log-v-floor", type=float, default=1e-12)
    parser.add_argument("--log-alpha2-floor", type=float, default=1e-30)
    args = parser.parse_args()

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
    print(f"Batch size:   {args.batch_size}")
    print(f"N seeds:      {args.n_seeds}")
    print(f"Model type:   {args.model_type}")
    print(f"Model file:   {args.model_file}")
    print(f"XGB file:     {args.xgb_model_file}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Plots dir:    {args.plots_dir}")
    print(f"Normalization:{args.normalization}")
    print(
        f"Architecture: {args.in_dim} -> {args.h1} ->"
        f" {args.h2} -> {args.out_dim}"
    )
    print(f"Device:       {device} ({describe_device(device)})")
    print("========================================")

    all_data = load_data(args.folder, args.n_ranks, args.in_dim, args.out_dim)
    if not all_data:
        raise RuntimeError("No data loaded. Exiting.")

    train_data, val_data, test_data = split_data(all_data, 0.70, 0.15, split_seed=42)
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
    x_floor = np.array([args.log_c2_floor, args.log_v_floor], dtype=np.float64)
    y_floor = np.array([args.log_alpha2_floor], dtype=np.float64)
    norm = build_normalization_config(args.normalization, train_x_raw, train_y_raw, x_floor, y_floor)

    print(f"x_mean:       {fmt_seq(norm.x_mean.tolist())}")
    print(f"x_std:        {fmt_seq(norm.x_std.tolist())}")
    print(f"y_mean:       {fmt_seq(norm.y_mean.tolist())}")
    print(f"y_std:        {fmt_seq(norm.y_std.tolist())}")
    if norm.mode == "log":
        print(f"x_floor:      {fmt_seq(norm.x_floor.tolist())}")
        print(f"y_floor:      {fmt_seq(norm.y_floor.tolist())}")

    best_model = None
    best_val_rmse = float("inf")
    best_seed = None
    best_iteration = None
    history_by_seed: Dict[int, List[Dict[str, float]]] = {}

    if args.model_type == "mlp":
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
                device=device,
            )
            history_by_seed[seed] = history

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model = model.cpu()
                best_seed = seed
                print(f"--> New best model at seed {seed}")
    else:
        model, val_rmse, history, best_iteration = train_xgboost(
            train_x_raw=train_x_raw,
            train_y_raw=train_y_raw,
            val_x_raw=val_x_raw,
            val_y_raw=val_y_raw,
            norm=norm,
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            lr=args.lr,
            print_every=args.print_every,
            seed=0,
            device=device,
        )
        best_val_rmse = val_rmse
        best_model = model
        best_seed = 0
        history_by_seed[best_seed] = history
        print(f"--> New best XGBoost model at round {best_iteration - 1}")

    assert best_model is not None
    assert best_seed is not None

    print("\n========================================")
    print(f"Best model (seed {best_seed}, Val RMSE = {fmt_sci(best_val_rmse)})")
    print("========================================")

    eval_device = torch.device("cpu")
    if args.model_type == "mlp":
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
    else:
        split_predictions = {
            "train": (
                train_y_raw,
                predict_xgboost_raw(best_model, train_x_raw, norm, iteration=best_iteration),
            ),
            "validation": (
                val_y_raw,
                predict_xgboost_raw(best_model, val_x_raw, norm, iteration=best_iteration),
            ),
            "test": (
                test_y_raw,
                predict_xgboost_raw(best_model, test_x_raw, norm, iteration=best_iteration),
            ),
        }

    print_accuracy_metrics("Train", compute_accuracy_metrics(*split_predictions["train"]))
    print_accuracy_metrics("Validation", compute_accuracy_metrics(*split_predictions["validation"]))
    print_accuracy_metrics("Test", compute_accuracy_metrics(*split_predictions["test"]))

    print_sample_predictions(test_data, split_predictions["test"][1], n_samples=5)
    save_accuracy_outputs(args.plots_dir, split_predictions, history_by_seed, best_seed, test_x_raw)

    common_metadata = {
        "model_type": args.model_type,
        "in_dim": args.in_dim,
        "out_dim": args.out_dim,
        "normalization": norm.mode,
        "x_mean": norm.x_mean.astype(np.float64),
        "x_std": norm.x_std.astype(np.float64),
        "y_mean": norm.y_mean.astype(np.float64),
        "y_std": norm.y_std.astype(np.float64),
        "x_floor": norm.x_floor.astype(np.float64),
        "y_floor": norm.y_floor.astype(np.float64),
    }
    if args.model_type == "mlp":
        checkpoint = {
            **common_metadata,
            "state_dict": best_model.state_dict(),
            "h1": args.h1,
            "h2": args.h2,
        }
        torch.save(checkpoint, args.checkpoint)
        print(f"Saved PyTorch checkpoint to {args.checkpoint}")
        export_to_cpp_text(best_model, norm, args.model_file)
    else:
        best_model.save_model(args.xgb_model_file)
        checkpoint = {
            **common_metadata,
            "xgb_model_file": args.xgb_model_file,
            "xgb_n_estimators": args.xgb_n_estimators,
            "xgb_max_depth": args.xgb_max_depth,
            "best_iteration": best_iteration,
        }
        torch.save(checkpoint, args.checkpoint)
        print(f"Saved XGBoost model to {args.xgb_model_file}")
        print(f"Saved XGBoost metadata checkpoint to {args.checkpoint}")
        print("Skipped C++ text export: friction_emulator.txt is currently an MLP-only format.")
    print("\nDone.")


if __name__ == "__main__":
    main()
