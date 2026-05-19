#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np


RAW_X_MEAN = np.array([9.05e6, 2.08e-5], dtype=np.float64)
RAW_X_STD = np.array([6.61e6, 4.67e-5], dtype=np.float64)
RAW_Y_MEAN = np.array([2.09e11], dtype=np.float64)
RAW_Y_STD = np.array([1.18e12], dtype=np.float64)

VARIABLES = ("C2", "vmag", "alpha2")


def is_header_line(row: List[str]) -> bool:
    joined = ",".join(row)
    for ch in joined:
        if ch.isalpha() and ch not in ("e", "E"):
            return True
    return False


def parse_row(row: List[str]) -> Tuple[float, float, float] | None:
    if not row or is_header_line(row):
        return None

    values = []
    for item in row:
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            return None

    if len(values) != 3:
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values[0], values[1], values[2]


def load_rank_data(folder: str, n_ranks: int) -> np.ndarray:
    rows = []
    for rank in range(n_ranks):
        path = Path(folder) / f"weertman_rank_{rank}.csv"
        if not path.exists():
            print(f"Warning: could not open {path}, skipping.")
            continue

        count = 0
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                parsed = parse_row(row)
                if parsed is not None:
                    rows.append(parsed)
                    count += 1
        print(f"Loaded {count} rows from {path}")

    if not rows:
        raise RuntimeError("No data loaded.")
    data = np.asarray(rows, dtype=np.float64)
    print(f"Total loaded: {len(data)} samples")
    return data


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values - mean) / std


def log_transform(values: np.ndarray, floors: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(values, floors))


def compute_log_normalized(data: np.ndarray, floors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logged = log_transform(data, floors)
    mean = logged.mean(axis=0)
    std = logged.std(axis=0)
    std[std < 1e-12] = 1.0
    return standardize(logged, mean, std), mean, std


def print_summary(name: str, data: np.ndarray) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for i, variable in enumerate(VARIABLES):
        col = data[:, i]
        finite = col[np.isfinite(col)]
        if len(finite) == 0:
            print(f"{variable}: no finite values")
            continue
        q = np.percentile(finite, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        print(
            f"{variable:<7}"
            f" min={q[0]:.6e}"
            f" p1={q[1]:.6e}"
            f" p5={q[2]:.6e}"
            f" p50={q[4]:.6e}"
            f" p95={q[6]:.6e}"
            f" p99={q[7]:.6e}"
            f" max={q[8]:.6e}"
        )


def plot_hist_grid(
    data: np.ndarray,
    title: str,
    path: Path,
    xlabels: Tuple[str, str, str] = VARIABLES,
    bins: int = 80,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes):
        values = data[:, i]
        values = values[np.isfinite(values)]
        ax.hist(values, bins=bins, alpha=0.85)
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_rank_counts(data: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(
        0.5,
        0.5,
        f"Loaded {len(data):,} samples",
        ha="center",
        va="center",
        fontsize=20,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw, raw-normalized, and log-normalized data distributions.")
    parser.add_argument("--folder", type=str, default="./data")
    parser.add_argument("--n-ranks", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="./data_distribution_plots")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--log-c2-floor", type=float, default=1e-30)
    parser.add_argument("--log-v-floor", type=float, default=1e-12)
    parser.add_argument("--log-alpha2-floor", type=float, default=1e-30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_rank_data(args.folder, args.n_ranks)
    floors = np.array([args.log_c2_floor, args.log_v_floor, args.log_alpha2_floor], dtype=np.float64)
    raw_mean = np.concatenate([RAW_X_MEAN, RAW_Y_MEAN])
    raw_std = np.concatenate([RAW_X_STD, RAW_Y_STD])
    raw_normalized = standardize(data, raw_mean, raw_std)
    log_data = log_transform(data, floors)
    log_normalized, log_mean, log_std = compute_log_normalized(data, floors)

    print_summary("Raw variables", data)
    print_summary("Raw-normalized variables", raw_normalized)
    print_summary("Log variables", log_data)
    print_summary("Log-normalized variables", log_normalized)
    print("\nLog normalization constants from loaded data")
    print("--------------------------------------------")
    for variable, mean, std, floor in zip(VARIABLES, log_mean, log_std, floors):
        print(f"{variable:<7} mean={mean:.17g} std={std:.17g} floor={floor:.17g}")

    plot_hist_grid(
        data,
        f"Raw Variable Distributions ({args.n_ranks} ranks)",
        out_dir / "raw_distributions.png",
        bins=args.bins,
    )
    plot_hist_grid(
        np.log10(np.maximum(data, floors)),
        f"Log10 Raw Variable Distributions ({args.n_ranks} ranks)",
        out_dir / "log10_raw_distributions.png",
        xlabels=("log10(C2)", "log10(vmag)", "log10(alpha2)"),
        bins=args.bins,
    )
    plot_hist_grid(
        raw_normalized,
        "Distributions After Current Raw Normalization",
        out_dir / "raw_normalized_distributions.png",
        xlabels=("C2 raw-normalized", "vmag raw-normalized", "alpha2 raw-normalized"),
        bins=args.bins,
    )
    plot_hist_grid(
        log_normalized,
        "Distributions After Log Normalization",
        out_dir / "log_normalized_distributions.png",
        xlabels=("C2 log-normalized", "vmag log-normalized", "alpha2 log-normalized"),
        bins=args.bins,
    )
    plot_rank_counts(data, out_dir / "sample_count.png")


if __name__ == "__main__":
    main()
