#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np


RAW_X_OFFSET = np.array([0.0, 0.0], dtype=np.float64)
RAW_X_SCALE = np.array([9.05e6, 2.08e-5], dtype=np.float64)
RAW_Y_OFFSET = np.array([0.0], dtype=np.float64)
RAW_Y_SCALE = np.array([2.09e11], dtype=np.float64)

VARIABLES = ("C2", "vmag", "alpha2")
RAW_LABELS = (r"$C^2$", r"$|u_b|$", r"$\alpha^2$")


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


def normalize(values: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - offset) / scale


def log_transform(values: np.ndarray, floors: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(values, floors))


def compute_log_normalized(data: np.ndarray, floors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logged = log_transform(data, floors)
    mean = logged.mean(axis=0)
    std = logged.std(axis=0)
    std[std < 1e-12] = 1.0
    return normalize(logged, mean, std), mean, std


def compute_mixed_normalized(data: np.ndarray, floors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mixed = data.copy()
    mixed[:, 1] = np.log(np.maximum(data[:, 1], floors[1]))
    mixed[:, 2] = np.log(np.maximum(data[:, 2], floors[2]))
    mean = np.array([RAW_X_OFFSET[0], mixed[:, 1].mean(), mixed[:, 2].mean()], dtype=np.float64)
    scale = np.array([RAW_X_SCALE[0], mixed[:, 1].std(), mixed[:, 2].std()], dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    return normalize(mixed, mean, scale), mean, scale


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
    raw_offset = np.concatenate([RAW_X_OFFSET, RAW_Y_OFFSET])
    raw_scale = np.concatenate([RAW_X_SCALE, RAW_Y_SCALE])
    raw_normalized = normalize(data, raw_offset, raw_scale)
    log_data = log_transform(data, floors)
    log_normalized, log_mean, log_std = compute_log_normalized(data, floors)
    mixed_normalized, mixed_mean, mixed_std = compute_mixed_normalized(data, floors)

    print_summary("Raw variables", data)
    print_summary("Positive raw-scaled variables", raw_normalized)
    print_summary("Log variables", log_data)
    print_summary("Log-normalized variables", log_normalized)
    print_summary("Mixed-normalized variables", mixed_normalized)
    print("\nLog normalization constants from loaded data")
    print("--------------------------------------------")
    for variable, mean, std, floor in zip(VARIABLES, log_mean, log_std, floors):
        print(f"{variable:<7} mean={mean:.17g} std={std:.17g} floor={floor:.17g}")
    print("\nMixed normalization constants from loaded data")
    print("----------------------------------------------")
    for variable, mean, std in zip(VARIABLES, mixed_mean, mixed_std):
        print(f"{variable:<7} mean={mean:.17g} std={std:.17g}")

    plot_hist_grid(
        data,
        "Raw Variable Distributions",
        out_dir / "raw_distributions.png",
        xlabels=RAW_LABELS,
        bins=args.bins,
    )
    plot_hist_grid(
        np.log10(np.maximum(data, floors)),
        f"Log10 Raw Variable Distributions ({args.n_ranks} ranks)",
        out_dir / "log10_raw_distributions.png",
        xlabels=(r"$\log_{10}(C^2)$", r"$\log_{10}(|u_b|)$", r"$\log_{10}(\alpha^2)$"),
        bins=args.bins,
    )
    plot_hist_grid(
        raw_normalized,
        "Distributions After Positive Raw Scaling",
        out_dir / "raw_normalized_distributions.png",
        xlabels=(r"$C^2 / 9.05e6$", r"$|u_b| / 2.08e-5$", r"$\alpha^2 / 2.09e11$"),
        bins=args.bins,
    )
    plot_hist_grid(
        log_normalized,
        "Distributions After Log Normalization",
        out_dir / "log_normalized_distributions.png",
        xlabels=(r"$C^2$ log-normalized", r"$|u_b|$ log-normalized", r"$\alpha^2$ log-normalized"),
        bins=args.bins,
    )
    plot_hist_grid(
        mixed_normalized,
        "Distributions After Mixed Normalization",
        out_dir / "mixed_normalized_distributions.png",
        xlabels=(r"$C^2 / 9.05e6$", r"$\log(|u_b|)$ normalized", r"$\log(\alpha^2)$ normalized"),
        bins=args.bins,
    )
    plot_rank_counts(data, out_dir / "sample_count.png")


if __name__ == "__main__":
    main()
