#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

from preprocessing import (
    RAW_X_OFFSET,
    RAW_X_SCALE,
    RAW_Y_OFFSET,
    RAW_Y_SCALE,
    log_transform,
    normalize,
    sqrt_transform,
    target_balanced_sample_indices,
)

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


def compute_log_normalized(data: np.ndarray, floors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logged = log_transform(data, floors)
    mean = logged.mean(axis=0)
    std = logged.std(axis=0)
    std[std < 1e-12] = 1.0
    return normalize(logged, mean, std), mean, std


def compute_sqrt_normalized(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rooted = sqrt_transform(data)
    mean = rooted.mean(axis=0)
    std = rooted.std(axis=0)
    std[std < 1e-12] = 1.0
    return normalize(rooted, mean, std), mean, std


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
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 640
    margin_x = 80
    top = 110
    bottom = 125
    gap = 70
    plot_w = (width - 2 * margin_x - 2 * gap) // 3
    plot_h = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    title_font = ImageFont.truetype(str(font_path), 32) if font_path.exists() else ImageFont.load_default()
    label_font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
    tick_font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    bar_color = (66, 145, 194)
    grid_color = (220, 220, 220)
    axis_color = (20, 20, 20)

    def text_center(x: float, y: float, text: str, font) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y), text, fill=axis_color, font=font)

    text_center(width / 2, 24, title, title_font)

    for i in range(3):
        values = data[:, i]
        values = values[np.isfinite(values)]
        counts, edges = np.histogram(values, bins=bins)
        log_counts = np.log10(np.maximum(counts, 1))
        y_max = max(float(log_counts.max()), 1.0)

        left = margin_x + i * (plot_w + gap)
        right = left + plot_w
        bottom_y = top + plot_h

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = bottom_y - frac * plot_h
            draw.line((left, y, right, y), fill=grid_color, width=1)
        for frac in (0.0, 0.5, 1.0):
            x = left + frac * plot_w
            draw.line((x, top, x, bottom_y), fill=grid_color, width=1)

        draw.rectangle((left, top, right, bottom_y), outline=axis_color, width=3)

        for j, count in enumerate(counts):
            if count <= 0:
                continue
            x0 = left + j * plot_w / bins
            x1 = left + (j + 1) * plot_w / bins
            y0 = bottom_y - (math.log10(count) / y_max) * plot_h
            draw.rectangle((x0, y0, max(x0 + 1, x1), bottom_y), fill=bar_color)

        xmin = float(edges[0])
        xmax = float(edges[-1])
        text_center(left, bottom_y + 12, f"{xmin:.2g}", tick_font)
        text_center((left + right) / 2, bottom_y + 12, f"{0.5 * (xmin + xmax):.2g}", tick_font)
        text_center(right, bottom_y + 12, f"{xmax:.2g}", tick_font)
        draw.text((left - 55, top - 8), f"1e{int(round(y_max))}", fill=axis_color, font=tick_font)
        draw.text((left - 55, bottom_y - 18), "1", fill=axis_color, font=tick_font)
        text_center((left + right) / 2, bottom_y + 48, xlabels[i], label_font)

    draw.text((18, top + plot_h / 2 - 20), "count (log)", fill=axis_color, font=label_font)
    path = path.with_suffix(".pdf")
    image.save(path, "PDF", resolution=300.0)
    print(f"Saved {path.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw and transformed data distributions.")
    parser.add_argument("--folder", type=str, default="./data")
    parser.add_argument("--n-ranks", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="./data_distribution_plots")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--balanced-target-bins", type=int, default=20)
    parser.add_argument("--balanced-samples", type=int, default=None)
    parser.add_argument("--balanced-power", type=float, default=0.5)
    parser.add_argument("--balanced-max-weight", type=float, default=20.0)
    parser.add_argument("--balanced-seed", type=int, default=42)
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
    sqrt_data = sqrt_transform(data)
    sqrt_normalized, sqrt_mean, sqrt_std = compute_sqrt_normalized(data)
    balanced_idx = target_balanced_sample_indices(
        sqrt_data[:, 2],
        n_bins=args.balanced_target_bins,
        n_samples=args.balanced_samples or len(data),
        balance_power=args.balanced_power,
        max_weight=args.balanced_max_weight,
        seed=args.balanced_seed,
    )
    raw_balanced = raw_normalized[balanced_idx]
    sqrt_balanced = sqrt_normalized[balanced_idx]

    print_summary("Raw variables", data)
    print_summary("Positive raw-scaled variables", raw_normalized)
    print_summary("Log variables", log_data)
    print_summary("Log-normalized variables", log_normalized)
    print_summary("Square-root variables", sqrt_data)
    print_summary("Square-root-normalized variables", sqrt_normalized)
    print_summary("Raw-normalized target-balanced samples", raw_balanced)
    print_summary("Square-root-normalized target-balanced samples", sqrt_balanced)
    print("\nLog normalization constants from loaded data")
    print("--------------------------------------------")
    for variable, mean, std, floor in zip(VARIABLES, log_mean, log_std, floors):
        print(f"{variable:<7} mean={mean:.17g} std={std:.17g} floor={floor:.17g}")
    print("\nSquare-root normalization constants from loaded data")
    print("----------------------------------------------------")
    for variable, mean, std in zip(VARIABLES, sqrt_mean, sqrt_std):
        print(f"{variable:<7} mean={mean:.17g} std={std:.17g}")
    print("\nTarget-balanced sampling")
    print("------------------------")
    print(f"target variable: sqrt(alpha2)")
    print(f"target bins: {args.balanced_target_bins}")
    print(f"balance power: {args.balanced_power}")
    print(f"max relative sample weight: {args.balanced_max_weight}")
    print(f"sampled training examples: {len(balanced_idx)}")

    plot_hist_grid(
        raw_normalized,
        "Distributions After Positive Raw Scaling",
        out_dir / "raw_normalized_distributions.pdf",
        xlabels=("C2 / 9.05e6", "|ub| / 2.08e-5", "alpha2 / 2.09e11"),
        bins=args.bins,
    )
    plot_hist_grid(
        log_normalized,
        "Distributions After Log Normalization",
        out_dir / "log_normalized_distributions.pdf",
        xlabels=(
            "(ln C2 - mean) / std",
            "(ln |ub| - mean) / std",
            "(ln alpha2 - mean) / std",
        ),
        bins=args.bins,
    )
    plot_hist_grid(
        sqrt_normalized,
        "Distributions After Square-Root Normalization",
        out_dir / "sqrt_normalized_distributions.pdf",
        xlabels=(
            "(sqrt C2 - mean) / std",
            "(sqrt |ub| - mean) / std",
            "(sqrt alpha2 - mean) / std",
        ),
        bins=args.bins,
    )
    plot_hist_grid(
        raw_balanced,
        "Raw-Scaled Distributions After Target-Balanced Sampling",
        out_dir / "raw_normalized_balanced_sampling_distributions.pdf",
        xlabels=(
            "C2 / 9.05e6",
            "|ub| / 2.08e-5",
            "alpha2 / 2.09e11",
        ),
        bins=args.bins,
    )
    plot_hist_grid(
        sqrt_balanced,
        "Square-Root Normalized Distributions After Target-Balanced Sampling",
        out_dir / "sqrt_normalized_balanced_sampling_distributions.pdf",
        xlabels=(
            "(sqrt C2 - mean) / std",
            "(sqrt |ub| - mean) / std",
            "(sqrt alpha2 - mean) / std",
        ),
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
