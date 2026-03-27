#!/usr/bin/env python3
"""
Print per-file vmag distributions for Weertman CSV datasets.

Expected file layout:
  folder/weertman_rank_0.csv
  folder/weertman_rank_1.csv
  ...

Each valid row is expected to contain:
  C2, vmag, alpha2
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List

import numpy as np


def _is_header_line(row: List[str]) -> bool:
    joined = ",".join(row)
    for ch in joined:
        if ch.isalpha() and ch not in ("e", "E"):
            return True
    return False


def parse_vmag(row: List[str]) -> float | None:
    if not row or _is_header_line(row):
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

    if len(items) != 3:
        return None

    vmag = items[1]
    if not math.isfinite(vmag):
        return None
    return vmag


def format_edges(edges: np.ndarray) -> str:
    return ", ".join(f"{edge:.6e}" for edge in edges)


def append_file_summary(lines: List[str], path: Path, vmag: np.ndarray, bins: int) -> None:
    lines.append(f"\nFile: {path}")
    lines.append(f"Rows: {len(vmag)}")

    if len(vmag) == 0:
        lines.append("No valid rows found.")
        return

    lines.append(f"Min:    {np.min(vmag):.6e}")
    lines.append(f"P1:     {np.percentile(vmag, 1):.6e}")
    lines.append(f"P5:     {np.percentile(vmag, 5):.6e}")
    lines.append(f"P50:    {np.percentile(vmag, 50):.6e}")
    lines.append(f"P95:    {np.percentile(vmag, 95):.6e}")
    lines.append(f"P99:    {np.percentile(vmag, 99):.6e}")
    lines.append(f"Max:    {np.max(vmag):.6e}")
    lines.append(f"Mean:   {np.mean(vmag):.6e}")
    lines.append(f"Std:    {np.std(vmag):.6e}")
    lines.append(f"Zeros:  {np.count_nonzero(vmag == 0.0)}")
    lines.append(f"Negs:   {np.count_nonzero(vmag < 0.0)}")

    linear_counts, linear_edges = np.histogram(vmag, bins=bins)
    lines.append("\nLinear-bin histogram:")
    lines.append(f"Edges:  {format_edges(linear_edges)}")
    for i, count in enumerate(linear_counts):
        lines.append(
            f"  [{linear_edges[i]:.6e}, {linear_edges[i + 1]:.6e})"
            f" -> {count}"
        )

    positive_vmag = vmag[vmag > 0.0]
    if len(positive_vmag) == 0:
        lines.append("\nLog-bin histogram: skipped because there are no positive vmag values.")
        return

    log_edges = np.logspace(
        np.log10(np.min(positive_vmag)),
        np.log10(np.max(positive_vmag)),
        num=bins + 1,
    )
    log_counts, _ = np.histogram(positive_vmag, bins=log_edges)
    lines.append("\nLog-bin histogram (positive vmag only):")
    lines.append(f"Edges:  {format_edges(log_edges)}")
    for i, count in enumerate(log_counts):
        right_bracket = "]" if i == len(log_counts) - 1 else ")"
        lines.append(
            f"  [{log_edges[i]:.6e}, {log_edges[i + 1]:.6e}{right_bracket}"
            f" -> {count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print per-file vmag distributions.")
    parser.add_argument("--folder", type=str, default="./data")
    parser.add_argument("--pattern", type=str, default="weertman_rank_*.csv")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output", type=str, default="vmag_distribution.txt")
    args = parser.parse_args()

    folder = Path(args.folder)
    paths = sorted(folder.glob(args.pattern))

    if not paths:
        raise RuntimeError(f"No files matched {args.pattern} in {folder}")

    lines: List[str] = []
    lines.append(f"Folder: {folder}")
    lines.append(f"Files:  {len(paths)}")
    lines.append(f"Bins:   {args.bins}")

    for path in paths:
        values = []
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                vmag = parse_vmag(row)
                if vmag is not None:
                    values.append(vmag)

        append_file_summary(lines, path, np.asarray(values, dtype=np.float64), args.bins)

    output_path = Path(args.output)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote vmag distribution report to {output_path}")


if __name__ == "__main__":
    main()
