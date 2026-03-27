#!/usr/bin/env python3
"""
Load all Weertman CSV datasets and print the global min/max of each variable.

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


def parse_row(row: List[str]) -> list[float] | None:
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

    if not all(math.isfinite(v) for v in items):
        return None

    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Print min/max of each variable across all data files.")
    parser.add_argument("--folder", type=str, default="./data")
    parser.add_argument("--pattern", type=str, default="weertman_rank_*.csv")
    args = parser.parse_args()

    paths = sorted(Path(args.folder).glob(args.pattern))
    if not paths:
        raise RuntimeError(f"No files matched {args.pattern} in {args.folder}")

    min_vals = [float("inf")] * 3
    max_vals = [float("-inf")] * 3
    names = ["C2", "vmag", "alpha2"]
    total_rows = 0
    all_values = [[], [], []]

    for path in paths:
        with path.open("r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                values = parse_row(row)
                if values is None:
                    continue
                total_rows += 1
                for i, value in enumerate(values):
                    all_values[i].append(value)
                    if value < min_vals[i]:
                        min_vals[i] = value
                    if value > max_vals[i]:
                        max_vals[i] = value

    print(f"Folder: {args.folder}")
    print(f"Files loaded: {len(paths)}")
    print(f"Valid rows: {total_rows}")
    print("")
    for i, name in enumerate(names):
        array = np.asarray(all_values[i], dtype=np.float64)
        print(f"{name}:")
        print(f"  min = {min_vals[i]:.6e}")
        print(f"  max = {max_vals[i]:.6e}")
        print(f"  mean = {np.mean(array):.6e}")
        print(f"  std = {np.std(array):.6e}")


if __name__ == "__main__":
    main()
