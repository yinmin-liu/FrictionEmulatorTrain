#!/usr/bin/env python3
from __future__ import annotations

import argparse

from accuracy_reports import load_accuracy_report_data, save_accuracy_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training accuracy plots from a saved report-data artifact."
    )
    parser.add_argument("--report-data", type=str, default="./friction_emulator/training_report_data.npz")
    parser.add_argument("--plots-dir", type=str, default="./plots")
    parser.add_argument("--filename-prefix", type=str, default="")
    args = parser.parse_args()

    split_predictions, history_by_seed, best_seed, test_x_raw = (
        load_accuracy_report_data(args.report_data)
    )
    save_accuracy_outputs(
        args.plots_dir,
        split_predictions,
        history_by_seed,
        best_seed,
        test_x_raw,
        filename_prefix=args.filename_prefix,
    )


if __name__ == "__main__":
    main()
