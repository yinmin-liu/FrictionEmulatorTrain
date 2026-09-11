#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from accuracy_reports import load_accuracy_report_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare uniformly sampled/standardized and fully preprocessed training reports."
    )
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--preprocessed-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("./plots_comparison"))
    parser.add_argument("--raw-label", type=str, default="uniform sampling + standardization")
    parser.add_argument("--preprocessed-label", type=str, default="full preprocessing")
    parser.add_argument("--scatter-split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-scatter-points", type=int, default=12000)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
        }
    )

    raw = load_report(args.raw_report)
    pre = load_report(args.preprocessed_report)

    if raw["test_x_raw"] is None or pre["test_x_raw"] is None or not np.array_equal(raw["test_x_raw"], pre["test_x_raw"]):
        raise ValueError("Test inputs differ. Retrain both workflows with the same velocity filter and seeded split.")
    for split in ("validation", "test"):
        if not np.array_equal(raw["split_predictions"][split][0], pre["split_predictions"][split][0]):
            raise ValueError(f"{split} targets differ. Retrain both workflows on the same source data.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(
        raw,
        pre,
        args.raw_label,
        args.preprocessed_label,
        args.out_dir / "learning_curves_comparison.png",
        plt,
    )
    plot_prediction_scatter(
        raw,
        pre,
        args.raw_label,
        args.preprocessed_label,
        args.scatter_split,
        args.max_scatter_points,
        args.out_dir / "prediction_scatter_comparison.png",
        plt,
    )


def load_report(path: Path) -> dict[str, object]:
    split_predictions, history_by_seed, best_seed, test_x_raw = load_accuracy_report_data(str(path))
    history = history_by_seed[best_seed]
    return {
        "path": path,
        "split_predictions": split_predictions,
        "history": history,
        "best_seed": best_seed,
        "test_x_raw": test_x_raw,
    }


def plot_learning_curves(
    raw: dict[str, object],
    preprocessed: dict[str, object],
    raw_label: str,
    preprocessed_label: str,
    out_path: Path,
    plt,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    styles = [
        (raw, raw_label, "#1f77b4", "-"),
        (preprocessed, preprocessed_label, "#d62728", "-"),
    ]
    for report, label, color, linestyle in styles:
        history = report["history"]
        epochs = np.asarray([item["epoch"] for item in history], dtype=np.float64)
        train_rmse = np.asarray([item["train_rmse"] for item in history], dtype=np.float64)
        val_rmse = np.asarray([item["val_rmse"] for item in history], dtype=np.float64)
        ax.plot(epochs, train_rmse, color=color, linestyle=linestyle, linewidth=1.2, label=f"{label} train")
        ax.plot(epochs, val_rmse, color=color, linestyle="--", linewidth=1.2, label=f"{label} validation")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"RMSE of $\alpha^2$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_prediction_scatter(
    raw: dict[str, object],
    preprocessed: dict[str, object],
    raw_label: str,
    preprocessed_label: str,
    split: str,
    max_points: int,
    out_path: Path,
    plt,
) -> None:
    reports = [
        (raw, raw_label, "#1f77b4"),
        (preprocessed, preprocessed_label, "#d62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), sharex=True, sharey=True)

    all_values = []
    for report, _, _ in reports:
        split_predictions = report["split_predictions"]
        y_true, y_pred = split_predictions[split]
        all_values.append(y_true.reshape(-1))
        all_values.append(y_pred.reshape(-1))
    all_flat = np.concatenate(all_values)
    lo = float(np.nanmin(all_flat))
    hi = float(np.nanmax(all_flat))

    for ax, (report, label, color) in zip(axes, reports):
        split_predictions = report["split_predictions"]
        y_true, y_pred = split_predictions[split]
        true_flat = y_true.reshape(-1)
        pred_flat = y_pred.reshape(-1)
        idx = scatter_indices(len(true_flat), max_points, seed=42)
        ax.scatter(
            true_flat[idx],
            pred_flat[idx],
            s=6,
            alpha=0.35,
            color=color,
            linewidths=0,
        )
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel(r"True $\alpha^2$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"Predicted $\alpha^2$")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def scatter_indices(n_points: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or n_points <= max_points:
        return np.arange(n_points)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


if __name__ == "__main__":
    main()
