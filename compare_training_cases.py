#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from accuracy_reports import infer_m_values, load_accuracy_report_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare uniformly sampled/standardized and fully preprocessed training reports."
    )
    parser.add_argument("--raw-report", type=Path)
    parser.add_argument("--uniform-checkpoint", type=Path)
    parser.add_argument("--uniform-report", type=Path, help="Training report from the uniform checkpoint run")
    parser.add_argument("--full-checkpoint", type=Path)
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
            "legend.fontsize": 12,
        }
    )

    if args.uniform_checkpoint or args.full_checkpoint:
        if not (args.uniform_checkpoint and args.full_checkpoint):
            parser.error("Both checkpoint arguments are required together")
        if args.uniform_report is None:
            parser.error("--uniform-report is required to plot the actual uniform training history")
        reference = load_report(args.preprocessed_report)
        raw = load_checkpoint_report(args.uniform_checkpoint, reference)
        pre = load_checkpoint_report(args.full_checkpoint, reference)
        raw["history"] = load_report(args.uniform_report)["history"]
        pre["history"] = reference["history"]
        plot_combined_comparison(
            raw, pre, "uniform sampling", args.preprocessed_label, "test",
            args.max_scatter_points, Path("final_plots/offline.png"), plt,
        )
        return
    if args.raw_report is None:
        parser.error("--raw-report is required when not comparing checkpoints")
    raw = load_report(args.raw_report)
    pre = load_report(args.preprocessed_report)
    uniform = load_report(args.uniform_report) if args.uniform_report else None

    if raw["test_x_raw"] is None or pre["test_x_raw"] is None or not np.array_equal(raw["test_x_raw"], pre["test_x_raw"]):
        raise ValueError("Test inputs differ. Retrain both workflows with the same velocity filter and seeded split.")
    for split in ("validation", "test"):
        if not np.array_equal(raw["split_predictions"][split][0], pre["split_predictions"][split][0]):
            raise ValueError(f"{split} targets differ. Retrain both workflows on the same source data.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_plots_dir = Path("./final_plots")
    final_plots_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(
        raw,
        pre,
        args.raw_label,
        args.preprocessed_label,
        args.out_dir / "learning_curves_comparison.png",
        plt,
        uniform=uniform,
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
        uniform=uniform,
    )
    plot_inferred_m_comparison(
        raw,
        pre,
        args.raw_label,
        args.preprocessed_label,
        args.out_dir / "inferred_m_comparison.png",
        plt,
        uniform=uniform,
    )
    plot_combined_comparison(
        raw,
        pre,
        args.raw_label,
        args.preprocessed_label,
        args.scatter_split,
        args.max_scatter_points,
        final_plots_dir / "offline.png",
        plt,
        uniform=uniform,
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


def load_checkpoint_report(path: Path, reference: dict[str, object]) -> dict[str, object]:
    """Evaluate checkpoint weights on the same saved test inputs for both cases."""
    import torch
    from friction_emulator.friction_emulator import FrictionMLP
    from preprocessing import NormalizationConfig
    from train import predict_raw

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = FrictionMLP(**{key: int(checkpoint[key]) for key in ("in_dim", "h1", "h2", "out_dim")})
    model.load_state_dict(checkpoint["state_dict"])
    mode = checkpoint["normalization"]
    # Standardization uses the stored means/scales without a nonlinear transform.
    if mode == "standard":
        mode = "raw"
    if mode not in {"raw", "sqrt", "log", "mixed"}:
        raise ValueError(f"Unsupported normalization: {mode}")
    norm = NormalizationConfig(mode=mode, **{
        key: checkpoint[key] for key in ("x_mean", "x_std", "y_mean", "y_std", "x_floor", "y_floor")
    })
    x = reference["test_x_raw"]
    prediction = predict_raw(model, x, norm, torch.device("cpu"), 4096)
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite predictions from {path}")
    truth = reference["split_predictions"]["test"][0]
    rmse = np.sqrt(np.mean((prediction.astype(np.float64) - truth) ** 2))
    print(f"{path}: test samples={len(x)}, RMSE={rmse:.6e}")
    return {"path": path, "history": [], "test_x_raw": x,
            "split_predictions": {"test": (truth, prediction)}}


def plot_learning_curves(
    raw: dict[str, object],
    preprocessed: dict[str, object],
    raw_label: str,
    preprocessed_label: str,
    out_path: Path,
    plt,
    uniform: dict[str, object] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    styles = [
        (raw, raw_label, "#1f77b4", "-"),
        (preprocessed, preprocessed_label, "#d62728", "-"),
    ]
    if uniform is not None:
        styles.insert(1, (uniform, "uniform sampling", "#d4a000", "-"))
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
    ax.legend(frameon=False, fontsize=12)
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
    uniform: dict[str, object] | None = None,
) -> None:
    reports = [
        (raw, raw_label, "#1f77b4"),
        (preprocessed, preprocessed_label, "#d62728"),
    ]
    if uniform is not None:
        reports.insert(1, (uniform, "uniform sampling", "#d4a000"))
    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    all_values = []
    for report, _, _ in reports:
        split_predictions = report["split_predictions"]
        y_true, y_pred = split_predictions[split]
        all_values.append(y_true.reshape(-1))
        all_values.append(y_pred.reshape(-1))
    all_flat = np.concatenate(all_values)
    lo = float(np.nanmin(all_flat))
    hi = float(np.nanmax(all_flat))
    scatter_lo = min(0.0, lo)
    scatter_hi = hi * 1.03
    if scatter_hi <= 3.0e12:
        scatter_ticks = np.array([0.0, 1.0e12, 2.0e12])
        scatter_hi = max(scatter_hi, 2.0e12)
    else:
        scatter_ticks = np.linspace(scatter_lo, scatter_hi, 4)

    for report, label, color in reports:
        split_predictions = report["split_predictions"]
        y_true, y_pred = split_predictions[split]
        true_flat = y_true.reshape(-1)
        pred_flat = y_pred.reshape(-1)
        idx = scatter_indices(len(true_flat), max_points, seed=42)
        ax.scatter(
            true_flat[idx],
            pred_flat[idx],
            s=9,
            alpha=0.34,
            color=color,
            linewidths=0,
            label=label,
        )
    ax.plot([scatter_lo, scatter_hi], [scatter_lo, scatter_hi], "k--", linewidth=0.8)
    ax.set_xlim(scatter_lo, scatter_hi)
    ax.set_ylim(scatter_lo, scatter_hi)
    ax.set_box_aspect(1)
    ax.set_xticks(scatter_ticks)
    ax.set_yticks(scatter_ticks)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
    ax.tick_params(axis="both", pad=2)
    ax.set_xlabel(r"True $\alpha^2$")
    ax.set_ylabel(r"Predicted $\alpha^2$")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=12)

    fig.tight_layout()
    fig.canvas.draw()
    offset_text = ax.xaxis.get_offset_text()
    offset_text.set_visible(False)
    ax.text(
        1.0,
        -0.075,
        r"$\times 10^{12}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=18,
        clip_on=False,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_inferred_m_comparison(
    raw: dict[str, object],
    preprocessed: dict[str, object],
    raw_label: str,
    preprocessed_label: str,
    out_path: Path,
    plt,
    uniform: dict[str, object] | None = None,
) -> None:
    reports = [
        (raw, raw_label, "#1f77b4"),
        (preprocessed, preprocessed_label, "#d62728"),
    ]
    if uniform is not None:
        reports.insert(1, (uniform, "uniform sampling", "#d4a000"))
    x_range = (2.5, 3.5)
    bins = 100

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for report, label, color in reports:
        split_predictions = report["split_predictions"]
        _, test_pred = split_predictions["test"]
        m_values = infer_m_values(report["test_x_raw"], test_pred)
        m_values = m_values[np.isfinite(m_values)]
        m_values = m_values[(m_values >= x_range[0]) & (m_values <= x_range[1])]
        if len(m_values) == 0:
            continue

        density, edges = np.histogram(m_values, bins=bins, range=x_range, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, density, color=color, linewidth=1.5, label=label)

    ax.set_xlim(*x_range)
    ax.set_xlabel(r"Inferred Weertman exponent ($m$)")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False, fontsize=12,loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined_comparison(
    raw: dict[str, object],
    preprocessed: dict[str, object],
    raw_label: str,
    preprocessed_label: str,
    split: str,
    max_points: int,
    out_path: Path,
    plt,
    uniform: dict[str, object] | None = None,
) -> None:
    blue = "#1f77b4"
    red = "#d62728"
    reports = [(raw, raw_label, blue), (preprocessed, preprocessed_label, red)]
    if uniform is not None:
        reports.insert(1, (uniform, "uniform sampling", "#d4a000"))
    figure_width_cm = 14.0
    figure_height_cm = figure_width_cm * (5.94 / 6.3)
    fig = plt.figure(
        figsize=(figure_width_cm / 2.54, figure_height_cm / 2.54)
    )
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(0.82, 1.18),
        left=0.12,
        right=0.985,
        bottom=0.10,
        top=0.95,
        hspace=0.22,
        wspace=0.30,
    )
    learning_ax = fig.add_subplot(grid[0, :])
    scatter_ax = fig.add_subplot(grid[1, 0])
    inferred_ax = fig.add_subplot(grid[1, 1])
    axes = (learning_ax, scatter_ax, inferred_ax)

    for report, label, color in reports:
        history = report["history"]
        epochs = np.asarray([item["epoch"] for item in history], dtype=np.float64)
        train_rmse = np.asarray([item["train_rmse"] for item in history], dtype=np.float64)
        val_rmse = np.asarray([item["val_rmse"] for item in history], dtype=np.float64)
        learning_ax.plot(
            epochs, train_rmse, color=color, linewidth=1.4, label=f"{label} train"
        )
        learning_ax.plot(
            epochs,
            val_rmse,
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=f"{label} validation",
        )
    learning_ax.set_yscale("log")
    learning_ax.set_xlabel("Epoch")
    learning_ax.set_ylabel(r"RMSE of $\alpha^2$")
    learning_ax.grid(alpha=0.25)
    learning_ax.legend(
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper right",
        handlelength=2.4,
        columnspacing=1.0,
    )

    all_values = []
    for report, _, _ in reports:
        y_true, y_pred = report["split_predictions"][split]
        all_values.extend((y_true.reshape(-1), y_pred.reshape(-1)))
    all_flat = np.concatenate(all_values)
    scatter_lo = min(0.0, float(np.nanmin(all_flat)))
    scatter_hi = float(np.nanmax(all_flat)) * 1.03
    if scatter_hi <= 3.0e12:
        scatter_ticks = np.array([0.0, 1.0e12, 2.0e12])
        scatter_hi = max(scatter_hi, 2.0e12)
    else:
        scatter_ticks = np.linspace(scatter_lo, scatter_hi, 4)

    for report, label, color in reports:
        y_true, y_pred = report["split_predictions"][split]
        true_flat = y_true.reshape(-1)
        pred_flat = y_pred.reshape(-1)
        idx = scatter_indices(len(true_flat), max_points, seed=42)
        scatter_ax.scatter(
            true_flat[idx],
            pred_flat[idx],
            s=7,
            alpha=0.34,
            color=color,
            linewidths=0,
            label=label,
        )
    scatter_ax.plot(
        [scatter_lo, scatter_hi],
        [scatter_lo, scatter_hi],
        "k--",
        linewidth=0.8,
    )
    scatter_ax.set_xlim(scatter_lo, scatter_hi)
    scatter_ax.set_ylim(scatter_lo, scatter_hi)
    scatter_ax.set_box_aspect(0.85)
    scatter_ax.set_xticks(scatter_ticks)
    scatter_ax.set_yticks(scatter_ticks)
    scatter_ax.ticklabel_format(
        axis="both", style="sci", scilimits=(0, 0), useMathText=True
    )
    scatter_ax.xaxis.get_offset_text().set_fontsize(7)
    scatter_ax.yaxis.get_offset_text().set_fontsize(7)
    scatter_ax.set_xlabel(r"True $\alpha^2$")
    scatter_ax.set_ylabel(r"Predicted $\alpha^2$")
    scatter_ax.legend(frameon=False, fontsize=7, loc="upper left")

    x_range = (2.5, 3.5)
    for report, label, color in reports:
        _, test_pred = report["split_predictions"]["test"]
        m_values = infer_m_values(report["test_x_raw"], test_pred)
        m_values = m_values[np.isfinite(m_values)]
        m_values = m_values[(m_values >= x_range[0]) & (m_values <= x_range[1])]
        if len(m_values) == 0:
            continue
        density, edges = np.histogram(m_values, bins=100, range=x_range, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        inferred_ax.plot(centers, density, color=color, linewidth=1.6, label=label)
    inferred_ax.set_xlim(2.7, 3.3)
    inferred_ax.set_box_aspect(0.85)
    inferred_ax.set_xlabel(r"Inferred Weertman exponent ($m$)")
    inferred_ax.set_ylabel("Probability density")
    for index, (_, label, color) in enumerate(reports):
        y = 0.95 - 0.13 * index
        inferred_ax.plot(
            [0.04, 0.17], [y, y], transform=inferred_ax.transAxes,
            color=color, linewidth=1.6, clip_on=False,
        )
        inferred_ax.text(
            0.04, y - 0.05, label, transform=inferred_ax.transAxes,
            fontsize=7, ha="left", va="top",
        )

    for ax in axes:
        ax.minorticks_off()
        ax.tick_params(axis="both", labelsize=7, width=0.5, length=2.5)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def scatter_indices(n_points: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or n_points <= max_points:
        return np.arange(n_points)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_points, size=max_points, replace=False))


if __name__ == "__main__":
    main()
