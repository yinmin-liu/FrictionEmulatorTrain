from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


LATEX_FIGSIZE = (3.27, 2.55)  # 8.3 cm wide
LATEX_SQUARE_FIGSIZE = (3.27, 3.0)


def fmt_sci(value: float) -> str:
    return f"{float(value):.6e}"


def fmt_percent(value: float) -> str:
    return f"{float(value):.6f}"


@torch.no_grad()
def predict_raw(
    model: nn.Module,
    x_raw: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds = []

    x_mean_t = torch.as_tensor(x_mean, dtype=torch.float32, device=device)
    x_std_t = torch.as_tensor(x_std, dtype=torch.float32, device=device)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    for start in range(0, len(x_raw), batch_size):
        end = min(start + batch_size, len(x_raw))
        xb_raw = torch.as_tensor(x_raw[start:end], dtype=torch.float32, device=device)
        xb = (xb_raw - x_mean_t) / x_std_t
        pred_raw = model(xb) * y_std_t + y_mean_t
        preds.append(pred_raw.detach().cpu().numpy())

    return np.concatenate(preds, axis=0)


def compute_accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    error = y_pred - y_true
    abs_error = np.abs(error)
    rmse = float(np.sqrt(np.mean(error * error)))
    mae = float(np.mean(abs_error))
    rel_error = abs_error / (np.abs(y_true) + 1e-10)
    mape = float(np.mean(rel_error) * 100.0)
    max_rel_error = float(np.max(rel_error) * 100.0)
    ss_res = float(np.sum(error * error))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "mape_percent": mape,
        "max_rel_error_percent": max_rel_error,
        "r2": float(r2),
    }


def print_accuracy_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(
        f"{name:<10} RMSE={fmt_sci(metrics['rmse'])}"
        f"  MAE={fmt_sci(metrics['mae'])}"
        f"  MAPE={fmt_percent(metrics['mape_percent'])}%"
        f"  MaxRelErr={fmt_percent(metrics['max_rel_error_percent'])}%"
        f"  R2={metrics['r2']:.6f}"
    )


def infer_m_values(
    x_raw: np.ndarray,
    alpha2_pred: np.ndarray,
    denominator_tol: float = 1e-12,
) -> np.ndarray:
    m_values = infer_m_values_full(x_raw, alpha2_pred, denominator_tol)
    return m_values[np.isfinite(m_values)]


def infer_m_values_full(
    x_raw: np.ndarray,
    alpha2_pred: np.ndarray,
    denominator_tol: float = 1e-12,
) -> np.ndarray:
    c2 = x_raw[:, 0].reshape(-1)
    vmag = x_raw[:, 1].reshape(-1)
    alpha2 = alpha2_pred.reshape(-1)

    positive = (c2 > 0.0) & (vmag > 0.0) & (alpha2 > 0.0)
    m_values = np.full_like(alpha2, np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_vmag = np.log(vmag[positive])
        denominator = np.log(alpha2[positive] / c2[positive]) + log_vmag
        valid_denominator = np.abs(denominator) > denominator_tol
        inferred = np.full_like(log_vmag, np.nan, dtype=np.float64)
        inferred[valid_denominator] = log_vmag[valid_denominator] / denominator[valid_denominator]
        m_values[positive] = inferred

    return m_values


def print_m_summary(m_values: np.ndarray, total_count: int) -> None:
    valid_count = len(m_values)
    skipped_count = total_count - valid_count
    if valid_count == 0:
        print("Inferred m: no valid test predictions for m diagnostic")
        return
    print(
        "Inferred m "
        f"valid={valid_count}/{total_count}"
        f"  skipped={skipped_count}"
        f"  mean={fmt_sci(float(np.mean(m_values)))}"
        f"  median={fmt_sci(float(np.median(m_values)))}"
        f"  std={fmt_sci(float(np.std(m_values)))}"
        f"  min={fmt_sci(float(np.min(m_values)))}"
        f"  max={fmt_sci(float(np.max(m_values)))}"
    )


def print_m_outlier_details(
    x_raw: np.ndarray,
    alpha2_true: np.ndarray,
    alpha2_pred: np.ndarray,
    m_values_full: np.ndarray,
    low: float = 2.0,
    high: float = 4.0,
) -> None:
    true_flat = alpha2_true.reshape(-1)
    pred_flat = alpha2_pred.reshape(-1)
    outlier_mask = np.isfinite(m_values_full) & ((m_values_full < low) | (m_values_full > high))
    outlier_indices = np.flatnonzero(outlier_mask)

    print(
        f"Inferred m outliers outside [{low:g}, {high:g}]: "
        f"{len(outlier_indices)}/{len(m_values_full)}"
    )
    if len(outlier_indices) == 0:
        return

    print(
        "  index"
        "  C2"
        "  |ub|"
        "  true_alpha2"
        "  pred_alpha2"
        "  rel_err_percent"
        "  inferred_m"
    )
    for idx in outlier_indices:
        true_val = float(true_flat[idx])
        pred_val = float(pred_flat[idx])
        rel_err = abs(pred_val - true_val) / (abs(true_val) + 1e-10) * 100.0
        print(
            f"  {idx:d}"
            f"  {float(x_raw[idx, 0]):.6e}"
            f"  {float(x_raw[idx, 1]):.6e}"
            f"  {true_val:.6e}"
            f"  {pred_val:.6e}"
            f"  {rel_err:.6f}"
            f"  {float(m_values_full[idx]):.6e}"
        )


def configure_matplotlib_for_latex(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )


def save_matplotlib_figure(fig, path: Path) -> None:
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {pdf_path}")


def _svg_scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return 0.5 * (dst_min + dst_max)
    return dst_min + (value - src_min) / (src_max - src_min) * (dst_max - dst_min)


def _write_svg(path: Path, width: int, height: int, body: List[str]) -> None:
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg) + "\n")


def save_accuracy_svgs(
    out_dir: Path,
    split_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    history_by_seed: Dict[int, List[Dict[str, float]]],
    best_seed: int,
    test_x_raw: np.ndarray,
) -> None:
    colors = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}
    width, height = 760, 620
    left, right, top, bottom = 90, 30, 50, 85
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_true = np.concatenate([yp[0].reshape(-1) for yp in split_predictions.values()])
    all_pred = np.concatenate([yp[1].reshape(-1) for yp in split_predictions.values()])
    lo = float(min(np.min(all_true), np.min(all_pred)))
    hi = float(max(np.max(all_true), np.max(all_pred)))
    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">Prediction Scatter</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#333" stroke-dasharray="5,5"/>',
    ]
    for name, (y_true, y_pred) in split_predictions.items():
        true_flat = y_true.reshape(-1)
        pred_flat = y_pred.reshape(-1)
        stride = max(1, len(true_flat) // 2500)
        for true_val, pred_val in zip(true_flat[::stride], pred_flat[::stride]):
            x = _svg_scale(float(true_val), lo, hi, left, left + plot_w)
            y = _svg_scale(float(pred_val), lo, hi, top + plot_h, top)
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{colors[name]}" opacity="0.45"/>')
    for i, name in enumerate(split_predictions):
        y = top + 24 + i * 22
        body.append(f'<circle cx="{left + 18}" cy="{y - 5}" r="5" fill="{colors[name]}"/>')
        body.append(f'<text x="{left + 32}" y="{y}" font-family="Arial" font-size="14">{name}</text>')
    body.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="15">True alpha^2</text>',
            f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="Arial" font-size="15">Predicted alpha^2</text>',
        ]
    )
    scatter_path = out_dir / "prediction_scatter.svg"
    _write_svg(scatter_path, width, height, body)

    test_true, test_pred = split_predictions["test"]
    test_error = (test_pred - test_true).reshape(-1)
    counts, edges = np.histogram(test_error, bins=60)
    max_count = max(int(np.max(counts)), 1)
    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">Test Error Histogram</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>',
    ]
    for count, x0, x1 in zip(counts, edges[:-1], edges[1:]):
        x = _svg_scale(float(x0), float(edges[0]), float(edges[-1]), left, left + plot_w)
        x_next = _svg_scale(float(x1), float(edges[0]), float(edges[-1]), left, left + plot_w)
        bar_h = _svg_scale(float(count), 0.0, float(max_count), 0.0, plot_h)
        body.append(
            f'<rect x="{x:.2f}" y="{top + plot_h - bar_h:.2f}" width="{max(x_next - x - 1, 1):.2f}" height="{bar_h:.2f}" fill="#4c78a8" opacity="0.85"/>'
        )
    zero_x = _svg_scale(0.0, float(edges[0]), float(edges[-1]), left, left + plot_w)
    body.append(f'<line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{top + plot_h}" stroke="#333" stroke-dasharray="5,5"/>')
    body.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="15">Prediction error, alpha^2_pred - alpha^2</text>',
            f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="Arial" font-size="15">Count</text>',
        ]
    )
    hist_path = out_dir / "test_error_histogram.svg"
    _write_svg(hist_path, width, height, body)

    history = history_by_seed[best_seed]
    epochs = np.array([item["epoch"] for item in history], dtype=np.float64)
    train_rmse = np.array([item["train_rmse"] for item in history], dtype=np.float64)
    val_rmse = np.array([item["val_rmse"] for item in history], dtype=np.float64)
    positive = np.concatenate([train_rmse[train_rmse > 0], val_rmse[val_rmse > 0]])
    y_min = float(np.min(positive)) if len(positive) else 1.0
    y_max = float(np.max(positive)) if len(positive) else 10.0
    log_y_min = math.log10(y_min)
    log_y_max = math.log10(y_max)

    def points(values: np.ndarray) -> str:
        out = []
        for epoch, value in zip(epochs, values):
            x = _svg_scale(float(epoch), float(np.min(epochs)), float(np.max(epochs)), left, left + plot_w)
            y = _svg_scale(math.log10(max(float(value), y_min)), log_y_min, log_y_max, top + plot_h, top)
            out.append(f"{x:.2f},{y:.2f}")
        return " ".join(out)

    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">Loss Curves (seed {best_seed})</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>',
        f'<polyline points="{points(train_rmse)}" fill="none" stroke="#1f77b4" stroke-width="2"/>',
        f'<polyline points="{points(val_rmse)}" fill="none" stroke="#ff7f0e" stroke-width="2"/>',
        f'<line x1="{left + 18}" y1="{top + 22}" x2="{left + 52}" y2="{top + 22}" stroke="#1f77b4" stroke-width="2"/>',
        f'<text x="{left + 62}" y="{top + 27}" font-family="Arial" font-size="14">train</text>',
        f'<line x1="{left + 18}" y1="{top + 44}" x2="{left + 52}" y2="{top + 44}" stroke="#ff7f0e" stroke-width="2"/>',
        f'<text x="{left + 62}" y="{top + 49}" font-family="Arial" font-size="14">validation</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="15">Epoch</text>',
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="Arial" font-size="15">RMSE of alpha^2 (log scale)</text>',
    ]
    loss_path = out_dir / "loss_curves.svg"
    _write_svg(loss_path, width, height, body)

    _, test_pred = split_predictions["test"]
    m_values = infer_m_values(test_x_raw, test_pred)
    print_m_summary(m_values, len(test_pred))
    if len(m_values) > 0:
        counts, edges = np.histogram(m_values, bins=60)
        max_count = max(int(np.max(counts)), 1)
        body = [
            f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">Inferred m Histogram (test)</text>',
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333"/>',
        ]
        for count, x0, x1 in zip(counts, edges[:-1], edges[1:]):
            x = _svg_scale(float(x0), float(edges[0]), float(edges[-1]), left, left + plot_w)
            x_next = _svg_scale(float(x1), float(edges[0]), float(edges[-1]), left, left + plot_w)
            bar_h = _svg_scale(float(count), 0.0, float(max_count), 0.0, plot_h)
            body.append(
                f'<rect x="{x:.2f}" y="{top + plot_h - bar_h:.2f}" width="{max(x_next - x - 1, 1):.2f}" height="{bar_h:.2f}" fill="#9467bd" opacity="0.85"/>'
            )
        body.extend(
            [
                f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="15">m = ln(|u_b|) / (ln(alpha^2/C^2) + ln(|u_b|))</text>',
                f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="Arial" font-size="15">Count</text>',
            ]
        )
        m_path = out_dir / "inferred_m_histogram.svg"
        _write_svg(m_path, width, height, body)
        print(f"Saved inferred m histogram to {m_path}")

    print(f"Saved prediction scatter to {scatter_path}")
    print(f"Saved test error histogram to {hist_path}")
    print(f"Saved loss curves to {loss_path}")


def save_accuracy_outputs(
    plots_dir: str,
    split_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    history_by_seed: Dict[int, List[Dict[str, float]]],
    best_seed: int,
    test_x_raw: np.ndarray,
) -> None:
    out_dir = Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_true, test_pred = split_predictions["test"]
    m_values_full = infer_m_values_full(test_x_raw, test_pred)
    m_values = m_values_full[np.isfinite(m_values_full)]
    print_m_summary(m_values, len(test_pred))
    print_m_outlier_details(test_x_raw, test_true, test_pred, m_values_full)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        configure_matplotlib_for_latex(plt)
    except ModuleNotFoundError:
        print(
            "matplotlib is not installed; skipping PDF plot generation. "
            "Install it with `python3 -m pip install matplotlib` for PDF output."
        )
        return

    fig, ax = plt.subplots(figsize=LATEX_SQUARE_FIGSIZE)
    all_true = []
    all_pred = []
    for name, (y_true, y_pred) in split_predictions.items():
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        all_true.append(y_true_flat)
        all_pred.append(y_pred_flat)
        ax.scatter(y_true_flat, y_pred_flat, s=6, alpha=0.45, label=name)
    all_true_flat = np.concatenate(all_true)
    all_pred_flat = np.concatenate(all_pred)
    lo = float(min(np.min(all_true_flat), np.min(all_pred_flat)))
    hi = float(max(np.max(all_true_flat), np.max(all_pred_flat)))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="ideal")
    ax.set_xlabel(r"True $\alpha^2$")
    ax.set_ylabel(r"Predicted $\alpha^2$")
    ax.set_title("Prediction Scatter")
    ax.legend(frameon=False)
    fig.tight_layout()
    scatter_path = out_dir / "prediction_scatter.pdf"
    save_matplotlib_figure(fig, scatter_path)
    plt.close(fig)

    test_error = (test_pred - test_true).reshape(-1)
    fig, ax = plt.subplots(figsize=LATEX_FIGSIZE)
    ax.hist(test_error, bins=60, alpha=0.85)
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Prediction error, $\hat{\alpha}^2-\alpha^2$")
    ax.set_ylabel("Count")
    ax.set_title("Test Error Histogram")
    fig.tight_layout()
    hist_path = out_dir / "test_error_histogram.pdf"
    save_matplotlib_figure(fig, hist_path)
    plt.close(fig)

    history = history_by_seed[best_seed]
    epochs = [item["epoch"] for item in history]
    train_rmse = [item["train_rmse"] for item in history]
    val_rmse = [item["val_rmse"] for item in history]
    fig, ax = plt.subplots(figsize=LATEX_FIGSIZE)
    ax.plot(epochs, train_rmse, label="train", linewidth=1.0)
    ax.plot(epochs, val_rmse, label="validation", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"RMSE of $\alpha^2$")
    ax.set_title(f"Loss Curves (seed {best_seed})")
    ax.legend(frameon=False)
    fig.tight_layout()
    loss_path = out_dir / "loss_curves.pdf"
    save_matplotlib_figure(fig, loss_path)
    plt.close(fig)

    if len(m_values) > 0:
        fig, ax = plt.subplots(figsize=LATEX_FIGSIZE)
        ax.hist(m_values, bins=60, alpha=0.85)
        ax.set_xlabel(r"$m=\ln(|u_b|)/(\ln(\alpha^2/C^2)+\ln(|u_b|))$")
        ax.set_ylabel("Count")
        ax.set_title(r"Inferred $m$ Histogram (test)")
        fig.tight_layout()
        m_path = out_dir / "inferred_m_histogram.pdf"
        save_matplotlib_figure(fig, m_path)
        plt.close(fig)
