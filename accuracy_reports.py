from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


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

    return m_values[np.isfinite(m_values)]


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
            f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="15">True alpha2</text>',
            f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="Arial" font-size="15">Predicted alpha2</text>',
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
            f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="15">Prediction error (predicted - true)</text>',
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
        f'<text x="24" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_h / 2})" font-family="Arial" font-size="15">RMSE (log scale)</text>',
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
                f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="15">m = ln(vmag) / (ln(alpha2/C2) + ln(vmag))</text>',
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

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            "matplotlib is not installed; saving lightweight SVG plots instead. "
            "Install it with `python3 -m pip install matplotlib` for PNG output."
        )
        save_accuracy_svgs(out_dir, split_predictions, history_by_seed, best_seed, test_x_raw)
        return

    plt.figure(figsize=(7, 6))
    all_true = []
    all_pred = []
    for name, (y_true, y_pred) in split_predictions.items():
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        all_true.append(y_true_flat)
        all_pred.append(y_pred_flat)
        plt.scatter(y_true_flat, y_pred_flat, s=10, alpha=0.45, label=name)
    all_true_flat = np.concatenate(all_true)
    all_pred_flat = np.concatenate(all_pred)
    lo = float(min(np.min(all_true_flat), np.min(all_pred_flat)))
    hi = float(max(np.max(all_true_flat), np.max(all_pred_flat)))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="ideal")
    plt.xlabel("True alpha2")
    plt.ylabel("Predicted alpha2")
    plt.title("Prediction Scatter")
    plt.legend()
    plt.tight_layout()
    scatter_path = out_dir / "prediction_scatter.png"
    plt.savefig(scatter_path, dpi=200)
    plt.close()

    test_true, test_pred = split_predictions["test"]
    test_error = (test_pred - test_true).reshape(-1)
    plt.figure(figsize=(7, 5))
    plt.hist(test_error, bins=60, alpha=0.85)
    plt.axvline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Prediction error (predicted - true)")
    plt.ylabel("Count")
    plt.title("Test Error Histogram")
    plt.tight_layout()
    hist_path = out_dir / "test_error_histogram.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()

    history = history_by_seed[best_seed]
    epochs = [item["epoch"] for item in history]
    train_rmse = [item["train_rmse"] for item in history]
    val_rmse = [item["val_rmse"] for item in history]
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_rmse, label="train")
    plt.plot(epochs, val_rmse, label="validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title(f"Loss Curves (seed {best_seed})")
    plt.legend()
    plt.tight_layout()
    loss_path = out_dir / "loss_curves.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()

    _, test_pred = split_predictions["test"]
    m_values = infer_m_values(test_x_raw, test_pred)
    print_m_summary(m_values, len(test_pred))
    if len(m_values) > 0:
        plt.figure(figsize=(7, 5))
        plt.hist(m_values, bins=60, alpha=0.85)
        plt.xlabel("m = ln(vmag) / (ln(alpha2/C2) + ln(vmag))")
        plt.ylabel("Count")
        plt.title("Inferred m Histogram (test)")
        plt.tight_layout()
        m_path = out_dir / "inferred_m_histogram.png"
        plt.savefig(m_path, dpi=200)
        plt.close()
        print(f"Saved inferred m histogram to {m_path}")

    print(f"Saved prediction scatter to {scatter_path}")
    print(f"Saved test error histogram to {hist_path}")
    print(f"Saved loss curves to {loss_path}")
