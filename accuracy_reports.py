from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


LATEX_FIGSIZE = (6.0, 4.0)
LATEX_SQUARE_FIGSIZE = (6.0, 4.0)


def save_accuracy_report_data(
    report_path: str,
    split_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    history_by_seed: Dict[int, List[Dict[str, float]]],
    best_seed: int,
    test_x_raw: np.ndarray,
) -> None:
    history = history_by_seed[best_seed]
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_true=split_predictions["train"][0],
        train_pred=split_predictions["train"][1],
        validation_true=split_predictions["validation"][0],
        validation_pred=split_predictions["validation"][1],
        test_true=split_predictions["test"][0],
        test_pred=split_predictions["test"][1],
        test_x_raw=test_x_raw,
        best_seed=np.asarray(best_seed, dtype=np.int64),
        history_epoch=np.asarray([item["epoch"] for item in history], dtype=np.int64),
        history_train_rmse=np.asarray(
            [item["train_rmse"] for item in history], dtype=np.float64
        ),
        history_val_rmse=np.asarray(
            [item["val_rmse"] for item in history], dtype=np.float64
        ),
    )
    print(f"Saved plotting data to {path}")


def load_accuracy_report_data(
    report_path: str,
) -> tuple[
    Dict[str, Tuple[np.ndarray, np.ndarray]],
    Dict[int, List[Dict[str, float]]],
    int,
    np.ndarray,
]:
    with np.load(report_path, allow_pickle=False) as data:
        best_seed = int(data["best_seed"])
        history = [
            {
                "epoch": int(epoch),
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
            }
            for epoch, train_rmse, val_rmse in zip(
                data["history_epoch"],
                data["history_train_rmse"],
                data["history_val_rmse"],
            )
        ]
        split_predictions = {
            "train": (data["train_true"].copy(), data["train_pred"].copy()),
            "validation": (
                data["validation_true"].copy(),
                data["validation_pred"].copy(),
            ),
            "test": (data["test_true"].copy(), data["test_pred"].copy()),
        }
        test_x_raw = data["test_x_raw"].copy()
    return split_predictions, {best_seed: history}, best_seed, test_x_raw


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
        pred_raw = torch.clamp_min(pred_raw, 0.0)
        preds.append(pred_raw.detach().cpu().numpy())

    return np.concatenate(preds, axis=0)


def compute_accuracy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relative_error_floor: float = 1e5,
) -> Dict[str, float]:
    error = y_pred - y_true
    abs_error = np.abs(error)
    rmse = float(np.sqrt(np.mean(error * error)))
    mae = float(np.mean(abs_error))
    rel_error = (abs_error / (np.abs(y_true) + 1e-10)).reshape(-1) * 100.0
    meaningful_mask = np.abs(y_true).reshape(-1) >= relative_error_floor
    meaningful_rel_error = rel_error[meaningful_mask]
    ss_res = float(np.sum(error * error))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    def percentile(values: np.ndarray, q: float) -> float:
        if len(values) == 0:
            return float("nan")
        return float(np.percentile(values, q))

    return {
        "rmse": rmse,
        "mae": mae,
        "rel_p50_percent": percentile(rel_error, 50.0),
        "rel_p90_percent": percentile(rel_error, 90.0),
        "rel_p95_percent": percentile(rel_error, 95.0),
        "rel_p99_percent": percentile(rel_error, 99.0),
        "relative_error_floor": float(relative_error_floor),
        "relative_error_count": float(np.count_nonzero(meaningful_mask)),
        "floor_rel_p50_percent": percentile(meaningful_rel_error, 50.0),
        "floor_rel_p95_percent": percentile(meaningful_rel_error, 95.0),
        "r2": float(r2),
    }


def print_accuracy_metrics(name: str, metrics: Dict[str, float]) -> None:
    print(
        f"{name:<10} RMSE={fmt_sci(metrics['rmse'])}"
        f"  MAE={fmt_sci(metrics['mae'])}"
        f"  RelErr[p50/p95/p99]="
        f"{fmt_percent(metrics['rel_p50_percent'])}/"
        f"{fmt_percent(metrics['rel_p95_percent'])}/"
        f"{fmt_percent(metrics['rel_p99_percent'])}%"
        f"  RelErr(alpha2>={fmt_sci(metrics['relative_error_floor'])})[p50/p95]="
        f"{fmt_percent(metrics['floor_rel_p50_percent'])}/"
        f"{fmt_percent(metrics['floor_rel_p95_percent'])}%"
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
    max_rows: int = 50,
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
    shown_indices = outlier_indices[:max_rows]
    for idx in shown_indices:
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
    if len(outlier_indices) > max_rows:
        print(f"  ... showing first {max_rows} of {len(outlier_indices)} inferred-m outliers")


def configure_matplotlib_for_latex(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )


def save_matplotlib_figure(fig, path: Path) -> None:
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"Saved {png_path}")


def save_relative_error_heatmap(
    out_dir: Path,
    test_x_raw: np.ndarray,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    plt,
    bins: int = 60,
) -> None:
    c2 = np.sqrt(np.maximum(test_x_raw[:, 0].astype(np.float64), 0.0))
    vmag = np.sqrt(np.maximum(test_x_raw[:, 1].astype(np.float64), 0.0))
    y_true = test_true.reshape(-1).astype(np.float64)
    y_pred = test_pred.reshape(-1).astype(np.float64)
    rel_error_percent = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-10) * 100.0

    finite = np.isfinite(c2) & np.isfinite(vmag) & np.isfinite(rel_error_percent)
    c2 = c2[finite]
    vmag = vmag[finite]
    rel_error_percent = rel_error_percent[finite]
    if len(c2) == 0:
        print("Skipped relative-error heatmap: no finite test points.")
        return

    c2_edges = np.linspace(float(c2.min()), float(c2.max()), bins + 1)
    vmag_edges = np.linspace(float(vmag.min()), float(vmag.max()), bins + 1)
    c2_ids = np.digitize(c2, c2_edges[1:-1], right=False)
    vmag_ids = np.digitize(vmag, vmag_edges[1:-1], right=False)

    sum_error = np.zeros((bins, bins), dtype=np.float64)
    counts = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(sum_error, (vmag_ids, c2_ids), rel_error_percent)
    np.add.at(counts, (vmag_ids, c2_ids), 1.0)
    mean_error = np.full((bins, bins), np.nan, dtype=np.float64)
    occupied = counts > 0
    mean_error[occupied] = sum_error[occupied] / counts[occupied]

    vmax = float(np.nanpercentile(mean_error, 95.0)) if np.any(occupied) else 1.0
    vmax = max(vmax, 1.0)
    fig, ax = plt.subplots(figsize=LATEX_SQUARE_FIGSIZE)
    mesh = ax.pcolormesh(
        c2_edges,
        vmag_edges,
        mean_error,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    fig.tight_layout()
    save_matplotlib_figure(fig, out_dir / "test_relative_error_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=LATEX_SQUARE_FIGSIZE)
    sample_count = min(len(c2), 12000)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(c2), size=sample_count, replace=False) if sample_count < len(c2) else np.arange(len(c2))
    scatter_values = np.clip(rel_error_percent[sample_idx], 0.0, float(np.percentile(rel_error_percent, 99.0)))
    sc = ax.scatter(
        c2[sample_idx],
        vmag[sample_idx],
        c=scatter_values,
        s=4,
        alpha=0.55,
        cmap="magma",
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    fig.tight_layout()
    save_matplotlib_figure(fig, out_dir / "test_relative_error_scatter.png")
    plt.close(fig)


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
            "matplotlib is not installed; skipping PNG plot generation. "
            "Install it with `python3 -m pip install matplotlib` for PNG output."
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
    ax.legend(frameon=False)
    fig.tight_layout()
    scatter_path = out_dir / "prediction_scatter.png"
    save_matplotlib_figure(fig, scatter_path)
    plt.close(fig)

    test_error = (test_pred - test_true).reshape(-1)
    fig, ax = plt.subplots(figsize=LATEX_FIGSIZE)
    ax.hist(test_error, bins=60, alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    hist_path = out_dir / "test_error_histogram.png"
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
    ax.legend(frameon=False)
    fig.tight_layout()
    loss_path = out_dir / "loss_curves.png"
    save_matplotlib_figure(fig, loss_path)
    plt.close(fig)

    if len(m_values) > 0:
        fig, ax = plt.subplots(figsize=LATEX_FIGSIZE)
        ax.hist(m_values, bins=60, alpha=0.85)
        ax.set_yscale("log")
        fig.tight_layout()
        m_path = out_dir / "inferred_m_histogram.png"
        save_matplotlib_figure(fig, m_path)
        plt.close(fig)

    save_relative_error_heatmap(out_dir, test_x_raw, test_true, test_pred, plt)
