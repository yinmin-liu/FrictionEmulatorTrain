from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


RAW_X_OFFSET = np.array([0.0, 0.0], dtype=np.float64)
RAW_X_SCALE = np.array([9.05e6, 2.08e-5], dtype=np.float64)
RAW_Y_OFFSET = np.array([0.0], dtype=np.float64)
RAW_Y_SCALE = np.array([2.09e11], dtype=np.float64)

RAW_VARIABLE_NAMES = ("C2", "vmag", "alpha2")


@dataclass
class NormalizationConfig:
    mode: str
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    x_floor: np.ndarray
    y_floor: np.ndarray


@dataclass
class OutlierFilterResult:
    mask: np.ndarray
    bounds: dict[str, tuple[float, float]]


def normalize(values: np.ndarray, offset: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - offset) / scale


def log_transform(values: np.ndarray, floors: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(values, floors))


def sqrt_transform(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(values, 0.0))


def raw_variable_matrix(x_raw: np.ndarray, y_raw: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(x_raw, dtype=np.float64),
            np.asarray(y_raw, dtype=np.float64),
        ],
        axis=1,
    )


def raw_outlier_filter_mask(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    columns: tuple[str, ...],
    mode: str,
    lower_percentile: float,
    upper_percentile: float,
    absolute_bounds: dict[str, tuple[float | None, float | None]] | None = None,
) -> OutlierFilterResult:
    data = raw_variable_matrix(x_raw, y_raw)
    column_to_index = {name: i for i, name in enumerate(RAW_VARIABLE_NAMES)}
    mask = np.ones(data.shape[0], dtype=bool)
    bounds: dict[str, tuple[float, float]] = {}

    for column in columns:
        if column not in column_to_index:
            raise RuntimeError(f"Unsupported outlier-filter column: {column}")
        values = data[:, column_to_index[column]]
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            raise RuntimeError(f"Column {column} has no finite values for outlier filtering.")

        if mode == "percentile":
            lower = float(np.percentile(finite, lower_percentile))
            upper = float(np.percentile(finite, upper_percentile))
        elif mode == "absolute":
            if absolute_bounds is None or column not in absolute_bounds:
                raise RuntimeError(f"Missing absolute outlier bounds for {column}.")
            lower_raw, upper_raw = absolute_bounds[column]
            lower = float(np.min(finite)) if lower_raw is None else float(lower_raw)
            upper = float(np.max(finite)) if upper_raw is None else float(upper_raw)
        else:
            raise RuntimeError(f"Unsupported outlier-filter mode: {mode}")

        bounds[column] = (lower, upper)
        mask &= np.isfinite(values) & (values >= lower) & (values <= upper)

    return OutlierFilterResult(mask=mask, bounds=bounds)


def build_normalization_config(
    mode: str,
    train_x_raw: np.ndarray,
    train_y_raw: np.ndarray,
    x_floor: np.ndarray,
    y_floor: np.ndarray,
) -> NormalizationConfig:
    if mode == "raw":
        return NormalizationConfig(
            mode=mode,
            x_mean=RAW_X_OFFSET.copy(),
            x_std=RAW_X_SCALE.copy(),
            y_mean=RAW_Y_OFFSET.copy(),
            y_std=RAW_Y_SCALE.copy(),
            x_floor=x_floor.astype(np.float64),
            y_floor=y_floor.astype(np.float64),
        )

    if mode == "mixed":
        x_trans = transform_x_np(train_x_raw, mode, x_floor)
        y_trans = transform_y_np(train_y_raw, mode, y_floor)
        x_mean = np.array([0.0, x_trans[:, 1].mean()], dtype=np.float64)
        x_std = np.array([RAW_X_SCALE[0], x_trans[:, 1].std()], dtype=np.float64)
        y_mean = y_trans.mean(axis=0)
        y_std = y_trans.std(axis=0)
        x_std[x_std < 1e-12] = 1.0
        y_std[y_std < 1e-12] = 1.0
        return NormalizationConfig(
            mode=mode,
            x_mean=x_mean.astype(np.float64),
            x_std=x_std.astype(np.float64),
            y_mean=y_mean.astype(np.float64),
            y_std=y_std.astype(np.float64),
            x_floor=x_floor.astype(np.float64),
            y_floor=y_floor.astype(np.float64),
        )

    if mode not in ("log", "sqrt"):
        raise RuntimeError(f"Unsupported normalization mode: {mode}")

    x_trans = transform_x_np(train_x_raw, mode, x_floor)
    y_trans = transform_y_np(train_y_raw, mode, y_floor)
    x_mean = x_trans.mean(axis=0)
    x_std = x_trans.std(axis=0)
    y_mean = y_trans.mean(axis=0)
    y_std = y_trans.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    y_std[y_std < 1e-12] = 1.0
    return NormalizationConfig(
        mode=mode,
        x_mean=x_mean.astype(np.float64),
        x_std=x_std.astype(np.float64),
        y_mean=y_mean.astype(np.float64),
        y_std=y_std.astype(np.float64),
        x_floor=x_floor.astype(np.float64),
        y_floor=y_floor.astype(np.float64),
    )


def transform_x_np(x_raw: np.ndarray, mode: str, x_floor: np.ndarray) -> np.ndarray:
    x = np.asarray(x_raw, dtype=np.float64)
    if mode == "raw":
        return x
    if mode == "mixed":
        x_trans = x.copy()
        x_trans[:, 1] = np.log(np.maximum(x[:, 1], x_floor[1]))
        return x_trans
    if mode == "sqrt":
        return sqrt_transform(x)
    return log_transform(x, x_floor)


def transform_y_np(y_raw: np.ndarray, mode: str, y_floor: np.ndarray) -> np.ndarray:
    y = np.asarray(y_raw, dtype=np.float64)
    if mode == "raw":
        return y
    if mode == "sqrt":
        return sqrt_transform(y)
    return log_transform(y, y_floor)


def normalize_x_np(x_raw: np.ndarray, norm: NormalizationConfig) -> np.ndarray:
    return normalize(transform_x_np(x_raw, norm.mode, norm.x_floor), norm.x_mean, norm.x_std)


def normalize_y_np(y_raw: np.ndarray, norm: NormalizationConfig) -> np.ndarray:
    return normalize(transform_y_np(y_raw, norm.mode, norm.y_floor), norm.y_mean, norm.y_std)


def inverse_transform_y_np(y_trans: np.ndarray, norm: NormalizationConfig) -> np.ndarray:
    if norm.mode == "raw":
        return y_trans
    if norm.mode == "sqrt":
        return np.maximum(y_trans, 0.0) ** 2
    return np.exp(y_trans)


def denormalize_y_np(y_norm: np.ndarray, norm: NormalizationConfig) -> np.ndarray:
    return inverse_transform_y_np(y_norm * norm.y_std + norm.y_mean, norm)


def transform_x_torch(x_raw: torch.Tensor, norm: NormalizationConfig, device: torch.device) -> torch.Tensor:
    if norm.mode == "raw":
        return x_raw
    x_floor = torch.as_tensor(norm.x_floor, dtype=torch.float32, device=device)
    if norm.mode == "mixed":
        c2 = x_raw[:, 0:1]
        vmag = torch.log(torch.maximum(x_raw[:, 1:2], x_floor[1]))
        return torch.cat((c2, vmag), dim=1)
    if norm.mode == "sqrt":
        return torch.sqrt(torch.clamp_min(x_raw, 0.0))
    return torch.log(torch.maximum(x_raw, x_floor))


def inverse_transform_y_torch(y_trans: torch.Tensor, norm: NormalizationConfig) -> torch.Tensor:
    if norm.mode == "raw":
        return y_trans
    if norm.mode == "sqrt":
        return torch.square(torch.clamp_min(y_trans, 0.0))
    return torch.exp(y_trans)


def target_balanced_sample_indices(
    target: np.ndarray,
    n_bins: int,
    n_samples: int,
    balance_power: float,
    max_weight: float,
    seed: int,
) -> np.ndarray:
    finite_target = target[np.isfinite(target)]
    if len(finite_target) == 0:
        raise RuntimeError("Target has no finite values for balanced sampling.")

    edges = np.linspace(float(finite_target.min()), float(finite_target.max()), n_bins + 1)
    bin_ids = np.digitize(target, edges[1:-1], right=False)
    counts = np.bincount(bin_ids, minlength=n_bins).astype(np.float64)
    occupied = counts > 0
    if not np.any(occupied):
        raise RuntimeError("No occupied target bins were found for balanced sampling.")

    bin_weights = np.zeros(n_bins, dtype=np.float64)
    bin_weights[occupied] = counts[occupied] ** (-balance_power)
    median_weight = np.median(bin_weights[occupied])
    if median_weight > 0.0:
        bin_weights[occupied] /= median_weight
    bin_weights[occupied] = np.minimum(bin_weights[occupied], max_weight)

    sample_weights = bin_weights[bin_ids]
    sample_weights = np.where(np.isfinite(target), sample_weights, 0.0)
    if sample_weights.sum() <= 0.0:
        raise RuntimeError("Balanced sampling weights sum to zero.")

    rng = np.random.default_rng(seed)
    probabilities = sample_weights / sample_weights.sum()
    return rng.choice(len(target), size=n_samples, replace=True, p=probabilities)


def target_balanced_sample_weights(
    target: np.ndarray,
    n_bins: int,
    balance_power: float,
    max_weight: float,
) -> np.ndarray:
    finite_target = target[np.isfinite(target)]
    if len(finite_target) == 0:
        raise RuntimeError("Target has no finite values for balanced sampling.")

    edges = np.linspace(float(finite_target.min()), float(finite_target.max()), n_bins + 1)
    bin_ids = np.digitize(target, edges[1:-1], right=False)
    counts = np.bincount(bin_ids, minlength=n_bins).astype(np.float64)
    occupied = counts > 0
    if not np.any(occupied):
        raise RuntimeError("No occupied target bins were found for balanced sampling.")

    bin_weights = np.zeros(n_bins, dtype=np.float64)
    bin_weights[occupied] = counts[occupied] ** (-balance_power)
    median_weight = np.median(bin_weights[occupied])
    if median_weight > 0.0:
        bin_weights[occupied] /= median_weight
    bin_weights[occupied] = np.minimum(bin_weights[occupied], max_weight)

    sample_weights = bin_weights[bin_ids]
    sample_weights = np.where(np.isfinite(target), sample_weights, 0.0)
    if sample_weights.sum() <= 0.0:
        raise RuntimeError("Balanced sampling weights sum to zero.")
    return sample_weights.astype(np.float64)
