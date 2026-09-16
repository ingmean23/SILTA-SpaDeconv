from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1.0e-12


def read_fractions(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < -1e-10):
        raise ValueError(f"invalid fractions in {path}")
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= EPS):
        raise ValueError(f"zero-sum fraction row in {path}")
    return pd.DataFrame(values / totals, index=frame.index, columns=frame.columns)


def align(prediction: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(prediction.index) != set(truth.index):
        raise ValueError("prediction and truth spot identifiers differ")
    unknown = sorted(set(prediction.columns) - set(truth.columns))
    if unknown:
        raise ValueError(f"prediction contains unknown cell types: {unknown[:8]}")
    prediction = prediction.reindex(index=truth.index, columns=truth.columns, fill_value=0.0)
    truth = truth.copy()
    prediction = prediction.div(prediction.sum(axis=1).clip(lower=EPS), axis=0)
    truth = truth.div(truth.sum(axis=1).clip(lower=EPS), axis=0)
    return prediction, truth


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    if np.std(left) <= EPS or np.std(right) <= EPS:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).ravel()
    right = np.asarray(second, dtype=np.float64).ravel()
    data_range = max(float(max(left.max(), right.max()) - min(left.min(), right.min())), EPS)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mean_left, mean_right = float(left.mean()), float(right.mean())
    var_left = float(((left - mean_left) ** 2).mean())
    var_right = float(((right - mean_right) ** 2).mean())
    covariance = float(((left - mean_left) * (right - mean_right)).mean())
    denominator = (mean_left**2 + mean_right**2 + c1) * (var_left + var_right + c2)
    if denominator <= EPS:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / denominator)


def _js_distance(first: np.ndarray, second: np.ndarray) -> float:
    left = np.clip(np.asarray(first, dtype=np.float64).ravel(), 0.0, None)
    right = np.clip(np.asarray(second, dtype=np.float64).ravel(), 0.0, None)
    if left.sum() <= EPS or right.sum() <= EPS:
        return math.nan
    left /= left.sum()
    right /= right.sum()
    midpoint = 0.5 * (left + right)
    left_mask, right_mask = left > 0, right > 0
    value = 0.5 * np.sum(left[left_mask] * np.log(left[left_mask] / midpoint[left_mask]))
    value += 0.5 * np.sum(right[right_mask] * np.log(right[right_mask] / midpoint[right_mask]))
    return float(np.sqrt(max(float(value), 0.0)))


def fraction_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    difference = pred - gt
    jsd = [
        _js_distance(pred[:, index], gt[:, index])
        for index in range(pred.shape[1])
        if pred[:, index].max() > 0 and gt[:, index].max() > 0
    ]
    return {
        "ST_RMSE": float(np.mean(np.sqrt(np.mean(difference**2, axis=0)))),
        "ST_RMSE_flat": float(np.sqrt(np.mean(difference**2))),
        "ST_JSD": float(np.mean(jsd)) if jsd else math.nan,
        "ST_SSIM": _ssim(gt, pred),
        "ST_Pearson": _pearson(gt, pred),
        "ST_eval_spots": int(pred.shape[0]),
        "ST_eval_types": int(pred.shape[1]),
    }
