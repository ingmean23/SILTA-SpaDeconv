from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12


def read_matrix(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate row identifiers in {path}")
    if frame.columns.has_duplicates:
        raise ValueError(f"Duplicate column identifiers in {path}")
    return frame


def write_json(path: Path | str, value: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ordered_text_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > EPS)


def align_prediction(
    prediction: pd.DataFrame, truth: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    unknown_spots = prediction.index.difference(truth.index).tolist()
    missing_spots = truth.index.difference(prediction.index).tolist()
    unknown_types = prediction.columns.difference(truth.columns).tolist()
    missing_types = truth.columns.difference(prediction.columns).tolist()
    if unknown_spots:
        raise ValueError(f"Prediction contains unknown spots: {unknown_spots[:5]}")
    if missing_spots:
        raise ValueError(f"Prediction is missing spots: {missing_spots[:5]}")
    if unknown_types:
        raise ValueError(f"Prediction contains unknown cell types: {unknown_types[:5]}")

    aligned = prediction.reindex(index=truth.index, columns=truth.columns, fill_value=0.0)
    pred_values = normalize_rows(aligned.to_numpy())
    truth_values = normalize_rows(truth.to_numpy())
    aligned = pd.DataFrame(pred_values, index=truth.index, columns=truth.columns)
    normalized_truth = pd.DataFrame(
        truth_values, index=truth.index, columns=truth.columns
    )
    diagnostics = {
        "missing_cell_types_filled_zero": missing_types,
        "prediction_zero_rows": int((pred_values.sum(axis=1) <= EPS).sum()),
        "prediction_row_sum_min": float(pred_values.sum(axis=1).min()),
        "prediction_row_sum_max": float(pred_values.sum(axis=1).max()),
    }
    return aligned, normalized_truth, diagnostics


def safe_ssim(true: np.ndarray, pred: np.ndarray) -> float:
    a = np.asarray(true, dtype=np.float64).ravel()
    b = np.asarray(pred, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return math.nan
    a, b = a[mask], b[mask]
    data_range = max(float(max(a.max(), b.max()) - min(a.min(), b.min())), EPS)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_b = float(a.mean()), float(b.mean())
    var_a = float(((a - mu_a) ** 2).mean())
    var_b = float(((b - mu_b) ** 2).mean())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    denom = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if denom <= EPS:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / denom)


def safe_pearson(true: np.ndarray, pred: np.ndarray, constant_value=math.nan) -> float:
    a = np.asarray(true, dtype=np.float64).ravel()
    b = np.asarray(pred, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return constant_value
    a, b = a[mask], b[mask]
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return constant_value
    return float(np.corrcoef(a, b)[0, 1])


def js_distance(first: np.ndarray, second: np.ndarray, zero_to_uniform=True) -> float:
    p = np.clip(np.asarray(first, dtype=np.float64).ravel(), 0.0, None)
    q = np.clip(np.asarray(second, dtype=np.float64).ravel(), 0.0, None)
    if p.size == 0 or q.size == 0 or p.size != q.size:
        return math.nan
    if p.sum() <= EPS:
        if not zero_to_uniform:
            return math.nan
        p = np.ones_like(p)
    if q.sum() <= EPS:
        if not zero_to_uniform:
            return math.nan
        q = np.ones_like(q)
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))

    return float(np.sqrt(max(0.0, 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint))))


def fraction_metrics(prediction: np.ndarray, truth: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError(f"Prediction/truth shape mismatch: {pred.shape} vs {gt.shape}")
    diff = pred - gt
    per_type_rmse = np.sqrt(np.mean(diff**2, axis=0))
    per_spot_rmse = np.sqrt(np.mean(diff**2, axis=1))
    per_type_mae = np.mean(np.abs(diff), axis=0)
    per_spot_mae = np.mean(np.abs(diff), axis=1)

    compat_type_jsd = [
        js_distance(pred[:, idx], gt[:, idx], zero_to_uniform=False)
        for idx in range(pred.shape[1])
        if pred[:, idx].max() > 0 and gt[:, idx].max() > 0
    ]
    balanced_type_jsd = [
        js_distance(pred[:, idx], gt[:, idx], zero_to_uniform=True)
        for idx in range(pred.shape[1])
    ]
    spot_jsd = [
        js_distance(pred[idx], gt[idx], zero_to_uniform=True)
        for idx in range(pred.shape[0])
    ]
    type_ssim = [
        safe_ssim(gt[:, idx], pred[:, idx]) for idx in range(pred.shape[1])
    ]
    spot_ssim = [safe_ssim(gt[idx], pred[idx]) for idx in range(pred.shape[0])]
    type_pearson = [
        safe_pearson(gt[:, idx], pred[:, idx], constant_value=0.0)
        for idx in range(pred.shape[1])
    ]

    dot = np.sum(pred * gt, axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(gt, axis=1)
    spot_cosine = np.divide(
        dot, denominator, out=np.zeros_like(dot), where=denominator > EPS
    )

    return {
        "ST_RMSE": float(np.mean(per_type_rmse)),
        "ST_RMSE_flat": float(np.sqrt(np.mean(diff**2))),
        "ST_RMSE_spot": float(np.mean(per_spot_rmse)),
        "ST_MAE": float(np.mean(per_type_mae)),
        "ST_MAE_spot": float(np.mean(per_spot_mae)),
        "ST_JSD": float(np.mean(compat_type_jsd)) if compat_type_jsd else math.nan,
        "ST_JSD_type_balanced": float(np.mean(balanced_type_jsd)),
        "ST_JSD_spot": float(np.mean(spot_jsd)),
        "ST_SSIM": safe_ssim(gt, pred),
        "ST_SSIM_type_mean": float(np.mean(type_ssim)),
        "ST_SSIM_spot_mean": float(np.mean(spot_ssim)),
        "ST_Pearson": safe_pearson(gt, pred),
        "ST_Pearson_type_mean": float(np.mean(type_pearson)),
        "ST_spot_cosine_mean": float(np.mean(spot_cosine)),
        "ST_eval_spots": int(pred.shape[0]),
        "ST_eval_types": int(pred.shape[1]),
    }


def rank_score(mean_rank: float, method_count: int) -> float:
    if method_count <= 1:
        return 1.0
    return float(1.0 - (mean_rank - 1.0) / (method_count - 1.0))
