from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np


EPS = 1e-12


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return math.nan
    a, b = a[mask], b[mask]
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def safe_ssim(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
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
    covariance = float(((a - mu_a) * (b - mu_b)).mean())
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if denominator <= EPS:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(
        ((2 * mu_a * mu_b + c1) * (2 * covariance + c2)) / denominator
    )


def js_distance(first: np.ndarray, second: np.ndarray, zero_to_uniform=True) -> float:
    p = np.clip(np.asarray(first, dtype=np.float64).ravel(), 0.0, None)
    q = np.clip(np.asarray(second, dtype=np.float64).ravel(), 0.0, None)
    if p.size == 0 or p.size != q.size:
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


def fraction_metrics(prediction: np.ndarray, truth: np.ndarray) -> Mapping[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(truth, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError(f"Prediction/truth shape mismatch: {pred.shape} vs {gt.shape}")
    difference = pred - gt
    compatible_jsd = [
        js_distance(pred[:, index], gt[:, index], zero_to_uniform=False)
        for index in range(pred.shape[1])
        if pred[:, index].max() > 0 and gt[:, index].max() > 0
    ]
    return {
        "ST_RMSE": float(np.mean(np.sqrt(np.mean(difference**2, axis=0)))),
        "ST_JSD": float(np.mean(compatible_jsd)) if compatible_jsd else math.nan,
        "ST_SSIM": safe_ssim(gt, pred),
        "ST_Pearson": safe_pearson(gt, pred),
        "ST_eval_spots": int(pred.shape[0]),
        "ST_eval_types": int(pred.shape[1]),
    }


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    values = np.clip(values, 0.0, None)
    return values / np.clip(values.sum(axis=1, keepdims=True), EPS, None)


def _binary_mcc(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = float(np.sum(predicted & truth))
    tn = float(np.sum(~predicted & ~truth))
    fp = float(np.sum(predicted & ~truth))
    fn = float(np.sum(~predicted & truth))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if denominator <= 0 else float((tp * tn - fp * fn) / denominator)


def _macro_f1(predicted: np.ndarray, truth: np.ndarray, count: int) -> float:
    scores = []
    for label in range(count):
        truth_positive = truth == label
        if not np.any(truth_positive):
            continue
        pred_positive = predicted == label
        tp = float(np.sum(pred_positive & truth_positive))
        fp = float(np.sum(pred_positive & ~truth_positive))
        fn = float(np.sum(~pred_positive & truth_positive))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator <= 0 else 2 * tp / denominator)
    return float(np.mean(scores)) if scores else 0.0


def structural_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    adjacency: Optional[np.ndarray],
    cell_types: Optional[Sequence[str]] = None,
) -> Mapping[str, float]:
    pred = _normalize_rows(prediction)
    gt = _normalize_rows(truth)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError("Structural metrics require matched 2D arrays")
    n, k = pred.shape
    pred_dominant = np.argmax(pred, axis=1)
    gt_dominant = np.argmax(gt, axis=1)
    pred_counts = np.bincount(pred_dominant, minlength=k)
    gt_counts = np.bincount(gt_dominant, minlength=k)
    edge_mcc = boundary = gt_boundary = math.nan
    if adjacency is not None:
        graph = np.maximum(adjacency, adjacency.T) > 0
        np.fill_diagonal(graph, False)
        source, destination = np.where(np.triu(graph, k=1))
        if len(source):
            pred_edges = pred_dominant[source] != pred_dominant[destination]
            gt_edges = gt_dominant[source] != gt_dominant[destination]
            edge_mcc = _binary_mcc(pred_edges, gt_edges)
            boundary = float(np.mean(pred_edges))
            gt_boundary = float(np.mean(gt_edges))
    type_correlations = []
    for index in range(k):
        if np.std(gt[:, index]) > EPS:
            type_correlations.append(safe_pearson(pred[:, index], gt[:, index]))
    pyramidal = [
        index for index, name in enumerate(cell_types or [])
        if "pyramidal" in str(name).lower()
    ]
    pyramidal_correlation = (
        safe_pearson(pred[:, pyramidal], gt[:, pyramidal]) if pyramidal else math.nan
    )
    if not np.isfinite(pyramidal_correlation):
        pyramidal_correlation = 0.0
    return {
        "ST_Pearson_type_macro": float(np.mean(type_correlations)),
        "ST_Pearson_type_median": float(np.median(type_correlations)),
        "ST_Pearson_type_valid": int(len(type_correlations)),
        "ST_dominant_macro_f1": _macro_f1(pred_dominant, gt_dominant, k),
        "ST_active_dominant_types": int(np.sum(pred_counts > 0)),
        "ST_gt_active_dominant_types": int(np.sum(gt_counts > 0)),
        "ST_dominant_top_fraction": float(np.max(pred_counts) / n),
        "ST_gt_dominant_top_fraction": float(np.max(gt_counts) / n),
        "ST_edge_MCC": float(edge_mcc),
        "ST_boundary_fraction": float(boundary),
        "ST_gt_boundary_fraction": float(gt_boundary),
        "ST_boundary_abs_error": float(abs(boundary - gt_boundary)),
        "ST_same_dominant_abs_error": float(abs(boundary - gt_boundary)),
        "ST_pyramidal_layer_Pearson": float(pyramidal_correlation),
        "ST_decon_std": float(np.std(pred)),
        "ST_gt_decon_std": float(np.std(gt)),
        "ST_decon_std_ratio": float(np.std(pred) / max(float(np.std(gt)), EPS)),
        "ST_max_prob_mean": float(np.mean(np.max(pred, axis=1))),
        "ST_gt_max_prob_mean": float(np.mean(np.max(gt, axis=1))),
    }


def structural_validity(metrics: Mapping[str, float]) -> tuple[bool, list[str]]:
    reasons = []
    if float(metrics["ST_max_prob_mean"]) < 0.25:
        reasons.append("flat_max_prob")
    if float(metrics["ST_decon_std_ratio"]) < 0.35:
        reasons.append("flat_decon_std")
    if float(metrics["ST_dominant_top_fraction"]) > 0.50:
        reasons.append("dominant_class_collapse")
    return not reasons, reasons
