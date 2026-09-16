import math
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


def _normalize_rows(values: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    return values / np.clip(values.sum(axis=1, keepdims=True), eps, None)


def _pearson(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return math.nan
    a = a[mask] - np.mean(a[mask])
    b = b[mask] - np.mean(b[mask])
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= eps:
        return math.nan
    return float(np.dot(a, b) / denominator)


def _undirected_edges(adjacency: Optional[np.ndarray], n: int):
    if adjacency is None:
        return np.array([], dtype=int), np.array([], dtype=int)
    if hasattr(adjacency, "detach"):
        adjacency = adjacency.detach().cpu().numpy()
    adjacency = np.squeeze(np.asarray(adjacency))
    if adjacency.ndim != 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    n = min(n, adjacency.shape[0], adjacency.shape[1])
    graph = np.maximum(adjacency[:n, :n], adjacency[:n, :n].T) > 0
    np.fill_diagonal(graph, False)
    return np.where(np.triu(graph, k=1))


def _binary_mcc(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = float(np.sum(predicted & truth))
    tn = float(np.sum(~predicted & ~truth))
    fp = float(np.sum(predicted & ~truth))
    fn = float(np.sum(~predicted & truth))
    denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator <= 0:
        return 0.0
    return float((tp * tn - fp * fn) / denominator)


def _dominant_macro_f1(predicted: np.ndarray, truth: np.ndarray, k: int) -> float:
    scores = []
    for label in range(k):
        truth_positive = truth == label
        if not np.any(truth_positive):
            continue
        pred_positive = predicted == label
        tp = float(np.sum(pred_positive & truth_positive))
        fp = float(np.sum(pred_positive & ~truth_positive))
        fn = float(np.sum(~pred_positive & truth_positive))
        denominator = 2.0 * tp + fp + fn
        scores.append(0.0 if denominator <= 0 else 2.0 * tp / denominator)
    return float(np.mean(scores)) if scores else 0.0


def osm_structural_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spatial_adjacency: Optional[np.ndarray],
    cell_types: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
) -> Mapping[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if prediction.ndim != 2 or ground_truth.ndim != 2:
        return {}
    n = min(prediction.shape[0], ground_truth.shape[0])
    k = min(prediction.shape[1], ground_truth.shape[1])
    if n == 0 or k == 0:
        return {}

    pred = _normalize_rows(prediction[:n, :k], eps)
    gt = _normalize_rows(ground_truth[:n, :k], eps)
    names = (
        [str(value) for value in cell_types[:k]]
        if cell_types is not None
        else [str(index) for index in range(k)]
    )

    per_type = []
    for index in range(k):
        if np.std(gt[:, index]) <= eps:
            continue
        corr = _pearson(pred[:, index], gt[:, index], eps)
        per_type.append(0.0 if not np.isfinite(corr) else corr)

    pred_dominant = np.argmax(pred, axis=1)
    gt_dominant = np.argmax(gt, axis=1)
    pred_counts = np.bincount(pred_dominant, minlength=k)
    gt_counts = np.bincount(gt_dominant, minlength=k)

    src, dst = _undirected_edges(spatial_adjacency, n)
    if len(src):
        pred_boundary = pred_dominant[src] != pred_dominant[dst]
        gt_boundary = gt_dominant[src] != gt_dominant[dst]
        pred_boundary_fraction = float(np.mean(pred_boundary))
        gt_boundary_fraction = float(np.mean(gt_boundary))
        edge_mcc = _binary_mcc(pred_boundary, gt_boundary)
    else:
        pred_boundary_fraction = math.nan
        gt_boundary_fraction = math.nan
        edge_mcc = math.nan

    pyramidal = [
        index for index, name in enumerate(names)
        if "pyramidal" in name.lower()
    ]
    pyramidal_corr = (
        _pearson(pred[:, pyramidal].ravel(), gt[:, pyramidal].ravel(), eps)
        if pyramidal else math.nan
    )
    if not np.isfinite(pyramidal_corr):
        pyramidal_corr = 0.0

    gt_decon_std = float(np.std(gt))
    gt_max_prob = float(np.mean(np.max(gt, axis=1)))
    pred_decon_std = float(np.std(pred))
    pred_max_prob = float(np.mean(np.max(pred, axis=1)))

    return {
        "ST_Pearson_type_macro": (
            float(np.mean(per_type)) if per_type else 0.0),
        "ST_Pearson_type_median": (
            float(np.median(per_type)) if per_type else 0.0),
        "ST_Pearson_type_valid": int(len(per_type)),
        "ST_dominant_macro_f1": _dominant_macro_f1(
            pred_dominant, gt_dominant, k),
        "ST_active_dominant_types": int(np.sum(pred_counts > 0)),
        "ST_gt_active_dominant_types": int(np.sum(gt_counts > 0)),
        "ST_dominant_top_fraction": float(np.max(pred_counts) / n),
        "ST_gt_dominant_top_fraction": float(np.max(gt_counts) / n),
        "ST_edge_MCC": float(edge_mcc),
        "ST_boundary_fraction": pred_boundary_fraction,
        "ST_gt_boundary_fraction": gt_boundary_fraction,
        "ST_boundary_abs_error": float(abs(
            pred_boundary_fraction - gt_boundary_fraction)),
        "ST_same_dominant_abs_error": float(abs(
            (1.0 - pred_boundary_fraction)
            - (1.0 - gt_boundary_fraction))),
        "ST_pyramidal_layer_Pearson": float(pyramidal_corr),
        "ST_decon_std": pred_decon_std,
        "ST_gt_decon_std": gt_decon_std,
        "ST_decon_std_ratio": float(
            pred_decon_std / max(gt_decon_std, eps)),
        "ST_max_prob_mean": pred_max_prob,
        "ST_gt_max_prob_mean": gt_max_prob,
    }


def structural_validity(
    metrics: Mapping[str, float],
    unique_rows_ratio: float,
) -> Tuple[bool, Sequence[str]]:
    reasons = []
    gt_active = int(metrics.get("ST_gt_active_dominant_types", 0))
    min_active = max(10, int(math.floor(0.40 * gt_active)))
    active = int(metrics.get("ST_active_dominant_types", 0))
    if active < min_active:
        reasons.append("active_dominant_types")
    if float(metrics.get("ST_dominant_top_fraction", 1.0)) > 0.45:
        reasons.append("dominant_top_fraction")

    gt_std = float(metrics.get("ST_gt_decon_std", 0.0))
    pred_std = float(metrics.get("ST_decon_std", 0.0))
    if gt_std <= 0 or pred_std < 0.40 * gt_std:
        reasons.append("decon_std")
    if float(metrics.get("ST_Pearson_type_macro", 0.0)) <= 0:
        reasons.append("macro_type_pearson")
    edge_mcc = float(metrics.get("ST_edge_MCC", math.nan))
    if not np.isfinite(edge_mcc) or edge_mcc <= 0:
        reasons.append("edge_mcc")
    if float(unique_rows_ratio) < 0.05:
        reasons.append("unique_rows_ratio")
    return not reasons, tuple(reasons)


def calibration_confidence_valid(
    calibrated_max_prob: float,
    gt_max_prob: float,
    tolerance: float = 0.30,
) -> bool:
    calibrated_max_prob = float(calibrated_max_prob)
    gt_max_prob = float(gt_max_prob)
    tolerance = float(tolerance)
    if not all(np.isfinite([
            calibrated_max_prob, gt_max_prob, tolerance])):
        return False
    if gt_max_prob <= 0 or tolerance < 0:
        return False
    return abs(calibrated_max_prob - gt_max_prob) <= tolerance * gt_max_prob
