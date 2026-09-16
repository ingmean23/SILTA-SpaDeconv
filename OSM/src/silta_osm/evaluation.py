from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from .io import align_prediction, read_fraction_table, sha256
from .metrics import fraction_metrics, structural_metrics, structural_validity


def grid_adjacency(locations: pd.DataFrame) -> np.ndarray:
    if not {"x", "y"}.issubset(locations.columns):
        raise ValueError("Locations must contain x and y columns")
    coordinates = locations[["x", "y"]].to_numpy(dtype=float)
    grid = [(int(round(x - 0.5)), int(round(y - 0.5))) for x, y in coordinates]
    lookup = {coordinate: index for index, coordinate in enumerate(grid)}
    adjacency = np.zeros((len(grid), len(grid)), dtype=np.uint8)
    for index, (x, y) in enumerate(grid):
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            other = lookup.get(neighbor)
            if other is not None:
                adjacency[index, other] = adjacency[other, index] = 1
    return adjacency


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def rare_type_auprc(
    prediction: np.ndarray,
    truth: np.ndarray,
    type_counts: Optional[pd.DataFrame],
    cell_types: list[str],
    min_positive_spots: int = 10,
) -> tuple[float, list[str]]:
    if type_counts is None:
        return float("nan"), []
    required = {"celltype", "target_cells"}
    if not required.issubset(type_counts.columns):
        raise ValueError(f"type_counts must contain {sorted(required)}")
    counts = type_counts.set_index("celltype")["target_cells"]
    total = max(float(counts.sum()), 1.0)
    rare = [name for name in cell_types if float(counts.get(name, 0)) / total <= 0.05]
    eligible = [
        name for index, name in enumerate(cell_types)
        if name in rare and int(np.sum(truth[:, index] > 0)) >= int(min_positive_spots)
    ]
    values = [
        _average_precision(truth[:, index] > 0, prediction[:, index])
        for index, name in enumerate(cell_types)
        if name in eligible
    ]
    finite = [value for value in values if np.isfinite(value)]
    return (float(np.mean(finite)) if finite else float("nan")), eligible


def evaluate_files(
    prediction_path: Path,
    truth_path: Path,
    coordinates_path: Path,
    type_counts_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    prediction = read_fraction_table(prediction_path)
    truth = read_fraction_table(truth_path)
    prediction, truth, alignment = align_prediction(prediction, truth)
    locations = pd.read_csv(coordinates_path, sep="\t")
    if "spot_id" in locations.columns:
        locations["spot_id"] = locations["spot_id"].astype(str)
        locations = locations.set_index("spot_id").reindex(truth.index).reset_index()
    if len(locations) != len(truth) or locations[["x", "y"]].isna().any().any():
        raise ValueError("Coordinates could not be aligned to truth spots")
    adjacency = grid_adjacency(locations)
    pred_values = prediction.to_numpy(dtype=float)
    truth_values = truth.to_numpy(dtype=float)
    result = dict(fraction_metrics(pred_values, truth_values))
    result.update(structural_metrics(
        pred_values, truth_values, adjacency, truth.columns.tolist()
    ))
    counts = pd.read_csv(type_counts_path, sep="\t") if type_counts_path else None
    rare_score, rare_types = rare_type_auprc(
        pred_values, truth_values, counts, truth.columns.tolist()
    )
    valid, reasons = structural_validity(result)
    result.update({
        "macro_rare_type_auprc": rare_score,
        "rare_type_positive_definition": "target_fraction > 0",
        "rare_type_min_positive_spots": 10,
        "rare_type_eligible_count": len(rare_types),
        "rare_type_eligible_types": rare_types,
        "structural_valid": valid,
        "structural_invalid_reasons": reasons,
        "alignment": dict(alignment),
        "prediction_sha256": sha256(prediction_path),
        "truth_sha256": sha256(truth_path),
        "coordinates_sha256": sha256(coordinates_path),
    })
    return result
