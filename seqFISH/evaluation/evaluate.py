#!/usr/bin/env python
"""Evaluate SILTA fractions with the seqFISH Benchmark B metric contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from metrics import align_prediction, file_sha256, fraction_metrics, js_distance, read_matrix
from structural_metrics import osm_structural_metrics


def grid_adjacency(locations: pd.DataFrame) -> np.ndarray:
    coordinates = locations[["x", "y"]].to_numpy(dtype=float)
    grid = [(int(round(x - 0.5)), int(round(y - 0.5))) for x, y in coordinates]
    lookup = {coordinate: index for index, coordinate in enumerate(grid)}
    adjacency = np.zeros((len(grid), len(grid)), dtype=np.uint8)
    for index, (x, y) in enumerate(grid):
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            other = lookup.get(neighbor)
            if other is not None:
                adjacency[index, other] = 1
                adjacency[other, index] = 1
    return adjacency


def mean_neighbor_jsd(values: np.ndarray, adjacency: np.ndarray) -> float:
    source, target = np.where(np.triu(adjacency > 0, k=1))
    if not len(source):
        return math.nan
    return float(np.mean([
        js_distance(values[left], values[right])
        for left, right in zip(source, target)
    ]))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def rare_type_auprc(
    prediction: np.ndarray,
    truth: np.ndarray,
    type_counts: pd.DataFrame,
    cell_types: Sequence[str],
) -> tuple[float, Sequence[str]]:
    target_counts = type_counts.set_index("celltype")["target_cells"]
    total = float(target_counts.sum())
    rare = [
        name for name in cell_types
        if float(target_counts.get(name, 0.0)) / max(total, 1.0) <= 0.05
    ]
    scores = []
    for index, name in enumerate(cell_types):
        if name in rare:
            score = average_precision(truth[:, index] > 0, prediction[:, index])
            if math.isfinite(score):
                scores.append(score)
    return (float(np.mean(scores)) if scores else math.nan), rare


def structural_validity(metrics: Mapping[str, Any]) -> tuple[bool, Sequence[str]]:
    reasons = []
    if float(metrics["ST_max_prob_mean"]) < 0.25:
        reasons.append("flat_max_prob")
    if float(metrics["ST_decon_std_ratio"]) < 0.35:
        reasons.append("flat_decon_std")
    if float(metrics["ST_dominant_top_fraction"]) > 0.50:
        reasons.append("dominant_class_collapse")
    return not reasons, reasons


def evaluate(
    prediction_path: Path,
    truth_path: Path,
    locations_path: Path,
    type_counts_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    truth = read_matrix(truth_path)
    prediction = read_matrix(prediction_path)
    aligned, normalized_truth, alignment = align_prediction(prediction, truth)
    locations = pd.read_csv(locations_path, sep="\t")
    if len(locations) != len(truth):
        raise ValueError("Location and truth spot counts differ")
    adjacency = grid_adjacency(locations)
    pred = aligned.to_numpy(dtype=float)
    target = normalized_truth.to_numpy(dtype=float)
    primary = fraction_metrics(pred, target)
    structural = osm_structural_metrics(
        pred, target, adjacency, truth.columns.tolist())
    valid, reasons = structural_validity(structural)
    type_counts = pd.read_csv(type_counts_path, sep="\t")
    rare_auprc, rare_types = rare_type_auprc(
        pred, target, type_counts, truth.columns.tolist())
    pred_neighbor = mean_neighbor_jsd(pred, adjacency)
    truth_neighbor = mean_neighbor_jsd(target, adjacency)
    result = {
        "benchmark": "seqfish_cell_disjoint_b",
        "method": "SILTA",
        **primary,
        **structural,
        "neighbor_jsd_pred": pred_neighbor,
        "neighbor_jsd_truth": truth_neighbor,
        "neighbor_jsd_abs_error": abs(pred_neighbor - truth_neighbor),
        "macro_rare_type_auprc": rare_auprc,
        "rare_types": list(rare_types),
        "structural_valid": valid,
        "structural_invalid_reasons": list(reasons),
        "alignment": alignment,
        "truth_sha256": file_sha256(truth_path),
        "prediction_sha256": file_sha256(prediction_path),
    }
    return result, aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument("--type-counts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics, aligned = evaluate(
        args.prediction, args.truth, args.locations, args.type_counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(
        args.output_dir / "predictions_aligned.tsv.gz",
        sep="\t",
        compression="gzip",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    keys = ("ST_RMSE", "ST_JSD", "ST_SSIM", "ST_Pearson",
            "ST_edge_MCC", "ST_boundary_abs_error",
            "macro_rare_type_auprc", "structural_valid")
    print(json.dumps({key: metrics[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
