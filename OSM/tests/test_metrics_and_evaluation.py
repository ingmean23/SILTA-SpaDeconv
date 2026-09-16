from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from silta_osm.evaluation import evaluate_files, grid_adjacency, rare_type_auprc
from silta_osm.metrics import fraction_metrics


def test_identity_metrics_are_exact() -> None:
    values = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    metrics = fraction_metrics(values, values)
    assert metrics["ST_RMSE"] == 0.0
    assert metrics["ST_JSD"] == 0.0
    assert np.isclose(metrics["ST_SSIM"], 1.0)
    assert np.isclose(metrics["ST_Pearson"], 1.0)


def test_grid_adjacency_for_two_by_two_layout() -> None:
    locations = pd.DataFrame({"x": [0.5, 1.5, 0.5, 1.5], "y": [0.5, 0.5, 1.5, 1.5]})
    adjacency = grid_adjacency(locations)
    assert adjacency.shape == (4, 4)
    assert int(adjacency.sum() // 2) == 4


def test_file_evaluator_contract(tmp_path: Path) -> None:
    index = ["s0", "s1", "s2", "s3"]
    truth = pd.DataFrame(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
        index=index,
        columns=["A", "B"],
    )
    prediction = truth.copy()
    truth_path = tmp_path / "truth.tsv"
    prediction_path = tmp_path / "prediction.tsv"
    coordinates_path = tmp_path / "locations.tsv"
    truth.to_csv(truth_path, sep="\t", index_label="spot_id")
    prediction.to_csv(prediction_path, sep="\t", index_label="spot_id")
    pd.DataFrame({
        "spot_id": index,
        "x": [0.5, 1.5, 0.5, 1.5],
        "y": [0.5, 0.5, 1.5, 1.5],
    }).to_csv(coordinates_path, sep="\t", index=False)
    result = evaluate_files(prediction_path, truth_path, coordinates_path)
    assert result["ST_RMSE"] == 0.0
    assert result["structural_valid"] is True
    assert result["ST_edge_MCC"] == 1.0
    json.dumps(result)


def test_rare_type_requires_ten_positive_spots() -> None:
    truth = np.zeros((12, 2), dtype=float)
    truth[:10, 0] = 1.0
    truth[:9, 1] = 1.0
    prediction = truth.copy()
    counts = pd.DataFrame({
        "celltype": ["eligible", "too_few", "common"],
        "target_cells": [1, 1, 100],
    })
    score, eligible = rare_type_auprc(
        prediction, truth, counts, ["eligible", "too_few"]
    )
    assert eligible == ["eligible"]
    assert score == 1.0
