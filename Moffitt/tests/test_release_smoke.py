from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silta.inference import load_checkpoint, predict
from silta.metrics import fraction_metrics


CHECKPOINT = ROOT / "checkpoints" / "silta_moffitt_B-M1_seed42.pt"


def test_checkpoint_deserializes_safely() -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert payload["kind"] == "silta_moffitt_inference_checkpoint"
    assert payload["model_name"] == "SILTA"
    assert payload["verification"]["status"] == "matched"
    assert payload["verification"]["max_abs_error"] <= 1.0e-6
    assert len(payload["genes"]) == 135
    assert len(payload["cell_types"]) == 45
    _, model, calibrator = load_checkpoint(CHECKPOINT)
    assert model.training is False
    assert calibrator.training is False


def test_offline_inference_smoke(tmp_path: Path) -> None:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    random = np.random.default_rng(17)
    spots = [f"spot_{index}" for index in range(8)]
    expression = pd.DataFrame(
        random.gamma(2.0, 2.0, size=(len(spots), len(payload["genes"]))),
        index=spots,
        columns=payload["genes"],
    )
    coordinates = pd.DataFrame({
        "spot_id": spots,
        "x": [0, 1, 2, 3, 0, 1, 2, 3],
        "y": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    expression_path = tmp_path / "expression.tsv"
    coordinates_path = tmp_path / "coordinates.tsv"
    output_path = tmp_path / "prediction.tsv"
    expression.to_csv(expression_path, sep="\t", index_label="spot_id")
    coordinates.to_csv(coordinates_path, sep="\t", index=False)
    report = predict(
        CHECKPOINT, expression_path, coordinates_path, output_path, "cpu"
    )
    result = pd.read_csv(output_path, sep="\t", index_col=0)
    assert result.shape == (8, 45)
    assert np.allclose(result.sum(axis=1), 1.0, atol=1e-6)
    assert report["spots"] == 8
    assert json.loads(output_path.with_suffix(".manifest.json").read_text())["model_name"] == "SILTA"


def test_metric_contract() -> None:
    truth = np.array([[0.75, 0.25], [0.10, 0.90]], dtype=float)
    perfect = fraction_metrics(truth, truth)
    assert perfect["ST_RMSE"] == 0.0
    assert perfect["ST_JSD"] == 0.0
    assert np.isclose(perfect["ST_SSIM"], 1.0)
    assert np.isclose(perfect["ST_Pearson"], 1.0)
