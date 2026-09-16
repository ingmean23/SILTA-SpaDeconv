from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from .calibration import build_calibrator
from .model import SILTAInferenceModel
from .preprocess import build_inputs, read_coordinates, read_expression


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path | str, device: str = "cpu") -> tuple[
    Mapping[str, Any], SILTAInferenceModel, torch.nn.Module
]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") != "silta_moffitt_inference_checkpoint":
        raise ValueError("not a SILTA Moffitt inference checkpoint")
    model = SILTAInferenceModel(payload)
    model.load_active_state(payload["model_state_dict"])
    calibrator = build_calibrator(payload["calibrator"])
    return payload, model.to(device).eval(), calibrator.to(device).eval()


def predict(checkpoint_path: Path | str, expression_path: Path | str,
            coordinates_path: Path | str, output_path: Path | str,
            device: str = "cpu") -> Mapping[str, Any]:
    payload, model, calibrator = load_checkpoint(checkpoint_path, device)
    expression = read_expression(expression_path, list(payload["genes"]))
    spot_ids = list(expression.index)
    coordinates = read_coordinates(coordinates_path, spot_ids)
    graph = build_inputs(
        expression, coordinates,
        int(payload["preprocessing"]["expr_neighbors"]),
        float(payload["preprocessing"]["spatial_dist"]),
    )
    x = torch.from_numpy(graph["x"]).to(device)
    ex_adj = torch.from_numpy(graph["ex_adj"]).to(device)
    sp_adj = torch.from_numpy(graph["sp_adj"]).to(device)
    model.install_distance(torch.from_numpy(graph["distance"]).to(device))
    with torch.inference_mode():
        logits = model(x, ex_adj, sp_adj)
        logits = calibrator(logits)
        fractions = torch.softmax(
            logits / float(payload["temperature"]), dim=-1
        ).squeeze(0).cpu().numpy()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        fractions, index=spot_ids, columns=list(payload["cell_types"])
    )
    frame.to_csv(output, sep="\t", index_label="spot_id")
    report = {
        "model_name": "SILTA",
        "checkpoint_sha256": sha256(checkpoint_path),
        "expression_sha256": sha256(expression_path),
        "coordinates_sha256": sha256(coordinates_path),
        "prediction_sha256": sha256(output),
        "spots": len(frame),
        "genes": len(payload["genes"]),
        "cell_types": len(payload["cell_types"]),
        "distance_normalization_scale": float(graph["distance_scale"]),
        "device": str(device),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
