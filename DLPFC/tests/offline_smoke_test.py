"""Offline end-to-end smoke test for the SILTA DLPFC release."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.direct_marker_colocalization.evaluate import evaluate_prediction
from plotting.marker_comparison.plot import render
from silta.inference import build_model, load_model_config
from silta.model import load_inference_state


def main() -> None:
    config = load_model_config(ROOT / "configs" / "model.json")
    model = build_model(config)
    load_inference_state(
        ROOT / "checkpoint" / "silta_dlpfc_mix5_best_single_seed.model",
        model,
    )
    model.eval()
    spot_count = 6
    generator = torch.Generator().manual_seed(17)
    expression = torch.randn(
        1, spot_count, int(config["num_genes"]), generator=generator
    )
    expression_graph = torch.eye(spot_count)
    spatial_graph = torch.eye(spot_count)
    with torch.no_grad():
        prediction = model(expression, expression_graph, spatial_graph).squeeze(0)
    assert prediction.shape == (spot_count, len(config["cell_types"]))
    assert torch.allclose(
        prediction.sum(dim=1), torch.ones(spot_count), atol=1e-6
    )

    panel = json.loads(
        (ROOT / "configs" / "marker_panel.json").read_text(encoding="utf-8")
    )
    genes = list(dict.fromkeys(
        item["gene"] if isinstance(item, dict) else item
        for cell_type in panel["cell_types"]
        for item in panel["markers_by_cell_type"][cell_type]
    ))
    spots = [f"spot_{index}" for index in range(spot_count)]
    rng = np.random.default_rng(17)
    counts = rng.poisson(4.0, size=(spot_count, len(genes))).astype(np.float32)
    counts[0] += 1
    coordinates = pd.DataFrame({
        "spot_id": spots,
        "x": np.arange(spot_count) % 3,
        "y": np.arange(spot_count) // 3,
    })
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        st_path = temp / "st.h5ad"
        coordinate_path = temp / "coordinates.tsv"
        prediction_path = temp / "prediction.csv"
        config_path = temp / "evaluation.json"
        output_dir = temp / "evaluation"
        ad.AnnData(
            X=counts,
            obs=pd.DataFrame(index=spots),
            var=pd.DataFrame(index=genes),
        ).write_h5ad(st_path)
        coordinates.to_csv(coordinate_path, sep="\t", index=False)
        pd.DataFrame(
            prediction.numpy(), index=spots, columns=config["cell_types"]
        ).rename_axis("spot_id").to_csv(prediction_path)
        config_path.write_text(json.dumps({
            "dataset_id": "synthetic",
            "panel": str((ROOT / "configs" / "marker_panel.json").resolve()),
            "strict_invalid_score": -1.0,
            "primary_cell_types": panel["cell_types"],
            "control_seed": 17,
            "bootstrap_repeats": 0,
            "bootstrap_grid_size": 4,
            "slices": {
                "smoke": {
                    "st": str(st_path),
                    "coordinates": str(coordinate_path),
                }
            },
        }), encoding="utf-8")
        summary = evaluate_prediction(
            config_path,
            "smoke",
            "SILTA",
            prediction_path,
            output_dir,
            render=False,
            controls=False,
        )
        assert summary["n_cell_types"] == 33
        metric_path = output_dir / "per_cell_type.csv"
        figures = render(
            st_path,
            coordinate_path,
            ROOT / "configs" / "marker_panel.json",
            "Mix_5",
            [("SILTA", prediction_path)],
            [("SILTA", metric_path)],
            temp / "figures",
        )
        assert all(path.exists() and path.stat().st_size > 0 for path in figures)
    print("offline smoke test: PASS")


if __name__ == "__main__":
    main()
