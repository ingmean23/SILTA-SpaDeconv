from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from silta_osm.figures import render_manifest


def test_figure_renderer_smoke(tmp_path: Path) -> None:
    spot_ids = ["s0", "s1", "s2", "s3"]
    truth = pd.DataFrame(
        [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]],
        index=spot_ids,
        columns=["A", "B"],
    )
    prediction = pd.DataFrame(
        [[0.8, 0.2], [0.65, 0.35], [0.3, 0.7], [0.15, 0.85]],
        index=spot_ids,
        columns=["A", "B"],
    )
    truth.to_csv(tmp_path / "truth.tsv", sep="\t", index_label="spot_id")
    prediction.to_csv(tmp_path / "prediction.tsv", sep="\t", index_label="spot_id")
    pd.DataFrame({
        "spot_id": spot_ids,
        "x": [0.5, 1.5, 0.5, 1.5],
        "y": [0.5, 0.5, 1.5, 1.5],
    }).to_csv(tmp_path / "locations.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"method": "SILTA", "RMSE": 0.1, "JSD": 0.2, "SSIM": 0.8, "PCC": 0.8}
    ]).to_csv(tmp_path / "metrics.csv", index=False)
    manifest = {
        "benchmark": "fixture",
        "ground_truth": "truth.tsv",
        "coordinates": "locations.tsv",
        "metrics": "metrics.csv",
        "output_dir": "figures",
        "primary_method": "SILTA",
        "methods": [{"name": "SILTA", "prediction": "prediction.tsv"}],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    outputs = render_manifest(manifest_path)
    expected = {
        "dominant_maps.png", "spot_pie_maps.png", "spatial_composite.png",
        "RMSE.png", "JSD.png",
        "SSIM.png", "PCC.png",
    }
    assert expected.issubset({path.name for path in outputs})
