from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np
import torch

from silta_osm.calibration import apply_bundle, load_bundle
from silta_osm.checkpoint import inspect as inspect_checkpoint
from silta_osm.inference import ReleaseAssetError, predict_pre_c04, validate_assets
from silta_osm.model import state_compatibility
from silta_osm.prepared import PreparedInputs, load_prepared, save_prepared


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "silta_osm_b_fold42_seed42.json"
AGGREGATE = ROOT / "evidence" / "aggregate_3x3_metrics.json"


def test_config_paths_are_relative() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    for section in ("inputs", "artifacts"):
            assert all(not Path(value).is_absolute() for value in payload[section].values())


def test_all_release_json_is_free_of_machine_paths() -> None:
    forbidden = re.compile(
        r"(?i)([A-Z]:[\\/]|/" + "home1/|/" + "data1/|\bssh\b)"
    )

    def strings(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from strings(key)
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
        elif isinstance(value, str):
            yield value

    for path in (ROOT / "configs").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert not [value for value in strings(payload) if forbidden.search(value)], path


def test_missing_release_assets_fail_clearly() -> None:
    with pytest.raises(ReleaseAssetError, match="Missing required SILTA release assets"):
        validate_assets(CONFIG)


def test_registered_3x3_aggregate_is_exact() -> None:
    aggregate = json.loads(
        (ROOT / "evidence" / "aggregate_3x3_metrics.json").read_text(encoding="utf-8")
    )
    with (ROOT / "evidence" / "per_run_3x3_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    for metric in ("RMSE", "JSD", "SSIM", "PCC", "Edge_MCC", "boundary_abs_error", "rare_type_AUPRC"):
        observed = sum(float(row[metric]) for row in rows) / 9
        assert math.isclose(observed, aggregate["metrics"][metric], abs_tol=1e-12)


def test_representative_and_aggregate_are_explicitly_distinct() -> None:
    representative = json.loads(
        (ROOT / "expected_results" / "representative_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    assert representative["metrics"]["RMSE"] == 0.06405834734937287
    assert aggregate["metrics"]["RMSE"] == 0.0642916255909646
    assert "not the arithmetic mean" in representative["note"]


def test_release_artifacts_are_real_and_sanitized() -> None:
    checkpoint = ROOT / "checkpoint" / "silta_osm_b_fold42_seed42.model"
    calibration = ROOT / "calibration" / "C01_C04_state.pt"
    summary = inspect_checkpoint(checkpoint)
    assert summary["state_element_count"] == 9_985_052
    cell_types, _, _ = load_bundle(calibration)
    logits = pd.DataFrame(
        torch.zeros((2, len(cell_types))).numpy(), columns=cell_types
    )
    fractions = apply_bundle(logits, calibration)
    assert fractions.shape == (2, 31)
    assert fractions.to_numpy().sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    payload = torch.load(calibration, map_location="cpu", weights_only=True)
    assert payload["checkpoint"] == "../checkpoint/silta_osm_b_fold42_seed42.model"


def _synthetic_prepared() -> PreparedInputs:
    calibration = ROOT / "calibration" / "C01_C04_state.pt"
    cell_types, _, _ = load_bundle(calibration)
    rng = np.random.default_rng(17)
    spots, genes = 8, 33
    coordinates = np.stack(
        np.meshgrid(np.arange(4), np.arange(2)), axis=-1
    ).reshape(-1, 2).astype(np.float32)
    adjacency = np.eye(spots, dtype=np.float32)
    for index in range(spots - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    return PreparedInputs(
        expression=rng.random((spots, genes), dtype=np.float32),
        expression_adjacency=adjacency,
        spatial_adjacency=adjacency,
        coordinates=coordinates,
        reference_signature=rng.random((31, genes), dtype=np.float32),
        spot_ids=tuple(f"spot_{index}" for index in range(spots)),
        gene_names=tuple(f"gene_{index}" for index in range(genes)),
        cell_types=tuple(cell_types),
    ).validate()


def test_checkpoint_matches_bundled_model_strictly() -> None:
    result = state_compatibility(
        ROOT / "checkpoint" / "silta_osm_b_fold42_seed42.model"
    )
    assert result == {
        "strict": True,
        "model_state_keys": 323,
        "checkpoint_state_keys": 323,
    }


def test_prepared_roundtrip_and_real_checkpoint_forward(tmp_path: Path) -> None:
    prepared_path = tmp_path / "prepared.npz"
    save_prepared(prepared_path, _synthetic_prepared())
    prepared = load_prepared(prepared_path)
    outputs = predict_pre_c04(
        prepared,
        ROOT / "checkpoint" / "silta_osm_b_fold42_seed42.model",
        ROOT / "calibration" / "C01_C04_state.pt",
    )
    assert set(outputs) == {
        "logits", "checkpoint_fractions", "pre_c04_fractions"
    }
    for name, frame in outputs.items():
        assert frame.shape == (8, 31), name
        assert np.isfinite(frame.to_numpy()).all(), name
    for name in ("checkpoint_fractions", "pre_c04_fractions"):
        assert outputs[name].sum(axis=1).to_numpy() == pytest.approx(np.ones(8))


def test_inference_cli_runs_real_checkpoint() -> None:
    with tempfile.TemporaryDirectory(prefix=".inference-test-", dir=ROOT) as directory:
        work = Path(directory)
        prepared_path = work / "prepared.npz"
        save_prepared(prepared_path, _synthetic_prepared())
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))

        def relative(path: Path) -> str:
            return Path(os.path.relpath(path, work)).as_posix()

        payload["inputs"] = {"prepared_input": relative(prepared_path)}
        payload["artifacts"] = {
            "checkpoint": relative(
                ROOT / "checkpoint" / "silta_osm_b_fold42_seed42.model"),
            "calibration_state": relative(
                ROOT / "calibration" / "C01_C04_state.pt"),
            "calibration_parameters": relative(
                ROOT / "calibration" / "C04_parameters.json"),
            "logits": "checkpoint_logits.tsv.gz",
            "checkpoint_fractions": "checkpoint_fractions.tsv.gz",
            "pre_c04_fractions": "pre_c04_fractions.tsv.gz",
            "prediction": "silta_fractions.tsv.gz",
        }
        test_config = work / "config.json"
        test_config.write_text(json.dumps(payload), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_inference.py"),
                "--config",
                str(test_config),
                "--device",
                "cpu",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        for name in (
            "checkpoint_logits.tsv.gz",
            "checkpoint_fractions.tsv.gz",
            "pre_c04_fractions.tsv.gz",
            "silta_fractions.tsv.gz",
        ):
            assert (work / name).is_file()


def test_capsule_validator_passes_without_private_data() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_release.py"), "--allow-missing-data"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "release artifacts" in result.stdout
    assert "bundled and strictly checkpoint-compatible" in result.stdout


def test_security_audit_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "security_audit.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "PASS"' in result.stdout
