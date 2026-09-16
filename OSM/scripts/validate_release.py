from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from _bootstrap import ROOT
from silta_osm.calibration import load_bundle
from silta_osm.checkpoint import inspect as inspect_checkpoint
from silta_osm.inference import load_config, resolved_paths
from silta_osm.io import sha256
from silta_osm.model import state_compatibility


CONFIG = ROOT / "configs" / "silta_osm_b_fold42_seed42.json"
AGGREGATE = ROOT / "evidence" / "aggregate_3x3_metrics.json"
RUNS = ROOT / "evidence" / "per_run_3x3_metrics.csv"
PROVENANCE = ROOT / "configs" / "provenance.json"


def _validate_relative_config() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    for section in ("inputs", "artifacts"):
        for name, value in payload[section].items():
            if Path(value).is_absolute():
                raise ValueError(f"Absolute path in config: {section}.{name}")


def _validate_evidence() -> None:
    aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    with RUNS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9:
        raise ValueError(f"Expected nine registered runs, found {len(rows)}")
    mapping = {
        "RMSE": "RMSE",
        "JSD": "JSD",
        "SSIM": "SSIM",
        "PCC": "PCC",
        "Edge_MCC": "Edge_MCC",
        "boundary_abs_error": "boundary_abs_error",
        "rare_type_AUPRC": "rare_type_AUPRC",
    }
    for output_name, column in mapping.items():
        observed = sum(float(row[column]) for row in rows) / len(rows)
        expected = float(aggregate["metrics"][output_name])
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Aggregate mismatch for {output_name}: {observed} != {expected}")
    if not all(row["structural_valid"].lower() == "true" for row in rows):
        raise ValueError("One or more registered runs is structurally invalid")


def _validate_release_artifacts() -> None:
    config = load_config(CONFIG)
    paths = resolved_paths(CONFIG, config)
    required = ("checkpoint", "calibration_state", "calibration_parameters")
    missing = [name for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError("Missing release artifacts: " + ", ".join(missing))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    hashes = provenance["release_artifact_hashes"]
    for name in required:
        if sha256(paths[name]) != hashes[name]:
            raise ValueError(f"Release artifact hash mismatch: {name}")
    summary = inspect_checkpoint(paths["checkpoint"])
    if summary["state_element_count"] != int(config["expected"]["state_elements"]):
        raise ValueError("Checkpoint state element count does not match config")
    cell_types, _, _ = load_bundle(paths["calibration_state"])
    if len(cell_types) != int(config["expected"]["cell_types"]):
        raise ValueError("Calibration ontology does not match config")
    compatibility = state_compatibility(paths["checkpoint"])
    if not compatibility["strict"]:
        raise ValueError("Bundled model is not strictly checkpoint-compatible")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the SILTA OSM release capsule")
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Validate the redistributable capsule without private OSM inputs.",
    )
    args = parser.parse_args()
    _validate_relative_config()
    _validate_evidence()
    _validate_release_artifacts()
    config = load_config(CONFIG)
    paths = resolved_paths(CONFIG, config)
    missing_data = sorted(
        name for name in config["inputs"] if not paths[name].is_file()
    )
    if missing_data and not args.allow_missing_data:
        raise FileNotFoundError(
            "Missing user-supplied OSM inputs: " + ", ".join(missing_data)
        )
    print("PASS: release artifacts, config, calibration, and 3x3 evidence are valid")
    if missing_data:
        print("EXPECTED USER-SUPPLIED DATA: " + ", ".join(missing_data))
    print("FULL MODEL INFERENCE RUNTIME: bundled and strictly checkpoint-compatible")


if __name__ == "__main__":
    main()
