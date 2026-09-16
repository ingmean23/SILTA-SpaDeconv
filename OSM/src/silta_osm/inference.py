from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .calibration import apply_bundle, apply_c01, load_bundle
from .io import resolve_relative, write_fraction_table
from .model import load_model
from .prepared import PreparedInputs, load_prepared


class ReleaseAssetError(RuntimeError):
    pass


def load_config(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolved_paths(config_path: Path, config: Mapping[str, Any]) -> Mapping[str, Path]:
    values = {}
    for section in ("inputs", "artifacts"):
        for name, value in config[section].items():
            values[name] = resolve_relative(config_path, str(value))
    return values


def validate_assets(
    config_path: Path, allow_missing: bool = False
) -> Mapping[str, Path]:
    config = load_config(config_path)
    paths = resolved_paths(config_path, config)
    required = ("prepared_input", "checkpoint", "calibration_state")
    missing = [f"{name}: {paths[name]}" for name in required if not paths[name].is_file()]
    if missing and not allow_missing:
        raise ReleaseAssetError(
            "Missing required SILTA release assets:\n  " + "\n  ".join(missing)
        )
    return paths


def _validate_endpoint_contract(
    prepared: PreparedInputs,
    config: Mapping[str, Any],
    calibration_types: list[str],
) -> None:
    expected = config["expected"]
    if prepared.expression.shape[1] != int(expected["genes"]):
        raise ValueError("prepared gene count does not match the endpoint")
    if len(prepared.cell_types) != int(expected["cell_types"]):
        raise ValueError("prepared cell-type count does not match the endpoint")
    if list(prepared.cell_types) != list(calibration_types):
        raise ValueError("prepared cell-type order does not match calibration")
    if prepared.reference_signature.shape != (
        int(expected["cell_types"]), int(expected["genes"])
    ):
        raise ValueError("reference signature does not match the endpoint")


def predict_pre_c04(
    prepared: PreparedInputs,
    checkpoint: Path | str,
    calibration_state: Path | str,
    device: torch.device | str = "cpu",
) -> Mapping[str, pd.DataFrame]:
    calibration_types, _, _ = load_bundle(Path(calibration_state), str(device))
    if list(prepared.cell_types) != calibration_types:
        raise ValueError("prepared cell-type order does not match calibration")
    model = load_model(checkpoint, device)
    tensors = prepared.tensors(device)
    with torch.no_grad():
        output = model(**tensors)
    logits_array = output["logits"].squeeze(0).cpu().numpy()
    fraction_array = output["fractions"].squeeze(0).cpu().numpy()
    if not np.isfinite(logits_array).all() or not np.isfinite(fraction_array).all():
        raise ValueError("SILTA inference produced non-finite outputs")
    logits = pd.DataFrame(
        logits_array, index=prepared.spot_ids, columns=calibration_types)
    checkpoint_fractions = pd.DataFrame(
        fraction_array, index=prepared.spot_ids, columns=calibration_types)
    pre_c04 = apply_c01(logits, Path(calibration_state))
    return {
        "logits": logits,
        "checkpoint_fractions": checkpoint_fractions,
        "pre_c04_fractions": pre_c04,
    }


def run(
    config_path: Path,
    device: str = "cpu",
    apply_c04: bool = True,
) -> Mapping[str, Path]:
    config = load_config(config_path)
    paths = validate_assets(config_path)
    prepared = load_prepared(paths["prepared_input"])
    cell_types, _, _ = load_bundle(paths["calibration_state"])
    _validate_endpoint_contract(prepared, config, cell_types)
    outputs = predict_pre_c04(
        prepared, paths["checkpoint"], paths["calibration_state"], device)

    paths["logits"].parent.mkdir(parents=True, exist_ok=True)
    outputs["logits"].to_csv(
        paths["logits"], sep="\t", compression="gzip", index_label="spot_id")
    write_fraction_table(paths["checkpoint_fractions"], outputs["checkpoint_fractions"])
    write_fraction_table(paths["pre_c04_fractions"], outputs["pre_c04_fractions"])
    written = {
        "logits": paths["logits"],
        "checkpoint_fractions": paths["checkpoint_fractions"],
        "pre_c04_fractions": paths["pre_c04_fractions"],
    }
    if apply_c04:
        final = apply_bundle(outputs["logits"], paths["calibration_state"])
        write_fraction_table(paths["prediction"], final)
        written["prediction"] = paths["prediction"]
    return written
