from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


EPS = 1e-12


def resolve_relative(base_file: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Release paths must be relative: {value}")
    return (base_file.resolve().parent / path).resolve()


def read_fraction_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    separator = "\t" if ".tsv" in path.name or path.suffix == ".txt" else ","
    frame = pd.read_csv(path, sep=separator, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate spot identifiers in {path}")
    if frame.columns.has_duplicates:
        raise ValueError(f"Duplicate cell-type names in {path}")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite fractions in {path}")
    if np.min(values) < -1e-10:
        raise ValueError(f"Negative fractions in {path}")
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= EPS):
        raise ValueError(f"Zero-sum fraction rows in {path}")
    values /= totals
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def align_prediction(
    prediction: pd.DataFrame, truth: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    unknown_spots = prediction.index.difference(truth.index).tolist()
    missing_spots = truth.index.difference(prediction.index).tolist()
    unknown_types = prediction.columns.difference(truth.columns).tolist()
    missing_types = truth.columns.difference(prediction.columns).tolist()
    if unknown_spots:
        raise ValueError(f"Unknown prediction spots: {unknown_spots[:5]}")
    if missing_spots:
        raise ValueError(f"Missing prediction spots: {missing_spots[:5]}")
    if unknown_types:
        raise ValueError(f"Unknown prediction cell types: {unknown_types[:5]}")
    aligned = prediction.reindex(
        index=truth.index, columns=truth.columns, fill_value=0.0
    )
    values = aligned.to_numpy(dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    values = np.divide(values, totals, out=np.zeros_like(values), where=totals > EPS)
    aligned = pd.DataFrame(values, index=truth.index, columns=truth.columns)
    return aligned, truth, {
        "missing_cell_types_filled_zero": missing_types,
        "prediction_zero_rows": int((values.sum(axis=1) <= EPS).sum()),
    }


def write_fraction_table(path: Path | str, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(path, sep="\t", compression=compression, index_label="spot_id")


def write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_text_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()

