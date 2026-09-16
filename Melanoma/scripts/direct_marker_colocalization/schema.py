"""Configuration and locked marker-panel contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]


def canonical_hash(payload: dict[str, Any], hash_key: str) -> str:
    value = dict(payload)
    value.pop(hash_key, None)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert numpy and non-finite values into strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def resolve_path(value: str | Path, base: Path = REPO) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


@dataclass(frozen=True)
class MarkerPanel:
    path: Path
    payload: dict[str, Any]
    cell_types: list[str]
    markers_by_type: dict[str, list[str]]

    @property
    def genes(self) -> list[str]:
        return list(dict.fromkeys(
            gene for cell_type in self.cell_types
            for gene in self.markers_by_type[cell_type]
        ))


@dataclass(frozen=True)
class SliceSpec:
    name: str
    st: Path
    coordinates: Path | None
    st_counts: Path | None


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    payload: dict[str, Any]
    dataset_id: str
    panel: MarkerPanel
    slices: dict[str, SliceSpec]
    strict_invalid_score: float
    primary_cell_types: list[str]
    control_min_delta: float
    bootstrap_repeats: int
    bootstrap_grid_size: int


def load_panel(path: str | Path) -> MarkerPanel:
    path = resolve_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    hash_key = "direct_panel_sha256" if "direct_panel_sha256" in payload else "panel_sha256"
    expected = payload.get(hash_key)
    if expected and expected != canonical_hash(payload, hash_key):
        raise ValueError(f"Marker panel hash mismatch: {path}")
    cell_types = [str(value) for value in payload["cell_types"]]
    raw = payload["markers_by_cell_type"]
    markers: dict[str, list[str]] = {}
    for cell_type in cell_types:
        items = raw.get(cell_type, [])
        markers[cell_type] = [
            str(item["gene"] if isinstance(item, dict) else item) for item in items
        ]
        if not markers[cell_type]:
            raise ValueError(f"No marker genes for {cell_type}: {path}")
    return MarkerPanel(path, payload, cell_types, markers)


def load_config(path: str | Path) -> DatasetConfig:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"dataset_id", "panel", "slices"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Config misses keys {sorted(missing)}: {path}")
    panel = load_panel(resolve_path(payload["panel"]))
    slices: dict[str, SliceSpec] = {}
    for name, item in payload["slices"].items():
        slices[str(name)] = SliceSpec(
            name=str(name),
            st=resolve_path(item["st"]),
            coordinates=resolve_path(item["coordinates"]) if item.get("coordinates") else None,
            st_counts=resolve_path(item["st_counts"]) if item.get("st_counts") else None,
        )
    if not slices:
        raise ValueError("Config must contain at least one slice")
    strict = float(payload.get("strict_invalid_score", -1.0))
    if not -1.0 <= strict <= 1.0:
        raise ValueError("strict_invalid_score must be within [-1, 1]")
    primary_cell_types = [
        str(value) for value in payload.get("primary_cell_types", panel.cell_types)
    ]
    unknown_primary = sorted(set(primary_cell_types) - set(panel.cell_types))
    if unknown_primary:
        raise ValueError(f"Unknown primary cell types: {unknown_primary}")
    if not primary_cell_types:
        raise ValueError("primary_cell_types must not be empty")
    bootstrap_repeats = int(payload.get("bootstrap_repeats", 0))
    bootstrap_grid_size = int(payload.get("bootstrap_grid_size", 4))
    if bootstrap_repeats < 0 or bootstrap_grid_size < 2:
        raise ValueError("Invalid spatial bootstrap settings")
    return DatasetConfig(
        path=path,
        payload=payload,
        dataset_id=str(payload["dataset_id"]),
        panel=panel,
        slices=slices,
        strict_invalid_score=strict,
        primary_cell_types=primary_cell_types,
        control_min_delta=float(payload.get("control_min_delta", 0.0)),
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_grid_size=bootstrap_grid_size,
    )
