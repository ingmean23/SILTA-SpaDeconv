from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


REQUIRED_ARRAYS = {
    "expression",
    "expression_adjacency",
    "spatial_adjacency",
    "coordinates",
    "reference_signature",
    "spot_ids",
    "gene_names",
    "cell_types",
}


def normalized_physical_distance(
    coordinates: np.ndarray, neighbors: int = 6
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [spots, 2]")
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=-1))
    count = min(max(1, int(neighbors)), max(1, len(coordinates) - 1))
    ranked = distance.copy()
    np.fill_diagonal(ranked, np.inf)
    local = np.sort(ranked, axis=1)[:, :count]
    eligible = local[np.isfinite(local) & (local > 0)]
    if not eligible.size:
        eligible = distance[distance > 0]
    if not eligible.size:
        raise ValueError("coordinates contain no positive pairwise distance")
    return (distance / float(np.median(eligible))).astype(np.float32)


@dataclass(frozen=True)
class PreparedInputs:
    expression: np.ndarray
    expression_adjacency: np.ndarray
    spatial_adjacency: np.ndarray
    coordinates: np.ndarray
    reference_signature: np.ndarray
    spot_ids: tuple[str, ...]
    gene_names: tuple[str, ...]
    cell_types: tuple[str, ...]

    def validate(self) -> "PreparedInputs":
        expression = np.asarray(self.expression)
        spots, genes = expression.shape if expression.ndim == 2 else (-1, -1)
        types = len(self.cell_types)
        expected = {
            "expression": (spots, genes),
            "expression_adjacency": (spots, spots),
            "spatial_adjacency": (spots, spots),
            "coordinates": (spots, 2),
            "reference_signature": (types, genes),
        }
        arrays = {
            "expression": expression,
            "expression_adjacency": np.asarray(self.expression_adjacency),
            "spatial_adjacency": np.asarray(self.spatial_adjacency),
            "coordinates": np.asarray(self.coordinates),
            "reference_signature": np.asarray(self.reference_signature),
        }
        if spots <= 1 or genes <= 0 or types <= 0:
            raise ValueError("prepared input dimensions must be positive")
        if len(self.spot_ids) != spots or len(self.gene_names) != genes:
            raise ValueError("identifier lengths do not match the numeric arrays")
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(
                    f"{name} has shape {arrays[name].shape}, expected {shape}"
                )
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in (
            "expression", "expression_adjacency", "spatial_adjacency",
            "reference_signature",
        ):
            if np.min(arrays[name]) < 0:
                raise ValueError(f"{name} contains negative values")
        for name, values in (
            ("spot_ids", self.spot_ids),
            ("gene_names", self.gene_names),
            ("cell_types", self.cell_types),
        ):
            if len(set(values)) != len(values) or any(not value for value in values):
                raise ValueError(f"{name} must contain unique non-empty strings")
        if np.any(self.reference_signature.sum(axis=1) <= 0):
            raise ValueError("every reference-signature row must have positive mass")
        return self

    def tensors(
        self, device: torch.device | str = "cpu"
    ) -> Mapping[str, torch.Tensor]:
        self.validate()
        expression_adjacency = np.asarray(
            self.expression_adjacency, dtype=np.float32).copy()
        spatial_adjacency = np.asarray(
            self.spatial_adjacency, dtype=np.float32).copy()
        np.fill_diagonal(expression_adjacency, 1.0)
        np.fill_diagonal(spatial_adjacency, 1.0)
        return {
            "expression": torch.as_tensor(
                np.asarray(self.expression, dtype=np.float32), device=device
            ).unsqueeze(0),
            "expression_adjacency": torch.as_tensor(
                expression_adjacency, device=device),
            "spatial_adjacency": torch.as_tensor(
                spatial_adjacency, device=device),
            "normalized_distance": torch.as_tensor(
                normalized_physical_distance(self.coordinates), device=device),
        }


def load_prepared(path: Path | str) -> PreparedInputs:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_ARRAYS.difference(archive.files))
        if missing:
            raise ValueError(f"prepared input is missing arrays: {missing}")
        value = PreparedInputs(
            expression=np.asarray(archive["expression"], dtype=np.float32),
            expression_adjacency=np.asarray(
                archive["expression_adjacency"], dtype=np.float32),
            spatial_adjacency=np.asarray(
                archive["spatial_adjacency"], dtype=np.float32),
            coordinates=np.asarray(archive["coordinates"], dtype=np.float32),
            reference_signature=np.asarray(
                archive["reference_signature"], dtype=np.float32),
            spot_ids=tuple(map(str, archive["spot_ids"].tolist())),
            gene_names=tuple(map(str, archive["gene_names"].tolist())),
            cell_types=tuple(map(str, archive["cell_types"].tolist())),
        )
    return value.validate()


def save_prepared(path: Path | str, value: PreparedInputs) -> None:
    value.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        expression=np.asarray(value.expression, dtype=np.float32),
        expression_adjacency=np.asarray(
            value.expression_adjacency, dtype=np.float32),
        spatial_adjacency=np.asarray(value.spatial_adjacency, dtype=np.float32),
        coordinates=np.asarray(value.coordinates, dtype=np.float32),
        reference_signature=np.asarray(
            value.reference_signature, dtype=np.float32),
        spot_ids=np.asarray(value.spot_ids, dtype=np.str_),
        gene_names=np.asarray(value.gene_names, dtype=np.str_),
        cell_types=np.asarray(value.cell_types, dtype=np.str_),
    )
