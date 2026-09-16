from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def read_expression(path: Path | str, genes: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("expression table has duplicate identifiers")
    missing = [gene for gene in genes if gene not in frame.columns]
    if missing:
        raise ValueError(f"expression table is missing {len(missing)} genes: {missing[:8]}")
    frame = frame.loc[:, genes].apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("expression table contains non-finite values")
    if np.any(values < 0):
        raise ValueError("input must be unscaled non-negative expression")
    if np.any(values.sum(axis=1) <= 0):
        raise ValueError("input contains zero-sum spots")
    return frame


def read_coordinates(path: Path | str, spot_ids: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, sep="\t")
    if "spot_id" not in frame or not {"x", "y"}.issubset(frame.columns):
        raise ValueError("coordinates must contain spot_id, x, and y columns")
    frame["spot_id"] = frame["spot_id"].astype(str)
    if frame["spot_id"].duplicated().any():
        raise ValueError("coordinate table has duplicate spot identifiers")
    missing = sorted(set(spot_ids) - set(frame["spot_id"]))
    if missing:
        raise ValueError(f"coordinates are missing {len(missing)} spots")
    values = frame.set_index("spot_id").loc[spot_ids, ["x", "y"]]
    values = values.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("coordinates contain non-finite values")
    return values


def zscore_genes(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, ddof=1, keepdims=True)
    return np.divide(
        values - mean, std, out=np.zeros_like(values), where=std > 0
    ).astype(np.float32)


def _cosine(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-8)
    return normalized @ normalized.T


def expression_graph(values: np.ndarray, k: int = 6,
                     min_degree: int = 6) -> np.ndarray:
    scaled = zscore_genes(values)
    similarity = _cosine(scaled)
    n = len(scaled)
    topk = min(max(int(k), 1), n)
    indices = torch.topk(
        torch.from_numpy(similarity), k=topk, dim=1
    ).indices.numpy()
    adjacency = np.zeros((n, n), dtype=bool)
    neighbor_sets = [set(map(int, row)) for row in indices]
    for left in range(n):
        for right_value in indices[left]:
            right = int(right_value)
            if left in neighbor_sets[right]:
                adjacency[left, right] = True
                adjacency[right, left] = True
    np.fill_diagonal(adjacency, False)
    target = min(max(int(min_degree), 0), max(n - 1, 0))
    fallback_similarity = _cosine(np.asarray(values, dtype=np.float32))
    for left in np.flatnonzero(adjacency.sum(axis=1) < target):
        scores = fallback_similarity[left].copy()
        scores[left] = -np.inf
        for right in np.argsort(scores)[::-1]:
            if adjacency[left].sum() >= target:
                break
            if not adjacency[left, right]:
                adjacency[left, right] = True
                adjacency[right, left] = True
    return adjacency.astype(np.float32)


def pairwise_distance(coordinates: np.ndarray) -> np.ndarray:
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def spatial_graph(coordinates: np.ndarray, distance_scale: float = 1.5,
                  min_neighbors: int = 4) -> np.ndarray:
    distances = pairwise_distance(np.asarray(coordinates, dtype=np.float64))
    n = len(distances)
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    median = float(np.median(nearest))
    threshold = float(distance_scale) * median
    adjacency = distances <= threshold
    neighbors = np.argsort(distances, axis=1)[:, : min(min_neighbors, n - 1)]
    adjacency[np.arange(n)[:, None], neighbors] = True
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, False)
    return adjacency.astype(np.float32)


def normalized_distance(coordinates: np.ndarray,
                        spatial_adjacency: np.ndarray) -> tuple[np.ndarray, float]:
    distances = pairwise_distance(np.asarray(coordinates, dtype=np.float64))
    mask = spatial_adjacency > 0
    np.fill_diagonal(mask, False)
    edge_lengths = distances[mask & (distances > 0)]
    if not edge_lengths.size:
        raise ValueError("at least one positive spatial edge is required")
    scale = float(np.median(edge_lengths))
    return (distances / scale).astype(np.float32), scale


def build_inputs(expression: pd.DataFrame, coordinates: np.ndarray,
                 expression_k: int = 6, spatial_scale: float = 1.5
                 ) -> dict[str, np.ndarray | float]:
    scaled = zscore_genes(expression.to_numpy(dtype=np.float32))
    ex_adj = expression_graph(scaled, k=expression_k, min_degree=expression_k)
    sp_adj = spatial_graph(coordinates, distance_scale=spatial_scale)
    distance, distance_scale = normalized_distance(coordinates, sp_adj)
    np.fill_diagonal(ex_adj, 1.0)
    np.fill_diagonal(sp_adj, 1.0)
    return {
        "x": scaled,
        "ex_adj": ex_adj,
        "sp_adj": sp_adj,
        "distance": distance,
        "distance_scale": distance_scale,
    }
