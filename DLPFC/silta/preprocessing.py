"""Preprocessing and graph construction for the locked DLPFC SILTA model."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.spatial import distance as spatial_distance
from sklearn.metrics.pairwise import cosine_similarity


_COORDINATE_PAIRS = (
    ("x", "y"),
    ("coor_X", "coor_Y"),
    ("scaled_x", "scaled_y"),
    ("array_row", "array_col"),
    ("pxl_row", "pxl_col"),
    ("imagecol", "imagerow"),
)


def _dense(value) -> np.ndarray:
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def prepare_expression(st_path: str | Path, gene_order: list[str]):
    adata = sc.read_h5ad(st_path)
    adata.var_names_make_unique()
    values = _dense(adata.X)
    already_processed = bool(values.min() < 0 or "log1p" in adata.uns)
    if already_processed:
        sc.pp.filter_genes(adata, min_cells=1)
    else:
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.filter_genes(adata, min_cells=1)
    missing = [gene for gene in gene_order if gene not in adata.var_names]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"ST input misses {len(missing)} required genes; first missing: {preview}"
        )
    adata = adata[:, gene_order].copy()
    adata.X = np.nan_to_num(
        _dense(adata.X), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)
    sc.pp.scale(adata, max_value=None, zero_center=True)
    expression = np.nan_to_num(
        _dense(adata.X), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)
    return [str(value) for value in adata.obs_names], expression


def _mutual_cosine_graph(expression: np.ndarray, neighbors: int) -> np.ndarray:
    scaled = sc.AnnData(expression.copy())
    sc.pp.scale(scaled, max_value=None, zero_center=True)
    similarity = cosine_similarity(_dense(scaled.X), _dense(scaled.X))
    count = min(max(int(neighbors), 1), expression.shape[0])
    top = torch.topk(torch.tensor(similarity), k=count, dim=1).indices.numpy()
    directed = np.zeros(similarity.shape, dtype=bool)
    directed[np.arange(similarity.shape[0])[:, None], top] = True
    mutual = directed & directed.T
    np.fill_diagonal(mutual, False)
    return mutual.astype(np.float32)


def _ensure_min_neighbors(
    adjacency: np.ndarray,
    expression: np.ndarray,
    min_degree: int,
    chunk_size: int = 256,
) -> np.ndarray:
    graph = np.asarray(adjacency > 0, dtype=bool)
    node_count = graph.shape[0]
    if node_count <= 1:
        return graph.astype(np.float32)
    np.fill_diagonal(graph, False)
    target = min(max(int(min_degree), 0), node_count - 1)
    if target == 0 or np.all(graph.sum(axis=1) >= target):
        return graph.astype(np.float32)
    clean = np.nan_to_num(
        expression, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)
    normalized = clean / np.maximum(
        np.linalg.norm(clean, axis=1, keepdims=True), 1e-8
    )
    rows_needed = np.where(graph.sum(axis=1) < target)[0]
    for start in range(0, len(rows_needed), chunk_size):
        rows = rows_needed[start : start + chunk_size]
        similarities = normalized[rows] @ normalized.T
        for local_index, row in enumerate(rows):
            degree = int(graph[row].sum())
            similarities[local_index, row] = -np.inf
            for column in np.argsort(similarities[local_index])[::-1]:
                if degree >= target:
                    break
                if column == row or graph[row, column]:
                    continue
                graph[row, column] = True
                graph[column, row] = True
                degree = int(graph[row].sum())
    return graph.astype(np.float32)


def refine_expression_graph(
    expression: np.ndarray,
    adjacency: np.ndarray,
    keep_ratio: float = 0.5,
    min_degree: int = 1,
    symmetric: bool = False,
) -> np.ndarray:
    graph = torch.tensor(adjacency, dtype=torch.float32)
    graph.fill_diagonal_(1.0)
    features = torch.tensor(expression, dtype=torch.float32)
    node_count = graph.shape[0]
    diagonal = torch.eye(node_count, dtype=torch.bool)
    edge_mask = (graph > 0) & (~diagonal)
    normalized = torch.nn.functional.normalize(
        torch.nan_to_num(features), p=2, dim=-1, eps=1e-8
    )
    refined = torch.zeros_like(graph)
    for row in range(node_count):
        neighbors = edge_mask[row].nonzero(as_tuple=False).flatten()
        degree = int(neighbors.numel())
        if degree == 0:
            continue
        keep = min(
            degree,
            max(int(math.ceil(float(keep_ratio) * degree)), int(min_degree)),
        )
        scores = torch.matmul(normalized[neighbors], normalized[row])
        selected = neighbors[torch.topk(scores, k=keep, largest=True).indices]
        refined[row, selected] = graph[row, selected]
    if symmetric:
        refined = torch.maximum(refined, refined.t())
    refined.fill_diagonal_(1.0)
    return refined.numpy()


def build_expression_graph(
    expression: np.ndarray,
    neighbors: int = 6,
    refine: bool = True,
    keep_ratio: float = 0.5,
    refine_min_degree: int = 1,
    refine_symmetric: bool = False,
) -> np.ndarray:
    graph = _mutual_cosine_graph(expression, neighbors)
    graph = _ensure_min_neighbors(graph, expression, max(6, int(neighbors)))
    if refine:
        return refine_expression_graph(
            expression,
            graph,
            keep_ratio=keep_ratio,
            min_degree=refine_min_degree,
            symmetric=refine_symmetric,
        )
    np.fill_diagonal(graph, 1.0)
    return graph


def read_coordinates(
    coordinate_path: str | Path, spot_ids: list[str]
) -> np.ndarray:
    frame = pd.read_csv(coordinate_path, sep="\t")
    object_columns = [
        column for column in frame.columns if frame[column].dtype == object
    ]
    if object_columns:
        barcode = object_columns[0]
        if frame[barcode].astype(str).duplicated().any():
            raise ValueError("coordinate table contains duplicate spot identifiers")
        frame[barcode] = frame[barcode].astype(str)
        missing = [spot for spot in spot_ids if spot not in set(frame[barcode])]
        if missing:
            raise ValueError(f"coordinates miss {len(missing)} retained ST spots")
        frame = frame.set_index(barcode).loc[spot_ids].reset_index()
    else:
        if len(frame) < len(spot_ids):
            raise ValueError("coordinate table has fewer rows than retained ST spots")
        frame = frame.iloc[: len(spot_ids)].copy()
    columns = next(
        (
            pair
            for pair in _COORDINATE_PAIRS
            if pair[0] in frame.columns and pair[1] in frame.columns
        ),
        None,
    )
    if columns is None:
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) < 2:
            raise ValueError("no usable coordinate columns found")
        columns = tuple(numeric[-2:])
    coordinates = frame[list(columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    if coordinates.shape[0] != len(spot_ids) or not np.isfinite(coordinates).all():
        raise ValueError("coordinates are incomplete or non-finite after alignment")
    return coordinates


def build_spatial_graph(
    coordinates: np.ndarray, distance_scale: float = 1.5
) -> np.ndarray:
    node_count = int(coordinates.shape[0])
    spatial_k = min(4, max(node_count - 1, 0))
    if node_count <= 1 or spatial_k == 0:
        graph = np.zeros((node_count, node_count), dtype=np.float32)
        np.fill_diagonal(graph, 1.0)
        return graph
    distances = spatial_distance.squareform(spatial_distance.pdist(coordinates))
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    positive = distances[np.isfinite(distances) & (distances > 0)]
    median = float(np.median(nearest)) if len(nearest) else 0.0
    if median <= 0 and len(positive):
        median = float(np.median(positive))
    threshold = float(distance_scale) * median if median > 0 else float(distance_scale)
    graph = np.where(distances <= threshold, 1.0, 0.0).astype(np.float32)
    nearest_indices = np.argsort(distances, axis=1)[:, :spatial_k]
    graph[np.arange(node_count)[:, None], nearest_indices] = 1.0
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 1.0)
    return graph


def prepare_inputs(
    st_path: str | Path,
    coordinate_path: str | Path,
    gene_order: list[str],
):
    spot_ids, expression = prepare_expression(st_path, gene_order)
    expression_graph = build_expression_graph(expression, refine=False)
    coordinates = read_coordinates(coordinate_path, spot_ids)
    spatial_graph = build_spatial_graph(coordinates)
    return spot_ids, expression, expression_graph, spatial_graph
