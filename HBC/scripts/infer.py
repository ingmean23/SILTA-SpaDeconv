#!/usr/bin/env python3
"""Run the locked SILTA HBC checkpoint on an HBC spatial transcriptomics slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.DACG_model import DACGModel  # noqa: E402


def dense_float32(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.nan_to_num(
        np.asarray(matrix, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def preprocess_expression(st_path: Path, genes: list[str]):
    import scanpy as sc
    adata = sc.read_h5ad(st_path)
    adata.var_names_make_unique()
    available = set(map(str, adata.var_names))
    missing = [gene for gene in genes if gene not in available]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"ST input is missing {len(missing)}/{len(genes)} checkpoint genes: {preview}"
        )

    already_processed = bool(
        dense_float32(adata.X).min() < 0 or "log1p" in adata.uns
    )
    if not already_processed:
        sc.pp.filter_cells(adata, min_counts=1)
    adata = adata[:, genes].copy()
    adata.X = dense_float32(adata.X)
    sc.pp.scale(adata, max_value=None, zero_center=True)
    adata.X = dense_float32(adata.X)
    return adata


def expression_adjacency(values: np.ndarray, neighbors: int = 6) -> np.ndarray:
    scaled = values.copy()
    mean = scaled.mean(axis=0, keepdims=True)
    std = scaled.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    scaled = np.nan_to_num((scaled - mean) / std)
    similarity = cosine_similarity(scaled, scaled)
    n_spots = len(scaled)
    k = min(max(int(neighbors), 1), n_spots)
    indices = torch.topk(torch.as_tensor(similarity), k=k, dim=1).indices.numpy()
    adjacency = np.zeros((n_spots, n_spots), dtype=np.float32)
    for row, columns in enumerate(indices):
        for column in columns:
            if row != column:
                adjacency[row, column] = 1.0
                adjacency[column, row] = 1.0

    target_degree = min(max(int(neighbors), 0), max(n_spots - 1, 0))
    np.fill_diagonal(adjacency, 0.0)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-8)
    for row in np.where(adjacency.sum(axis=1) < target_degree)[0]:
        scores = normalized[row] @ normalized.T
        scores[row] = -np.inf
        for column in np.argsort(scores)[::-1]:
            if adjacency[row].sum() >= target_degree:
                break
            if column != row and adjacency[row, column] == 0:
                adjacency[row, column] = 1.0
                adjacency[column, row] = 1.0
    return adjacency


def read_aligned_coordinates(metadata_path: Path, spot_ids: list[str]) -> np.ndarray:
    metadata = pd.read_csv(metadata_path, sep="\t")
    object_columns = [
        column for column in metadata.columns
        if pd.api.types.is_object_dtype(metadata[column])
    ]
    if object_columns:
        id_column = object_columns[0]
        metadata[id_column] = metadata[id_column].astype(str)
        metadata = metadata.set_index(id_column).reindex(spot_ids)
        if metadata.index.has_duplicates or metadata.isna().all(axis=1).any():
            raise ValueError("metadata does not cover every ST spot exactly once")
    else:
        if len(metadata) < len(spot_ids):
            raise ValueError("metadata has fewer rows than the ST matrix")
        metadata = metadata.iloc[: len(spot_ids)]

    coordinate_pairs = (
        ("x", "y"), ("coor_X", "coor_Y"), ("scaled_x", "scaled_y"),
        ("array_row", "array_col"), ("pxl_row", "pxl_col"),
        ("imagecol", "imagerow"),
    )
    columns = next(
        ([left, right] for left, right in coordinate_pairs
         if left in metadata.columns and right in metadata.columns),
        None,
    )
    if columns is None:
        numeric = metadata.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) < 2:
            raise ValueError("metadata has no usable coordinate columns")
        columns = numeric[-2:]
    coords = metadata[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(coords).all():
        raise ValueError("coordinates contain missing or non-finite values")
    return coords


def spatial_adjacency(coords: np.ndarray, distance_scale: float = 1.5) -> np.ndarray:
    from scipy.spatial.distance import cdist

    n_spots = len(coords)
    if n_spots <= 1:
        return np.zeros((n_spots, n_spots), dtype=np.float32)
    distances = cdist(coords, coords)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    positive = distances[np.isfinite(distances) & (distances > 0)]
    median = float(np.median(nearest)) if len(nearest) else 0.0
    if median <= 0 and len(positive):
        median = float(np.median(positive))
    threshold = distance_scale * median if median > 0 else distance_scale
    adjacency = (distances <= threshold).astype(np.float32)
    k = min(4, n_spots - 1)
    knn = np.argsort(distances, axis=1)[:, :k]
    adjacency[np.arange(n_spots)[:, None], knn] = 1.0
    return np.maximum(adjacency, adjacency.T)


def refine_expression_adjacency(
    x: torch.Tensor,
    adjacency: torch.Tensor,
    keep_ratio: float = 0.5,
    min_degree: int = 1,
) -> torch.Tensor:
    values = x.squeeze(0)
    values = values / values.norm(dim=1, keepdim=True).clamp_min(1e-8)
    source = adjacency > 0
    source.fill_diagonal_(False)
    refined = torch.zeros_like(adjacency)
    for row in range(source.shape[0]):
        candidates = source[row].nonzero(as_tuple=False).flatten()
        degree = int(candidates.numel())
        if degree == 0:
            continue
        keep = min(degree, max(int(np.ceil(keep_ratio * degree)), min_degree))
        scores = torch.matmul(values[candidates], values[row])
        selected = candidates[torch.topk(scores, k=keep).indices]
        refined[row, selected] = adjacency[row, selected]
    refined.fill_diagonal_(1.0)
    return refined


def build_model(config: dict) -> DACGModel:
    architecture = config["architecture"]
    return DACGModel(
        num_genes=len(config["genes"]),
        num_cell_types=len(config["cell_types"]),
        encoder_out_channels=[256, 256, 512],
        use_structured_latent=True,
        structured_use_c_context=False,
        structured_h_dim=architecture["structured_h_dim"],
        structured_g_dim=architecture["structured_g_dim"],
        structured_mode_dim=architecture["structured_mode_dim"],
        structured_mode_tau=architecture["structured_mode_tau"],
        use_m_residual=False,
        use_mode_prior_head=True,
        m_fusion_scale=0.0,
        mode_num_classes=architecture["mode_num_classes"],
        spatial_gat_mode=architecture["spatial_gat_mode"],
        spatial_expr_bias_weight=0.0,
        spatial_type_bias_weight=0.0,
        decon_architecture=architecture["decon_architecture"],
        coop_fusion_dim=architecture["coop_fusion_dim"],
        coop_lambda_g=architecture["coop_lambda_g"],
        coop_lambda_x=architecture["coop_lambda_x"],
        coop_lambda_m=architecture["coop_lambda_m"],
        use_h_stochastic=False,
        h_inference_use_mu=True,
        use_g_stochastic=False,
        g_inference_use_mu=True,
        use_zig_latent=False,
        kl_soft_cap=1000.0,
        h_kl_soft_cap=1000.0,
        g_kl_soft_cap=1000.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--st", required=True, type=Path, help="HBC ST .h5ad")
    parser.add_argument("--metadata", required=True, type=Path, help="HBC metadata.tsv")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "checkpoints" / "SILTA_HBC_TrackB_J0.model",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "model_config.json"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    adata = preprocess_expression(args.st, config["genes"])
    spot_ids = list(map(str, adata.obs_names))
    values = dense_float32(adata.X)
    ex_adj = expression_adjacency(values, config["preprocessing"]["expr_neighbors"])
    coords = read_aligned_coordinates(args.metadata, spot_ids)
    sp_adj = spatial_adjacency(coords, config["preprocessing"]["spatial_distance_scale"])

    device = torch.device(args.device)
    model = build_model(config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    x = torch.as_tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
    ex = torch.as_tensor(ex_adj, dtype=torch.float32, device=device)
    sp = torch.as_tensor(sp_adj, dtype=torch.float32, device=device)
    ex.fill_diagonal_(1.0)
    sp.fill_diagonal_(1.0)
    refined = refine_expression_adjacency(
        x, ex,
        keep_ratio=config["preprocessing"]["expression_refine_keep_ratio"],
        min_degree=config["preprocessing"]["expression_refine_min_degree"],
    )
    with torch.inference_mode():
        output = model(x, [refined, sp, ex], c=None, mode="st")
        prediction = output["decon"].squeeze(0).cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(prediction, columns=config["cell_types"])
    frame.insert(0, "spot_id", spot_ids)
    frame.to_csv(args.output, index=False)
    diagnostics = {
        "display_name": "SILTA",
        "n_spots": len(frame),
        "n_genes": len(config["genes"]),
        "n_cell_types": len(config["cell_types"]),
        "dominant_top_fraction": float(
            np.bincount(np.argmax(prediction, axis=1), minlength=prediction.shape[1]).max()
            / len(prediction)
        ),
        "decon_std": float(np.std(prediction, axis=0).mean()),
        "row_sum_max_error": float(np.abs(prediction.sum(axis=1) - 1.0).max()),
    }
    args.output.with_suffix(".diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()

