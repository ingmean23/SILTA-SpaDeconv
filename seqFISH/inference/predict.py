#!/usr/bin/env python
"""Run SILTA checkpoint inference on preprocessed seqFISH arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import seqfish_attention_matrix_runtime as attention_runtime
from runtime import seqfish_sota_component_runtime as route_runtime


def _strings(archive: np.lib.npyio.NpzFile, key: str, count: int, prefix: str):
    if key not in archive.files:
        return [f"{prefix}_{index:04d}" for index in range(count)]
    values = np.asarray(archive[key])
    if values.shape != (count,):
        raise ValueError(f"{key} must have shape ({count},)")
    return [str(value) for value in values.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=(
        ROOT / "configs" / "silta_seqfish_sp17_ms17_e005.json"))
    parser.add_argument("--temperature", type=float, default=0.525)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    archive = np.load(args.input_npz, allow_pickle=False)
    required = {"x", "expression_adjacency", "spatial_adjacency"}
    missing = sorted(required.difference(archive.files))
    if missing:
        raise KeyError(f"Missing NPZ arrays: {missing}")
    x = np.asarray(archive["x"], dtype=np.float32)
    ex_adj = np.asarray(archive["expression_adjacency"], dtype=np.float32)
    sp_adj = np.asarray(archive["spatial_adjacency"], dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("x must have shape [spots, genes]")
    spots, genes = x.shape
    if ex_adj.shape != (spots, spots) or sp_adj.shape != (spots, spots):
        raise ValueError("adjacency arrays must have shape [spots, spots]")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected_genes = int(state["branch1.0.query.weight"].shape[1])
    cell_type_count = int(
        state["cooperative_decon.decon_head.4.weight"].shape[0])
    if genes != expected_genes:
        raise ValueError(f"Checkpoint expects {expected_genes} genes, got {genes}")

    route_runtime.install(config)
    from models.DACG_model import DACGModel

    model = DACGModel(
        num_genes=genes,
        num_cell_types=cell_type_count,
        encoder_out_channels=[256, 256, 512],
        use_structured_latent=True,
        structured_h_dim=192,
        structured_g_dim=192,
        structured_mode_dim=64,
        structured_mode_tau=1.5,
        use_mode_prior_head=True,
        m_fusion_scale=0.0,
        mode_num_classes=cell_type_count,
        spatial_gat_mode="hypergraph_attention",
        decon_architecture="coop_gaussian",
        coop_fusion_dim=256,
        coop_lambda_g=0.55,
        coop_lambda_x=0.025,
        coop_lambda_m=0.0,
        use_h_stochastic=True,
        h_inference_use_mu=True,
        use_g_stochastic=False,
        g_inference_use_mu=True,
        use_zig_latent=False,
        kl_soft_cap=100.0,
        h_kl_soft_cap=100.0,
        g_kl_soft_cap=100.0,
    )
    attention_runtime.configure_attention(
        model, "dense", "learned", args.locations)
    model.cooperative_decon.m_semantic_gate = torch.nn.Linear(
        3 * model.cooperative_decon.fusion_dim,
        model.cooperative_decon.fusion_dim,
    ).to(next(model.parameters()).device)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))

    device = torch.device(args.device)
    model.to(device).eval()
    with torch.inference_mode():
        output = model(
            torch.from_numpy(x).unsqueeze(0).to(device),
            [torch.from_numpy(ex_adj).to(device),
             torch.from_numpy(sp_adj).to(device)],
            mode="st",
        )
        logits = output["decon_logits"].squeeze(0)
        fractions = torch.softmax(logits / args.temperature, dim=-1)
        fractions = fractions.cpu().numpy()

    spots_ids = _strings(archive, "spot_ids", spots, "spot")
    cell_types = _strings(archive, "cell_types", cell_type_count, "type")
    frame = pd.DataFrame(fractions, index=spots_ids, columns=cell_types)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, sep="\t")


if __name__ == "__main__":
    main()
