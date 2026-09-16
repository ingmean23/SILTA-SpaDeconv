#!/usr/bin/env python3
"""Reproduce the locked Melanoma ds6 fraction prediction from a DACG checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DACG_ROOT = RELEASE_ROOT / "dacg"
if str(DACG_ROOT) not in sys.path:
    sys.path.insert(0, str(DACG_ROOT))

from models.DACG_model import DACGModel  # noqa: E402
from stdatasets.datasets import StDataset  # noqa: E402


MODEL_DEFAULTS = {
    "decon_architecture": "concat",
    "spatial_gat_mode": "mixed",
    "coop_lambda_g": 0.5,
    "coop_lambda_x": 0.25,
    "coop_lambda_m": 0.0,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cfg_get(config: Mapping[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def evaluation_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(config))
    for key, value in MODEL_DEFAULTS.items():
        resolved.setdefault(key, value)
    dataset = (resolved.get("datasets") or [{}])[0]
    for key in MODEL_DEFAULTS:
        if key in dataset:
            resolved[key] = dataset[key]
    return resolved


def prediction_cell_types(pseudo_path: Path) -> list[str]:
    adata = sc.read_h5ad(pseudo_path, backed="r")
    try:
        names = [
            str(column)
            for column in adata.obs.columns
            if column not in ("cell_num", "n_counts", "n_genes")
        ]
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    if not names:
        raise ValueError(f"Pseudo file has no fraction columns: {pseudo_path}")
    return names


def build_model(config: Mapping[str, Any], num_genes: int, num_types: int) -> DACGModel:
    return DACGModel(
        num_genes=num_genes,
        num_cell_types=num_types,
        encoder_out_channels=[256, 256, 512],
        use_structured_latent=bool(cfg_get(config, "use_structured_latent", True)),
        structured_use_c_context=bool(cfg_get(config, "structured_use_c_context", False)),
        structured_h_dim=int(cfg_get(config, "structured_h_dim", 192)),
        structured_g_dim=int(cfg_get(config, "structured_g_dim", 192)),
        structured_mode_dim=int(cfg_get(config, "structured_mode_dim", 64)),
        structured_mode_tau=float(cfg_get(config, "structured_mode_tau", 1.5)),
        use_m_residual=bool(cfg_get(config, "use_m_residual", False)),
        m_residual_scale=float(cfg_get(config, "m_residual_scale", 0.05)),
        m_gate_init=float(cfg_get(config, "m_gate_init", 0.0)),
        use_mode_prior_head=bool(cfg_get(config, "use_mode_prior_head", False)),
        m_fusion_scale=float(cfg_get(config, "m_fusion_scale", 0.0)),
        mode_num_classes=num_types,
        spatial_gat_mode=str(cfg_get(config, "spatial_gat_mode", "hypergraph_attention")),
        spatial_expr_bias_weight=float(cfg_get(config, "spatial_expr_bias_weight", 0.0)),
        spatial_type_bias_weight=float(cfg_get(config, "spatial_type_bias_weight", 0.0)),
        decon_architecture=str(cfg_get(config, "decon_architecture", "coop_direct")),
        coop_fusion_dim=int(cfg_get(config, "coop_fusion_dim", 256)),
        coop_lambda_g=float(cfg_get(config, "coop_lambda_g", 1.0)),
        coop_lambda_x=float(cfg_get(config, "coop_lambda_x", 0.25)),
        coop_lambda_m=float(cfg_get(config, "coop_lambda_m", 0.0)),
        use_h_stochastic=bool(cfg_get(config, "use_h_stochastic", False)),
        h_logvar_min=float(cfg_get(config, "h_logvar_min", -6.0)),
        h_logvar_max=float(cfg_get(config, "h_logvar_max", 2.0)),
        h_inference_use_mu=bool(cfg_get(config, "h_inference_use_mu", True)),
        use_g_stochastic=bool(cfg_get(config, "use_g_stochastic", False)),
        g_logvar_min=float(cfg_get(config, "g_logvar_min", -6.0)),
        g_logvar_max=float(cfg_get(config, "g_logvar_max", 2.0)),
        g_inference_use_mu=bool(cfg_get(config, "g_inference_use_mu", True)),
        use_zig_latent=bool(cfg_get(config, "use_zig_latent", False)),
        zig_tau=float(cfg_get(config, "zig_tau", 1.0)),
        zig_pi_init=float(cfg_get(config, "zig_pi_init", 0.05)),
        zig_pi_min=float(cfg_get(config, "zig_pi_min", 0.01)),
        zig_pi_max=float(cfg_get(config, "zig_pi_max", 0.4)),
        zig_inference_use_expected=bool(cfg_get(config, "zig_inference_use_expected", True)),
        zig_scale_min=float(cfg_get(config, "zig_scale_min", 0.5)),
        zig_scale_max=float(cfg_get(config, "zig_scale_max", 1.5)),
        zig_residual_alpha=float(cfg_get(config, "zig_residual_alpha", 0.1)),
        kl_soft_cap=float(cfg_get(config, "kl_soft_cap", 100.0)),
        h_kl_soft_cap=float(cfg_get(config, "h_kl_soft_cap", 100.0)),
        g_kl_soft_cap=float(cfg_get(config, "g_kl_soft_cap", 100.0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(RELEASE_ROOT / "configs" / "inference_ds6_T_seed42_epoch2.json")
    )
    parser.add_argument(
        "--checkpoint",
        default=str(RELEASE_ROOT / "checkpoints" / "ds6_T_seed42_epoch2" / "model.model"),
    )
    parser.add_argument("--st", required=True, help="Melanoma ds6 st.h5ad")
    parser.add_argument("--pseudo", required=True, help="Melanoma ds6 PNP3 pseudo H5AD")
    parser.add_argument("--coordinates", required=True, help="Melanoma ds6 coordinate table")
    parser.add_argument("--output", required=True, help="Output fraction CSV")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--expected-sha256",
        default="cf7cde5149f00116c3d17ded16af2f3b00801688311b8807aa9e7ff12ecdb0c4",
        help="Expected byte-level hash for the reproduced CSV.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Allow numerically equivalent output when library versions change CSV bytes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    dataset_cfg = config["datasets"][0]
    dataset_cfg["real_st_path"] = str(Path(args.st).resolve())
    dataset_cfg["pseudo_st_path"] = str(Path(args.pseudo).resolve())
    dataset_cfg["real_location_path"] = str(Path(args.coordinates).resolve())

    pseudo_path = Path(args.pseudo).resolve()
    names = prediction_cell_types(pseudo_path)
    panel = json.loads(
        (RELEASE_ROOT / "configs" / "melanoma_input_only_literature_v2.json").read_text(
            encoding="utf-8"
        )
    )
    if names != list(map(str, panel["cell_types"])):
        raise ValueError(f"Cell-type order mismatch: pseudo={names}, expected={panel['cell_types']}")

    dataset = StDataset(
        data_path=str(Path(args.st).resolve()),
        location_path=str(Path(args.coordinates).resolve()),
        pseudo_st_path=str(pseudo_path),
        hvg=bool(config.get("hvg", True)),
        scale=bool(config.get("scale", True)),
        marker_path=config.get("marker_path"),
        spatial_dist=float(config.get("spatial_dist", 1.5)),
        k=int(config.get("expr_neighbors", 6)),
        sp_adj_path=dataset_cfg.get("sp_adj_path"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    x, ex_adj, sp_adj = next(iter(loader))
    device = torch.device(args.device)
    x = x.to(device)
    ex_adj = torch.squeeze(ex_adj.to(device)).fill_diagonal_(1.0)
    sp_adj = torch.squeeze(sp_adj.to(device)).fill_diagonal_(1.0)

    model_config = evaluation_model_config(config)
    model = build_model(model_config, len(dataset.final_genes), len(names))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint architecture mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.decon_temp.fill_(float(model_config.get("decon_temperature", 1.0)))
    model.to(device).eval()
    with torch.no_grad():
        prediction = model(x, [ex_adj, sp_adj], c=None, mode="st")["decon"]
    prediction = prediction.squeeze(0).detach().cpu().numpy().astype(np.float32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        prediction,
        index=list(dataset.filtered_barcodes),
        columns=names,
    )
    frame.index.name = "spot_id"
    frame.to_csv(output)
    digest = sha256_file(output)
    print(json.dumps({"prediction": str(output.resolve()), "sha256": digest}, indent=2))
    if not args.skip_hash_check and args.expected_sha256 and digest != args.expected_sha256:
        raise SystemExit(
            "Prediction hash mismatch. Check package versions, CUDA determinism, and input hashes: "
            f"expected={args.expected_sha256}, observed={digest}"
        )


if __name__ == "__main__":
    main()
