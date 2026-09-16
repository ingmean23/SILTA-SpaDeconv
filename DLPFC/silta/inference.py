"""Run the locked SILTA DLPFC model on one spatial slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .model import SILTAInferenceModel, load_inference_state
from .preprocessing import prepare_inputs


def load_model_config(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"num_genes", "cell_types", "architecture"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"model config misses keys: {sorted(missing)}")
    return payload


def build_model(payload: dict) -> SILTAInferenceModel:
    architecture = payload["architecture"]
    return SILTAInferenceModel(
        num_genes=int(payload["num_genes"]),
        num_cell_types=len(payload["cell_types"]),
        encoder_out_channels=tuple(architecture["encoder_out_channels"]),
        num_heads=int(architecture["num_heads"]),
        h_dim=int(architecture["h_dim"]),
        g_dim=int(architecture["g_dim"]),
        mode_dim=int(architecture["mode_dim"]),
        mode_num_classes=int(architecture["mode_num_classes"]),
        mode_tau=float(architecture["mode_tau"]),
        fusion_dim=int(architecture["fusion_dim"]),
        lambda_g=float(architecture["lambda_g"]),
        lambda_x=float(architecture["lambda_x"]),
        lambda_m=float(architecture["lambda_m"]),
    )


def run_inference(
    checkpoint: str | Path,
    model_config: str | Path,
    st: str | Path,
    coordinates: str | Path,
    output: str | Path,
    device: str = "cpu",
) -> Path:
    model_config = Path(model_config)
    config = load_model_config(model_config)
    gene_path = Path(config["gene_order"])
    if not gene_path.is_absolute():
        gene_path = model_config.parent / gene_path
    genes = json.loads(gene_path.read_text(encoding="utf-8"))
    if len(genes) != int(config["num_genes"]):
        raise ValueError("gene_order length does not match num_genes")
    spot_ids, expression, expression_graph, spatial_graph = prepare_inputs(
        st, coordinates, genes
    )
    torch_device = torch.device(device)
    model = build_model(config).to(torch_device)
    load_inference_state(checkpoint, model, map_location=torch_device)
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor(expression, dtype=torch.float32, device=torch_device).unsqueeze(0),
            torch.tensor(expression_graph, dtype=torch.float32, device=torch_device),
            torch.tensor(spatial_graph, dtype=torch.float32, device=torch_device),
        ).squeeze(0)
    frame = pd.DataFrame(
        prediction.cpu().numpy(), index=spot_ids, columns=config["cell_types"]
    )
    frame.index.name = "spot_id"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument("--st", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = run_inference(
        args.checkpoint,
        args.model_config,
        args.st,
        args.coordinates,
        args.output,
        args.device,
    )
    print(output)


if __name__ == "__main__":
    main()
