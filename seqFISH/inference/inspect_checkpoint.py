#!/usr/bin/env python
"""Safely inspect a tensor-only SILTA checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise TypeError("Expected a non-empty tensor state_dict")
    non_tensors = [key for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensors:
        raise TypeError(f"Checkpoint contains non-tensors: {non_tensors[:5]}")
    report = {
        "path": str(args.checkpoint),
        "sha256": sha256(args.checkpoint),
        "tensor_count": len(state),
        "num_genes": int(state["branch1.0.query.weight"].shape[1]),
        "num_cell_types": int(
            state["cooperative_decon.decon_head.4.weight"].shape[0]),
        "mode_classes": int(state["mode_embedding"].shape[0]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
