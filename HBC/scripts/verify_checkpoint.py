#!/usr/bin/env python3
"""Verify the published checkpoint checksum and strict model-state contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer import build_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = json.loads((ROOT / "configs" / "model_config.json").read_text())
    checkpoint = ROOT / "checkpoints" / "SILTA_HBC_TrackB_J0.model"
    actual = sha256(checkpoint)
    expected = config["checkpoint"]["sha256"]
    if actual != expected:
        raise SystemExit(f"checkpoint SHA256 mismatch: {actual} != {expected}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    print(json.dumps({
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "sha256": actual,
        "tensors": len(state),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }, indent=2))


if __name__ == "__main__":
    main()
