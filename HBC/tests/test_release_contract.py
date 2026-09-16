from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_model_contract() -> None:
    config = json.loads((ROOT / "configs/model_config.json").read_text())
    assert config["display_name"] == "SILTA"
    assert len(config["genes"]) == 4784
    assert len(config["cell_types"]) == 13
    assert config["architecture"]["decon_architecture"] == "coop_direct"
    assert config["architecture"]["direct_m_enabled"] is False
    assert config["architecture"]["readout_adapter_enabled"] is False
    assert config["architecture"]["c01_calibration_enabled"] is False


def test_checkpoint_hash() -> None:
    config = json.loads((ROOT / "configs/model_config.json").read_text())
    checkpoint = ROOT / config["checkpoint"]["file"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == config["checkpoint"]["sha256"]


def test_no_data_files_are_bundled() -> None:
    forbidden = {".h5ad", ".h5", ".loom", ".env", ".pem", ".key", ".p12", ".pfx"}
    assert not [path for path in ROOT.rglob("*") if path.is_file() and path.suffix.lower() in forbidden]
