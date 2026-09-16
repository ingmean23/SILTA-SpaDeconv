"""Run formal input-only ranking and a separate full-panel sensitivity audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.direct_marker_colocalization.compare import compare_methods
from scripts.direct_marker_colocalization.schema import file_sha256, json_safe, load_config, load_panel


def validate_input_only_panel(primary_panel_path: str | Path, source_panel_path: str | Path) -> None:
    primary = load_panel(primary_panel_path)
    source = load_panel(source_panel_path)
    audit = primary.payload.get("training_use_audit", {})
    if primary.payload.get("panel_role") != "input_only_primary":
        raise ValueError("Primary panel must declare panel_role=input_only_primary")
    if audit.get("status") != "audited":
        raise ValueError("Primary panel lacks a completed training-use audit")
    if audit.get("source_panel_file_sha256") != file_sha256(source.path):
        raise ValueError("Primary panel was not derived from this full source panel")
    for cell_type in primary.cell_types:
        for item in primary.payload["markers_by_cell_type"][cell_type]:
            if not isinstance(item, dict) or item.get("training_use_status") != "input_only":
                raise ValueError(f"Unaudited primary marker in {cell_type}")


def _write_config(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def compare_dual_panels(
    config_path: str | Path,
    primary_panel_path: str | Path,
    predictions_manifest: str | Path,
    output_dir: str | Path,
    primary_method: str | None = None,
    render: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    primary_panel = load_panel(primary_panel_path)
    validate_input_only_panel(primary_panel.path, config.panel.path)
    output_dir = Path(output_dir).resolve()
    config_dir = output_dir / "protocol_configs"

    formal_payload = dict(config.payload)
    formal_payload["panel"] = str(primary_panel.path)
    formal_payload["primary_cell_types"] = primary_panel.cell_types
    formal_payload["benchmark_role"] = "formal_input_only_primary"
    sensitivity_payload = dict(config.payload)
    sensitivity_payload["panel"] = str(config.panel.path)
    sensitivity_payload["benchmark_role"] = "training_visible_full_panel_sensitivity"
    formal_config = _write_config(formal_payload, config_dir / "formal_input_only.json")
    sensitivity_config = _write_config(
        sensitivity_payload, config_dir / "full_panel_sensitivity.json"
    )

    formal_summary = compare_methods(
        formal_config, predictions_manifest, output_dir / "formal_input_only",
        primary_method, render,
    )
    sensitivity_summary = compare_methods(
        sensitivity_config, predictions_manifest, output_dir / "full_panel_sensitivity",
        primary_method, render,
    )
    formal = pd.read_csv(output_dir / "formal_input_only" / "method_ranking.csv")
    sensitivity = pd.read_csv(output_dir / "full_panel_sensitivity" / "method_ranking.csv")
    formal = formal.add_prefix("formal_").rename(columns={"formal_method": "method"})
    sensitivity = sensitivity.add_prefix("sensitivity_").rename(
        columns={"sensitivity_method": "method"}
    )
    combined = formal.merge(sensitivity, on="method", how="outer")
    combined = combined.sort_values("formal_rank").reset_index(drop=True)
    combined.to_csv(output_dir / "dual_panel_method_ranking.csv", index=False)

    summary = json_safe({
        "protocol": "direct marker-fraction colocalization dual-panel v1",
        "formal_ranking_panel": "input_only_primary",
        "sensitivity_panel": "full_panel_training_visible",
        "formal_primary_cell_types": primary_panel.cell_types,
        "formal_summary": formal_summary,
        "sensitivity_summary": sensitivity_summary,
        "warning": "The full panel is sensitivity evidence and must not determine the formal rank.",
    })
    (output_dir / "dual_panel_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--primary-panel", required=True)
    parser.add_argument("--predictions-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-method")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    summary = compare_dual_panels(
        args.config,
        args.primary_panel,
        args.predictions_manifest,
        args.output_dir,
        args.primary_method,
        args.render,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
