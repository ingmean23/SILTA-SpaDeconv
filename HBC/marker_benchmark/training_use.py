"""Audit marker training use and build an input-only primary panel."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marker_benchmark.schema import (
    canonical_hash,
    file_sha256,
    load_panel,
)


MARKER_ROLES = ("train_markers", "selector_markers")


def _gene(item: Any) -> str:
    return str(item["gene"] if isinstance(item, dict) else item)


def _used_markers(manifest: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    cell_types = manifest.get("cell_types")
    if not isinstance(cell_types, dict) or not cell_types:
        raise ValueError("Training manifest must contain a non-empty cell_types mapping")
    used: dict[str, list[dict[str, str]]] = defaultdict(list)
    recognized = False
    for cell_type, entry in cell_types.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid marker entry for {cell_type}")
        for role in MARKER_ROLES:
            values = entry.get(role, [])
            if values:
                recognized = True
            for item in values:
                used[_gene(item)].append({"cell_type": str(cell_type), "role": role})
    if not recognized:
        raise ValueError("Training manifest contains no train_markers or selector_markers")
    return dict(used)


def build_input_only_panel(
    source_panel_path: str | Path,
    training_manifest_path: str | Path,
    output_panel_path: str | Path,
    audit_output_path: str | Path,
    min_markers_per_type: int = 3,
) -> tuple[Path, Path]:
    if min_markers_per_type < 1:
        raise ValueError("min_markers_per_type must be positive")
    source_panel = load_panel(source_panel_path)
    manifest_path = Path(training_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    used = _used_markers(manifest)

    retained: dict[str, list[dict[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    ineligible: dict[str, dict[str, Any]] = {}
    raw_markers = source_panel.payload["markers_by_cell_type"]
    for cell_type in source_panel.cell_types:
        kept = []
        for item in raw_markers[cell_type]:
            gene = _gene(item)
            if gene in used:
                excluded.append({
                    "panel_cell_type": cell_type,
                    "gene": gene,
                    "training_uses": used[gene],
                })
                continue
            value = dict(item) if isinstance(item, dict) else {"gene": gene}
            value["training_use_status"] = "input_only"
            kept.append(value)
        if len(kept) >= min_markers_per_type:
            retained[cell_type] = kept
        else:
            ineligible[cell_type] = {
                "retained_marker_count": len(kept),
                "required_marker_count": min_markers_per_type,
                "retained_genes": [_gene(item) for item in kept],
            }
    if not retained:
        raise ValueError("No cell type retains enough input-only markers")

    payload = {
        "protocol": "direct marker-fraction colocalization v1",
        "dataset_id": source_panel.payload.get("dataset_id"),
        "panel_role": "input_only_primary",
        "selection_source": source_panel.payload.get("selection_source"),
        "celltype_key": source_panel.payload.get("celltype_key"),
        "cell_types": list(retained),
        "markers_by_cell_type": retained,
        "training_use_audit": {
            "status": "audited",
            "definition": "genes absent from marker-loss and selector lists in the exact locked training manifest",
            "source_panel_file_sha256": file_sha256(source_panel.path),
            "training_manifest_file_sha256": file_sha256(manifest_path),
            "min_markers_per_type": min_markers_per_type,
            "excluded_gene_count": len({row["gene"] for row in excluded}),
            "eligible_cell_type_count": len(retained),
            "ineligible_cell_types": list(ineligible),
        },
    }
    payload["direct_panel_sha256"] = canonical_hash(payload, "direct_panel_sha256")

    output_panel = Path(output_panel_path).resolve()
    audit_output = Path(audit_output_path).resolve()
    output_panel.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    output_panel.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    audit = {
        "status": "PASS",
        "source_panel": str(source_panel.path),
        "source_panel_file_sha256": file_sha256(source_panel.path),
        "training_manifest": str(manifest_path),
        "training_manifest_file_sha256": file_sha256(manifest_path),
        "output_panel": str(output_panel),
        "output_panel_sha256": payload["direct_panel_sha256"],
        "eligible_cell_types": list(retained),
        "ineligible_cell_types": ineligible,
        "excluded_markers": excluded,
        "retained_markers_by_cell_type": {
            cell_type: [_gene(item) for item in items]
            for cell_type, items in retained.items()
        },
    }
    audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return output_panel, audit_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output-panel", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--min-markers-per-type", type=int, default=3)
    args = parser.parse_args()
    outputs = build_input_only_panel(
        args.source_panel,
        args.training_manifest,
        args.output_panel,
        args.audit_output,
        args.min_markers_per_type,
    )
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
