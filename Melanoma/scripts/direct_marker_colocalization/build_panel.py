"""Lock an existing SC-derived or literature marker map for direct evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .schema import canonical_hash, load_panel
except ImportError:
    from schema import canonical_hash, load_panel


def lock_source_panel(
    source: str | Path,
    output: str | Path,
    dataset_id: str,
    training_use_status: str = "unknown",
) -> dict[str, Any]:
    panel = load_panel(source)
    source_hash = panel.payload.get("panel_sha256", panel.payload.get("direct_panel_sha256"))
    payload: dict[str, Any] = {
        "protocol": "direct marker-fraction colocalization v1",
        "dataset_id": dataset_id,
        "selection_source": panel.payload.get("selection_source", "locked source panel"),
        "celltype_key": panel.payload.get("celltype_key", "celltype"),
        "cell_types": panel.cell_types,
        "markers_by_cell_type": {
            cell_type: [
                {
                    "gene": gene,
                    "source": "locked_source_panel",
                    "training_use_status": training_use_status,
                }
                for gene in panel.markers_by_type[cell_type]
            ]
            for cell_type in panel.cell_types
        },
        "marker_genes": panel.genes,
        "source_panel": str(panel.path),
        "source_panel_sha256": source_hash,
        "marker_genes_remain_in_model_input": True,
        "final_metric_uses_sc_signature": False,
        "final_metric_definition": "corr(predicted fraction, observed ST marker expression)",
    }
    payload["direct_panel_sha256"] = canonical_hash(payload, "direct_panel_sha256")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--training-use-status", choices=("input_only", "used_by_training", "unknown"),
        default="unknown",
    )
    args = parser.parse_args()
    payload = lock_source_panel(
        args.source_panel, args.output, args.dataset_id, args.training_use_status,
    )
    print(json.dumps({
        "output": args.output,
        "direct_panel_sha256": payload["direct_panel_sha256"],
        "cell_types": len(payload["cell_types"]),
        "marker_genes": len(payload["marker_genes"]),
    }, indent=2))


if __name__ == "__main__":
    main()

