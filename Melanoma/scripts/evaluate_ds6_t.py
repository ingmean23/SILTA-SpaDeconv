#!/usr/bin/env python3
"""Evaluate and render a Melanoma ds6 fraction table with the frozen G5 contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from scripts.direct_marker_colocalization.evaluate import evaluate_prediction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--st", required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="DACG_ds6_T_C0_e2_seed42")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    template = json.loads(
        (RELEASE_ROOT / "configs" / "marker_evidence_ds6.template.json").read_text(
            encoding="utf-8"
        )
    )
    template["panel"] = str(
        (RELEASE_ROOT / "configs" / "melanoma_input_only_literature_v2.json").resolve()
    )
    template["slices"] = {
        "ds6": {
            "st": str(Path(args.st).resolve()),
            "st_counts": str(Path(args.counts).resolve()),
            "coordinates": str(Path(args.coordinates).resolve()),
        }
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = output_dir / "runtime_marker_evidence_ds6.json"
    runtime_config.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")

    summary = evaluate_prediction(
        runtime_config,
        "ds6",
        args.method,
        args.prediction,
        output_dir,
        render=not args.no_render,
        controls=False,
    )
    print(json.dumps({
        "summary": str((output_dir / "summary.json").resolve()),
        "per_cell_type": str((output_dir / "per_cell_type.csv").resolve()),
        "prediction_sha256": summary["prediction_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
