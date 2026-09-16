#!/usr/bin/env python3
"""Evaluate a regenerated fixed HBC candidate pool with nested Track-B selection.

This re-runs selection for the supplied candidates. It does not reconstruct a
larger historical search pool whose predictions or checkpoints are absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.hbc_pathology_unified_sota_v1 import protocol  # noqa: E402


def candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be ID=/path/to/prediction.csv")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", action="append", required=True, type=candidate,
        help="Repeat as ID=/path/to/prediction.csv for each locked endpoint.",
    )
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument(
        "--protocol", type=Path,
        default=ROOT / "configs" / "pathology_protocol.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    predictions = dict(args.candidate)
    settings = json.loads(args.protocol.read_text(encoding="utf-8"))["pathology"]
    random_frame, spatial_frame, selection = protocol.evaluate_pool(
        predictions,
        {name: True for name in predictions},
        args.metadata,
        settings,
        args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random_frame.to_csv(args.output_dir / "random_direct_metrics.csv", index=False)
    spatial_frame.to_csv(args.output_dir / "spatial_trackb_metrics.csv", index=False)
    selection.to_csv(args.output_dir / "inner_selection_audit.csv", index=False)
    summary = {
        "candidate_count": len(predictions),
        "random_direct": protocol.summary(random_frame),
        "spatial_trackb": protocol.summary(spatial_frame),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
