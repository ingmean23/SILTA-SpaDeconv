from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from silta_osm.checkpoint import inspect
from silta_osm.io import sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the released SILTA state dictionary"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoint" / "silta_osm_b_fold42_seed42.model",
    )
    args = parser.parse_args()
    result = dict(inspect(args.checkpoint))
    result.update({
        "file": args.checkpoint.name,
        "bytes": args.checkpoint.stat().st_size,
        "sha256": sha256(args.checkpoint),
    })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
