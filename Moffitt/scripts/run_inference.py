from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silta.inference import predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SILTA inference on Moffitt-format inputs")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "silta_moffitt_B-M1_seed42.pt")
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(predict(
        args.checkpoint, args.expression, args.coordinates, args.output, args.device
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
