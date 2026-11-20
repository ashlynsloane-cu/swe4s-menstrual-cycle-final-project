"""Audit the number of distinct participants in cluster-bootstrap draws.

Repeated participants are expected when sampling *with replacement*. This
script reproduces the ID sampling scheme only; it does not recalculate CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from statistical_diagnostics import bootstrap_id_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=6118)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/aim2/audit/bootstrap_diversity.json"),
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, usecols=["target_id"])
    summary = bootstrap_id_diagnostics(
        frame["target_id"].dropna().to_numpy(),
        draws=args.draws, seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved aggregate bootstrap audit to {args.output}")


if __name__ == "__main__":
    main()
