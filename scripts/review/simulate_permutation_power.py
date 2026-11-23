"""Simulate Aim 1 permutation rejection rates across larger cohorts.

No raw or participant-level mcPHASES data are required or imported.
Results are scenario-conditional, and therefore exploratory rather than
retrospective estimates of observed-study power.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from statistical_diagnostics import simulate_power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participants", type=int, nargs="+", default=[8, 25, 50]
    )
    parser.add_argument("--cycles", type=int, nargs="+", default=[2, 4])
    parser.add_argument(
        "--effects", type=float, nargs="+", default=[0.0, 0.5, 1.5]
    )
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--drift-sd", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/aim1/audit/synthetic_power_grid.csv"),
    )
    args = parser.parse_args()
    rows = []
    for n in args.participants:
        for cycles in args.cycles:
            for effect in args.effects:
                result = simulate_power(
                    n, cycles, effect, simulations=args.simulations,
                    permutations=args.permutations, seed=args.seed,
                    stratified=args.stratified, drift_sd=args.drift_sd,
                )
                rows.append(result)
                print(
                    f"N={n}, cycles={cycles}, profile_sd={effect}: "
                    f"rejection={result['rejection_fraction']:.3f} "
                    f"(95% MC CI {result['wilson_ci_95_low']:.3f}–"
                    f"{result['wilson_ci_95_high']:.3f})"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved synthetic power grid: {args.output}")


if __name__ == "__main__":
    main()
