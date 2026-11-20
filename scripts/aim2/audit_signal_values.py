"""Summarize wearable variable ranges without exposing individual records.

This cannot confirm original Fitbit-export units: consult the mcPHASES
source dictionary and the exact computed_temperature field definition.
Do not use value ranges alone to assign degrees Celsius or Fahrenheit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SIGNALS = (
    "resting_heart_rate",
    "hrv_rmssd",
    "nightly_temperature",
    "respiratory_rate",
)


def summarize_signal_values(daily: pd.DataFrame) -> dict:
    """Provide only aggregate bounds and quantiles, without participant IDs."""
    result = {}
    for signal in SIGNALS:
        if signal not in daily:
            raise ValueError(
                f"Missing expected physiological signal: {signal}"
            )
        numeric = pd.to_numeric(daily[signal], errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        result[signal] = {
            "nonmissing_n": int(finite.size),
            "missing_or_nonfinite_n": int(len(numeric) - finite.size),
            "minimum": float(finite.min()) if len(finite) else None,
            "median": float(finite.median()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
            "p05": float(finite.quantile(0.05)) if len(finite) else None,
            "p95": float(finite.quantile(0.95)) if len(finite) else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily", type=Path,
        default=Path("results/preprocessed/mcphases_daily_master.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/aim2/audit/signal_value_summary.json"),
    )
    args = parser.parse_args()
    summary = summarize_signal_values(pd.read_csv(args.daily))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote aggregate data-quality summary: {args.output}")
    print("Units still require source-dictionary verification.")


if __name__ == "__main__":
    main()
