"""Compare saved fold log loss against aligned prediction probabilities.

Reads local ignored output files; writes only aggregate mismatch statistics.
Does not fit models, modify scientific outputs, or expose participant rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PHASES = ("Menstrual", "Follicular", "Fertility", "Luteal")
KEY = ["target_id", "target_interval", "model"]


def aligned_group_log_loss(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recompute fold NLL from explicitly named class columns."""
    needed = set(KEY + ["phase"])
    needed.update(f"prob_{phase.lower()}" for phase in PHASES)
    missing = sorted(needed.difference(predictions.columns))
    if missing:
        raise ValueError("Missing prediction columns: " + ", ".join(missing))

    values = predictions[[f"prob_{p.lower()}" for p in PHASES]].to_numpy()
    true_phase = predictions["phase"].astype(str).to_numpy()
    lookup = {phase: index for index, phase in enumerate(PHASES)}
    unknown = sorted(set(true_phase).difference(lookup))
    if unknown:
        raise ValueError(f"Unrecognized phase labels: {unknown}")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities contain non-finite entries")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Probability rows must sum to one")

    indices = np.fromiter(
        (lookup[phase] for phase in true_phase), dtype=int,
        count=len(true_phase),
    )
    log_p = -np.log(np.clip(values[np.arange(len(indices)), indices],
                            1e-15, 1.0))
    audit = predictions[KEY].copy()
    audit["aligned_nll"] = log_p
    return (
        audit.groupby(KEY, as_index=False, dropna=False)
        .agg(recomputed_log_loss=("aligned_nll", "mean"),
             n_prediction_rows=("aligned_nll", "size"))
    )


def compare_saved_log_loss(
    predictions: pd.DataFrame,
    fold_results: pd.DataFrame,
    tolerance: float = 1e-7,
) -> dict:
    """Check bidirectional fold coverage and saved-versus-aligned NLL."""
    expected = aligned_group_log_loss(predictions)
    if not set(KEY + ["log_loss"]).issubset(fold_results.columns):
        raise ValueError("Fold result missing identifying columns or log_loss")
    reported = fold_results[KEY + ["log_loss"]].copy()
    if reported.duplicated(KEY).any() or expected.duplicated(KEY).any():
        raise ValueError("Duplicate fold-model keys in log-loss audit")
    checked = expected.merge(
        reported, how="outer", on=KEY, indicator=True,
        validate="one_to_one",
    )
    if not (checked["_merge"] == "both").all():
        raise ValueError("Saved fold models do not match prediction groups")
    delta = (checked["recomputed_log_loss"] - checked["log_loss"]).abs()
    if not np.isfinite(delta).all():
        raise ValueError("Reported fold log-loss contains invalid values")
    return {
        "n_folds_scored": int(len(delta)),
        "tolerance": float(tolerance),
        "n_mismatched_fold_log_losses": int((delta > tolerance).sum()),
        "maximum_absolute_log_loss_difference": float(delta.max()),
        "all_fold_log_losses_agree": bool((delta <= tolerance).all()),
        "caution": (
            "This checks saved scores against named columns, but does not "
            "verify posterior fitting, calibration or data provenance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", choices=("primary", "longitudinal"), required=True
    )
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("--tolerance must be nonnegative")
    root = Path(
        "results/aim2/primary_models" if args.analysis == "primary"
        else "results/aim2/longitudinal_models"
    )
    predictions_path = root / "all_predictions.csv"
    metrics_path = root / "all_fold_results.csv"
    summary = compare_saved_log_loss(
        pd.read_csv(predictions_path), pd.read_csv(metrics_path),
        tolerance=args.tolerance,
    )
    output = Path("results/aim2/audit") / (
        f"{args.analysis}_fold_log_loss_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote aggregate fold-log-loss audit: {output}")
    print(f"Mismatches: {summary['n_mismatched_fold_log_losses']}")


if __name__ == "__main__":
    main()
