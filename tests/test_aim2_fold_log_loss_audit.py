"""Detect incorrect saved log loss without rerunning Bayesian fits."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


FILE = (
    Path(__file__).resolve().parents[1]
    / "scripts/aim2/audit_fold_log_loss.py"
)
SPEC = importlib.util.spec_from_file_location("fold_log_loss_audit", FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_data():
    records = []
    for model in ("population_only", "hybrid_personalized"):
        for index, phase in enumerate(("Menstrual", "Luteal")):
            probabilities = [0.9, 0.05, 0.025, 0.025]
            if phase == "Luteal":
                probabilities = [0.05, 0.025, 0.025, 0.9]
            records.append({
                "target_id": 4, "target_interval": 2024, "model": model,
                "phase": phase,
                "prob_menstrual": probabilities[0],
                "prob_follicular": probabilities[1],
                "prob_fertility": probabilities[2],
                "prob_luteal": probabilities[3],
            })
    predictions = pd.DataFrame(records)
    expected = -np.log(0.9)
    fold_results = pd.DataFrame({
        "target_id": [4, 4], "target_interval": [2024, 2024],
        "model": ["population_only", "hybrid_personalized"],
        "log_loss": [expected, expected],
    })
    return predictions, fold_results


def test_correct_fold_log_loss_matches_reported():
    predictions, folds = make_data()
    result = MODULE.compare_saved_log_loss(predictions, folds)
    assert result["n_folds_scored"] == 2
    assert result["all_fold_log_losses_agree"]
    assert result["n_mismatched_fold_log_losses"] == 0


def test_wrong_fold_log_loss_is_flagged():
    predictions, folds = make_data()
    folds.loc[0, "log_loss"] = 1.95
    result = MODULE.compare_saved_log_loss(predictions, folds)
    assert result["n_mismatched_fold_log_losses"] == 1
    assert not result["all_fold_log_losses_agree"]
    assert result["maximum_absolute_log_loss_difference"] > 1


def test_unmatched_model_fold_is_rejected():
    predictions, folds = make_data()
    with pytest.raises(ValueError, match="do not match"):
        MODULE.compare_saved_log_loss(predictions, folds.iloc[:1])
