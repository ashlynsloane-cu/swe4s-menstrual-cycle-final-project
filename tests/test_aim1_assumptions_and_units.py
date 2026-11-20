"""Fast checks for descriptive audits, without accessing restricted data."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1] / "scripts/aim2"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aim1_assumption_report_uses_only_aggregated_pair_counts():
    audit = load_script("audit_aim1_assumptions")
    signatures = pd.DataFrame({
        "id": [1, 1, 2, 2, 3, 3, 4, 4],
        "study_interval": [2022, 2022, 2022, 2022, 2024, 2024, 2024, 2024],
        "cycle_number": [1, 2] * 4,
        "feature_a": [1.0, 1.3, 3.0, 3.3, 5.0, 5.3, 7.0, 7.3],
        "feature_b": [8.0, 7.7, 6.0, 5.7, 4.0, 3.7, 2.0, 1.7],
    })
    summary = audit.audit_signatures(signatures, permutations=29)
    assert summary["n_complete_signatures"] == 8
    assert summary["n_within_pairs"] == 4
    assert summary["n_between_pairs"] == 24
    assert summary["n_cross_interval_same_person_pairs"] == 0
    assert summary["global_null_finite"]
    assert summary["interval_null_finite"]
    assert 0 < summary["global_permutation_p"] <= 1
    assert 0 < summary["interval_stratified_permutation_p"] <= 1
    assert "participant_ids" not in summary


def test_aim1_assumptions_reject_nonpositive_permutations():
    audit = load_script("audit_aim1_assumptions")
    with pytest.raises(ValueError, match="positive integer"):
        audit.audit_signatures(
            pd.DataFrame(columns=["id", "study_interval", "cycle_number"]),
            permutations=0,
        )


def test_signal_unit_audit_never_returns_participant_identifiers():
    audit = load_script("audit_signal_values")
    data = pd.DataFrame({
        "id": [1, 1, 3],
        "resting_heart_rate": [60, 70, np.nan],
        "hrv_rmssd": [20, 25, 30],
        "nightly_temperature": [-0.2, 0.1, 0.2],
        "respiratory_rate": [12, 13, 14],
    })
    summary = audit.summarize_signal_values(data)
    assert set(summary) == set(audit.SIGNALS)
    assert summary["resting_heart_rate"]["nonmissing_n"] == 2
    assert summary["resting_heart_rate"]["missing_or_nonfinite_n"] == 1
    assert "id" not in str(summary)


def test_signal_unit_audit_fails_for_unknown_column():
    audit = load_script("audit_signal_values")
    with pytest.raises(ValueError, match="Missing expected"):
        audit.summarize_signal_values(
            pd.DataFrame({"resting_heart_rate": [60]})
        )
