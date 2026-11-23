"""Fail closed on actual leakage scenarios; honor personal calibration."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "scripts" / "review"))
from audit_split_leakage import inspect_fold  # noqa: E402
from check_mcphases_inputs import (  # noqa: E402
    RAW_FILES,
    check_source_directory,
)


def make_fold(year=2022, test_year=2022):
    pop = pd.DataFrame({
        "source_id": [2, 3], "study_interval": [year, year],
        "day_in_study": [1, 2],
    })
    calibration = pd.DataFrame({
        "source_id": [1, 1], "study_interval": [year, year],
        "day_in_study": [10, 11],
    })
    test = pd.DataFrame({
        "source_id": [1, 1], "study_interval": [test_year, test_year],
        "day_in_study": [20, 21] if year == test_year else [900, 901],
    })
    return pop, calibration, test


def test_primary_fold_excludes_target_from_population():
    result = inspect_fold(*make_fold(), analysis="primary")
    assert result["population_excludes_target"]
    assert result["target_in_both_calibration_and_test_by_design"]
    assert result["calibration_test_day_overlap"] == 0


def test_population_target_overlap_is_rejected():
    pop, cal, test = make_fold()
    pop.loc[0, "source_id"] = 1
    with pytest.raises(ValueError, match="population training"):
        inspect_fold(pop, cal, test)


def test_calibration_test_day_overlap_is_rejected():
    pop, cal, test = make_fold()
    test.loc[0, "day_in_study"] = 10
    with pytest.raises(ValueError, match="participant-days"):
        inspect_fold(pop, cal, test)


def test_primary_temporal_leakage_is_rejected():
    pop, cal, test = make_fold()
    test["day_in_study"] = [9, 20]
    with pytest.raises(ValueError, match="precedes"):
        inspect_fold(pop, cal, test)


def test_longitudinal_temporal_holdout():
    frames = make_fold(year=2022, test_year=2024)
    result = inspect_fold(*frames, analysis="longitudinal")
    assert result["chronology_verified"]
    with pytest.raises(ValueError, match="follow calibration"):
        inspect_fold(*make_fold(year=2024, test_year=2022),
                     analysis="longitudinal")


def test_preflight_fails_when_raw_data_missing(tmp_path):
    result = check_source_directory(tmp_path)
    assert not result["ok"]
    assert len(result["problems"]) == len(RAW_FILES)
    assert "credential" not in " ".join(result["problems"]).lower()


def test_preflight_accepts_local_csv_headers_only(tmp_path):
    for name in RAW_FILES:
        (tmp_path / name).write_text("id,day_in_study\n1,1\n")
    assert check_source_directory(tmp_path)["ok"]
    (tmp_path / RAW_FILES[0]).write_text("")
    assert not check_source_directory(tmp_path)["ok"]
