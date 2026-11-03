"""Test whether personal phase profiles transfer to held-out cycles."""

from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_ROOT = Path(
    "results/aim2/centered_primary"
)

LONGITUDINAL_ROOT = Path(
    "results/aim2/longitudinal_folds"
)

OUTPUT_ROOT = Path(
    "results/aim2/profile_transfer"
)

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

SIGNALS = {
    "rhr": "resting_heart_rate",
    "hrv": "hrv_rmssd",
    "temp": "nightly_temperature",
    "resp": "respiratory_rate",
}


def safe_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Return Pearson correlation when defined."""

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0:
        return np.nan

    if np.std(y) == 0:
        return np.nan

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def cycle_profiles(
    data: pd.DataFrame,
    signal: str,
) -> pd.DataFrame:
    """Calculate phase-centered profile for each observed cycle."""

    grouped = (
        data.groupby(
            [
                "study_interval",
                "cycle_number",
                "phase",
            ],
            dropna=True,
        )[signal]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot_table(
        index=[
            "study_interval",
            "cycle_number",
        ],
        columns="phase",
        values=signal,
    )

    pivot = pivot.reindex(
        columns=PHASES
    )

    # Remove each cycle's overall physiological level.
    pivot = pivot.sub(
        pivot.mean(
            axis=1,
            skipna=True,
        ),
        axis=0,
    )

    return pivot


def calibration_signature(
    calibration: pd.DataFrame,
    signal: str,
):
    """Calculate two-cycle calibration signature and agreement."""

    profiles = cycle_profiles(
        calibration,
        signal,
    )

    complete = profiles.dropna(
        subset=PHASES
    )

    if len(complete) < 2:
        return (
            None,
            np.nan,
        )

    complete = complete.iloc[:2]

    first = complete.iloc[0].to_numpy(
        dtype=float
    )

    second = complete.iloc[1].to_numpy(
        dtype=float
    )

    agreement = safe_corr(
        first,
        second,
    )

    signature = complete.mean(
        axis=0
    )

    signature = (
        signature
        - signature.mean()
    )

    return (
        signature,
        agreement,
    )


def test_signature(
    test: pd.DataFrame,
    signal: str,
) -> pd.Series:
    """Calculate average held-out phase profile."""

    profiles = cycle_profiles(
        test,
        signal,
    )

    if profiles.empty:
        return pd.Series(
            index=PHASES,
            dtype=float,
        )

    # Equal-cycle weighting within each phase.
    signature = profiles.mean(
        axis=0,
        skipna=True,
    )

    signature = signature.reindex(
        PHASES
    )

    observed = signature.notna()

    if observed.sum() >= 1:
        signature.loc[observed] = (
            signature.loc[observed]
            - signature.loc[observed].mean()
        )

    return signature


def compare_signatures(
    calibration: pd.Series,
    test: pd.Series,
) -> dict:
    """Compare calibration and held-out phase signatures."""

    common = (
        calibration.notna()
        & test.notna()
    )

    phases = [
        phase
        for phase in PHASES
        if common.loc[phase]
    ]

    n_common = len(
        phases
    )

    if n_common < 3:
        return {
            "n_common_phases": n_common,
            "cal_to_test_corr": np.nan,
            "cal_to_test_rmse": np.nan,
            "sign_agreement": np.nan,
        }

    x = calibration.loc[
        phases
    ].to_numpy(
        dtype=float
    )

    y = test.loc[
        phases
    ].to_numpy(
        dtype=float
    )

    # Recenter on exactly the phases shared by both.
    x = x - x.mean()
    y = y - y.mean()

    correlation = safe_corr(
        x,
        y,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (x - y) ** 2
            )
        )
    )

    sign_agreement = float(
        np.mean(
            np.sign(x)
            == np.sign(y)
        )
    )

    return {
        "n_common_phases": n_common,
        "cal_to_test_corr": correlation,
        "cal_to_test_rmse": rmse,
        "sign_agreement": sign_agreement,
    }


def analyze_fold(
    fold_dir: Path,
    analysis: str,
) -> list[dict]:
    """Analyze one empirical Aim 2 fold."""

    calibration = pd.read_csv(
        fold_dir
        / "personal_calibration.csv"
    )

    test = pd.read_csv(
        fold_dir
        / "future_test.csv"
    )

    target_id = int(
        calibration[
            "target_id"
        ].iloc[0]
    )

    target_interval = int(
        test[
            "study_interval"
        ].iloc[0]
    )

    rows = []

    for short_name, column in SIGNALS.items():
        (
            calibration_profile,
            calibration_cycle_corr,
        ) = calibration_signature(
            calibration,
            column,
        )

        if calibration_profile is None:
            continue

        heldout_profile = test_signature(
            test,
            column,
        )

        comparison = compare_signatures(
            calibration_profile,
            heldout_profile,
        )

        row = {
            "analysis": analysis,
            "fold": fold_dir.name,
            "target_id": target_id,
            "target_interval": (
                target_interval
            ),
            "signal": short_name,
            "calibration_cycle_corr": (
                calibration_cycle_corr
            ),
        }

        row.update(
            comparison
        )

        rows.append(
            row
        )

    return rows


def summarize(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize profile transfer across folds."""

    rows = []

    for (
        analysis,
        signal,
    ), group in results.groupby(
        [
            "analysis",
            "signal",
        ]
    ):
        valid_transfer = group[
            "cal_to_test_corr"
        ].dropna()

        valid_calibration = group[
            "calibration_cycle_corr"
        ].dropna()

        valid_rmse = group[
            "cal_to_test_rmse"
        ].dropna()

        valid_sign = group[
            "sign_agreement"
        ].dropna()

        rows.append(
            {
                "analysis": analysis,
                "signal": signal,
                "n_folds": len(group),
                "n_transfer_evaluable": (
                    len(valid_transfer)
                ),
                "median_calibration_cycle_corr": (
                    valid_calibration.median()
                    if len(valid_calibration)
                    else np.nan
                ),
                "median_cal_to_test_corr": (
                    valid_transfer.median()
                    if len(valid_transfer)
                    else np.nan
                ),
                "fraction_transfer_corr_positive": (
                    float(
                        np.mean(
                            valid_transfer > 0
                        )
                    )
                    if len(valid_transfer)
                    else np.nan
                ),
                "median_cal_to_test_rmse": (
                    valid_rmse.median()
                    if len(valid_rmse)
                    else np.nan
                ),
                "median_sign_agreement": (
                    valid_sign.median()
                    if len(valid_sign)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    """Run empirical phase-profile transfer diagnostics."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    primary_folds = sorted(
        PRIMARY_ROOT.glob(
            "id*_*"
        )
    )

    longitudinal_folds = sorted(
        LONGITUDINAL_ROOT.glob(
            "id*"
        )
    )

    for fold_dir in primary_folds:
        if not (
            fold_dir
            / "personal_calibration.csv"
        ).exists():
            continue

        rows.extend(
            analyze_fold(
                fold_dir,
                "within_interval",
            )
        )

    for fold_dir in longitudinal_folds:
        if not (
            fold_dir
            / "personal_calibration.csv"
        ).exists():
            continue

        rows.extend(
            analyze_fold(
                fold_dir,
                "longitudinal_2022_to_2024",
            )
        )

    results = pd.DataFrame(
        rows
    )

    summary = summarize(
        results
    )

    results.to_csv(
        OUTPUT_ROOT
        / "fold_profile_transfer.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_ROOT
        / "profile_transfer_summary.csv",
        index=False,
    )

    print("=" * 86)
    print("EMPIRICAL PERSONAL-PROFILE TRANSFER")
    print("=" * 86)

    print(
        "\nWithin-interval folds:",
        results.loc[
            results["analysis"]
            == "within_interval",
            "fold",
        ].nunique(),
    )

    print(
        "Longitudinal folds:",
        results.loc[
            results["analysis"]
            == "longitudinal_2022_to_2024",
            "fold",
        ].nunique(),
    )

    print("\nSUMMARY")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print(
        "\nLONGITUDINAL FOLD-LEVEL TRANSFER"
    )

    longitudinal = results[
        results["analysis"]
        == "longitudinal_2022_to_2024"
    ][
        [
            "target_id",
            "signal",
            "calibration_cycle_corr",
            "cal_to_test_corr",
            "cal_to_test_rmse",
            "sign_agreement",
        ]
    ]

    print(
        longitudinal.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )

    print(
        "\nOutputs:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
