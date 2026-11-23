"""Shared, model-independent output helpers for both final Aim 2 fits.

The primary and longitudinal models intentionally have different fold
layouts and different analysis populations. Only the identical probability
extraction and CSV aggregation logic lives here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def extract_probabilities(
    model,
    idata,
    test: pd.DataFrame,
    include_group_specific: bool,
) -> tuple[np.ndarray, list[str]]:
    """Return posterior-mean class probabilities in model category order.

    Parameters
    ----------
    model : object
        Fitted Bambi model exposing ``predict``.
    idata : object
        Inference data passed to ``model.predict``.
    test : pandas.DataFrame
        Held-out rows in prediction order.
    include_group_specific : bool
        Whether to include participant-specific model effects.

    Returns
    -------
    tuple[numpy.ndarray, list[str]]
        Matrix of shape (test rows, classes) and ordered phase labels.
    """
    predicted = model.predict(
        idata,
        data=test,
        kind="response_params",
        inplace=False,
        include_group_specific=include_group_specific,
    )

    probability_data = predicted.posterior["p"].mean(
        dim=("chain", "draw")
    )

    observation_dims = [
        dim for dim in probability_data.dims if "obs" in dim
    ]

    if len(observation_dims) != 1:
        raise RuntimeError(
            "Could not identify prediction observation dimension."
        )

    observation_dim = observation_dims[0]
    category_dims = [
        dim for dim in probability_data.dims if dim != observation_dim
    ]

    if len(category_dims) != 1:
        raise RuntimeError(
            "Could not identify prediction category dimension."
        )

    category_dim = category_dims[0]
    probability_data = probability_data.transpose(
        observation_dim, category_dim
    )
    labels = [
        str(value)
        for value in probability_data.coords[category_dim].values
    ]

    return probability_data.to_numpy(), labels


def combine_results(output_dir: Path, fold_dir_glob: str) -> None:
    """Aggregate completed Aim 2 folds using the caller's folder layout.

    ``fold_dir_glob='id*_*'`` matches within-interval primary folders;
    ``fold_dir_glob='id*'`` matches longitudinal folders. Do not merge the
    two patterns: they describe different scientific evaluation designs.
    """
    output_dir = Path(output_dir)
    result_files = sorted(
        output_dir.glob(f"{fold_dir_glob}/fold_result.csv")
    )
    prediction_files = sorted(
        output_dir.glob(f"{fold_dir_glob}/predictions.csv")
    )

    if result_files:
        results = pd.concat(
            [pd.read_csv(path) for path in result_files],
            ignore_index=True,
        )
        results.to_csv(output_dir / "all_fold_results.csv", index=False)

    if prediction_files:
        predictions = pd.concat(
            [pd.read_csv(path) for path in prediction_files],
            ignore_index=True,
        )
        predictions.to_csv(output_dir / "all_predictions.csv", index=False)


def aligned_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> float:
    """Return categorical negative log likelihood in actual column order.

    Unlike sklearn's ``log_loss(labels=...)``, this function does not assume
    its probability columns are lexicographically sorted by class label.
    The index map is constructed directly from Bambi's posterior category
    coordinate, which must match the probability matrix's column order.
    """
    labels = [str(label) for label in labels]
    y_true = np.asarray(y_true).astype(str)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(labels) != len(set(labels)):
        raise ValueError("Class labels must be unique")
    if (
        probabilities.ndim != 2
        or probabilities.shape != (len(y_true), len(labels))
        or len(y_true) == 0
    ):
        raise ValueError("Probability matrix and labels must match rows")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("Probabilities must be valid categorical vectors")

    lookup = {label: index for index, label in enumerate(labels)}
    try:
        indices = np.array([lookup[value] for value in y_true])
    except KeyError as error:
        raise ValueError(
            "Observed class absent from probability labels"
        ) from error

    true_probabilities = probabilities[
        np.arange(len(y_true)), indices
    ]
    return float(-np.log(np.clip(true_probabilities, 1e-15, 1.0)).mean())
