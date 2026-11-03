"""Simulate how calibration-cycle depth affects personalization."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPONENT_ROOT = Path(
    "results/aim2/simulation_components"
)

DEFAULT_OUTPUT = Path(
    "results/aim2/cycle_depth_simulation"
)

DEPTHS = [1, 2, 3, 4, 6, 8]

PHASES = [
    "Menstrual",
    "Follicular",
    "Fertility",
    "Luteal",
]

SIGNALS = [
    "rhr",
    "hrv",
    "temp",
    "resp",
]

HIGHER_IS_BETTER = {
    "accuracy": True,
    "balanced_accuracy": True,
    "macro_f1": True,
    "log_loss": False,
    "brier_score": False,
    "mean_true_probability": True,
}


def parse_args():
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Data-calibrated simulation of Aim 2 "
            "personalization depth."
        )
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--participants",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=6118,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def load_parameters():
    """Load parameters estimated previously from mcPHASES."""

    repeatability = pd.read_csv(
        COMPONENT_ROOT
        / "phase_profile_repeatability.csv"
    )

    population = pd.read_csv(
        COMPONENT_ROOT
        / "population_phase_patterns.csv"
    )

    population_means = (
        population.pivot(
            index="phase",
            columns="signal",
            values="population_phase_mean_z",
        )
        .reindex(
            index=PHASES,
            columns=SIGNALS,
        )
    )

    if population_means.isna().any().any():
        raise RuntimeError(
            "Population phase-pattern table is incomplete."
        )

    repeatability = (
        repeatability
        .set_index("signal")
        .reindex(SIGNALS)
    )

    stable_var = repeatability[
        "stable_person_phase_variance"
    ].to_numpy(
        dtype=float
    )

    transient_var = repeatability[
        "transient_cycle_phase_variance"
    ].to_numpy(
        dtype=float
    )

    day_var = repeatability[
        "day_variance"
    ].to_numpy(
        dtype=float
    )

    return (
        population_means.to_numpy(
            dtype=float
        ),
        stable_var,
        transient_var,
        day_var,
        repeatability,
    )


def load_cycle_templates():
    """Load empirical complete-cycle phase durations."""

    complete_sets = []
    decomposition_tables = {}

    for signal in SIGNALS:
        path = (
            COMPONENT_ROOT
            / f"{signal}_decomposition.csv"
        )

        data = pd.read_csv(path)

        decomposition_tables[signal] = data

        n_phases = (
            data.groupby("cycle_id")["phase"]
            .nunique()
        )

        complete_sets.append(
            set(
                n_phases[
                    n_phases == 4
                ].index
            )
        )

    common_cycles = set.intersection(
        *complete_sets
    )

    if not common_cycles:
        raise RuntimeError(
            "No complete four-phase cycles were found "
            "across all four signals."
        )

    base = decomposition_tables["rhr"]

    base = base[
        base["cycle_id"].isin(
            common_cycles
        )
    ].copy()

    templates = (
        base.groupby(
            [
                "cycle_id",
                "phase",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            columns=PHASES
        )
    )

    templates = templates[
        (templates > 0).all(axis=1)
    ]

    if templates.empty:
        raise RuntimeError(
            "No usable empirical cycle templates."
        )

    phase_counts = (
        templates.sum(
            axis=0
        )
        .reindex(PHASES)
        .to_numpy(
            dtype=float
        )
    )

    priors = (
        phase_counts
        / phase_counts.sum()
    )

    return (
        templates.to_numpy(
            dtype=int
        ),
        templates.index.tolist(),
        priors,
    )


def centered_effect(
    rng,
    variances,
):
    """Draw a four-phase effect with zero phase mean."""

    n_phases = len(PHASES)

    output = np.zeros(
        (
            n_phases,
            len(SIGNALS),
        ),
        dtype=float,
    )

    correction = (
        1.0
        - 1.0 / n_phases
    )

    for signal_index, variance in enumerate(
        variances
    ):
        if variance <= 0:
            continue

        raw_sd = np.sqrt(
            variance
            / correction
        )

        values = rng.normal(
            loc=0.0,
            scale=raw_sd,
            size=n_phases,
        )

        values = (
            values
            - values.mean()
        )

        output[
            :,
            signal_index,
        ] = values

    return output


def simulate_calibration(
    rng,
    depth,
    templates,
    population_means,
    stable_effect,
    stable_var,
    transient_var,
    day_var,
):
    """Estimate a stable personal phase profile from prior cycles."""

    observed_offsets = []

    noise_variances = []

    for _ in range(depth):
        template_index = rng.integers(
            0,
            len(templates),
        )

        durations = templates[
            template_index
        ].astype(float)

        transient = centered_effect(
            rng,
            transient_var,
        )

        day_mean_sd = np.sqrt(
            day_var[np.newaxis, :]
            / durations[:, np.newaxis]
        )

        day_mean_noise = rng.normal(
            loc=0.0,
            scale=day_mean_sd,
        )

        observed_phase_mean = (
            population_means
            + stable_effect
            + transient
            + day_mean_noise
        )

        observed_offsets.append(
            observed_phase_mean
            - population_means
        )

        noise_variances.append(
            transient_var[
                np.newaxis,
                :
            ]
            + (
                day_var[
                    np.newaxis,
                    :
                ]
                / durations[
                    :,
                    np.newaxis,
                ]
            )
        )

    observed_offsets = np.stack(
        observed_offsets,
        axis=0,
    )

    noise_variances = np.stack(
        noise_variances,
        axis=0,
    )

    raw_estimate = observed_offsets.mean(
        axis=0
    )

    # Variance of the equal-cycle average.
    calibration_noise_var = (
        noise_variances.sum(
            axis=0
        )
        / (depth ** 2)
    )

    stable_matrix = np.broadcast_to(
        stable_var[
            np.newaxis,
            :
        ],
        raw_estimate.shape,
    )

    denominator = (
        stable_matrix
        + calibration_noise_var
    )

    shrinkage = np.divide(
        stable_matrix,
        denominator,
        out=np.zeros_like(
            stable_matrix
        ),
        where=denominator > 0,
    )

    estimated_effect = (
        shrinkage
        * raw_estimate
    )

    # Personal phase profiles were estimated after
    # removing overall cycle level, so retain that
    # zero-across-phase constraint here.
    estimated_effect = (
        estimated_effect
        - estimated_effect.mean(
            axis=0,
            keepdims=True,
        )
    )

    posterior_var = np.divide(
        stable_matrix
        * calibration_noise_var,
        denominator,
        out=np.zeros_like(
            stable_matrix
        ),
        where=denominator > 0,
    )

    return (
        estimated_effect,
        posterior_var,
    )


def simulate_test_cycle(
    rng,
    templates,
    population_means,
    stable_effect,
    transient_var,
    day_var,
):
    """Generate the next hormone-defined test cycle."""

    template_index = rng.integers(
        0,
        len(templates),
    )

    durations = templates[
        template_index
    ]

    transient = centered_effect(
        rng,
        transient_var,
    )

    observations = []
    labels = []

    for phase_index, n_days in enumerate(
        durations
    ):
        if n_days <= 0:
            continue

        mean = (
            population_means[
                phase_index
            ]
            + stable_effect[
                phase_index
            ]
            + transient[
                phase_index
            ]
        )

        noise = rng.normal(
            loc=0.0,
            scale=np.sqrt(
                day_var
            ),
            size=(
                int(n_days),
                len(SIGNALS),
            ),
        )

        values = (
            mean[
                np.newaxis,
                :
            ]
            + noise
        )

        observations.append(
            values
        )

        labels.extend(
            [phase_index]
            * int(n_days)
        )

    return (
        np.vstack(
            observations
        ),
        np.asarray(
            labels,
            dtype=int,
        ),
    )


def predict_probabilities(
    observations,
    means,
    predictive_variance,
    priors,
):
    """Calculate Gaussian phase-class probabilities."""

    n_obs = observations.shape[0]

    if predictive_variance.ndim == 1:
        variance = np.broadcast_to(
            predictive_variance[
                np.newaxis,
                :
            ],
            means.shape,
        )
    else:
        variance = predictive_variance

    variance = np.maximum(
        variance,
        1e-12,
    )

    log_scores = np.empty(
        (
            n_obs,
            len(PHASES),
        ),
        dtype=float,
    )

    for phase_index in range(
        len(PHASES)
    ):
        difference = (
            observations
            - means[
                phase_index
            ]
        )

        gaussian = -0.5 * np.sum(
            np.log(
                2.0
                * np.pi
                * variance[
                    phase_index
                ]
            )
            + (
                difference ** 2
                / variance[
                    phase_index
                ]
            ),
            axis=1,
        )

        log_scores[
            :,
            phase_index,
        ] = (
            np.log(
                priors[
                    phase_index
                ]
            )
            + gaussian
        )

    maximum = log_scores.max(
        axis=1,
        keepdims=True,
    )

    exp_scores = np.exp(
        log_scores
        - maximum
    )

    probabilities = (
        exp_scores
        / exp_scores.sum(
            axis=1,
            keepdims=True,
        )
    )

    return probabilities


def evaluate(
    truth,
    probabilities,
):
    """Calculate multiclass metrics."""

    predicted = probabilities.argmax(
        axis=1
    )

    accuracy = np.mean(
        predicted == truth
    )

    recalls = []
    f1_values = []

    for phase_index in range(
        len(PHASES)
    ):
        true_mask = (
            truth == phase_index
        )

        predicted_mask = (
            predicted == phase_index
        )

        true_positive = np.sum(
            true_mask
            & predicted_mask
        )

        false_positive = np.sum(
            (~true_mask)
            & predicted_mask
        )

        false_negative = np.sum(
            true_mask
            & (~predicted_mask)
        )

        recall_denominator = (
            true_positive
            + false_negative
        )

        precision_denominator = (
            true_positive
            + false_positive
        )

        recall = (
            true_positive
            / recall_denominator
            if recall_denominator
            else 0.0
        )

        precision = (
            true_positive
            / precision_denominator
            if precision_denominator
            else 0.0
        )

        recalls.append(
            recall
        )

        if precision + recall > 0:
            f1_value = (
                2.0
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )
        else:
            f1_value = 0.0

        f1_values.append(
            f1_value
        )

    row_index = np.arange(
        len(truth)
    )

    true_probability = probabilities[
        row_index,
        truth,
    ]

    clipped = np.clip(
        true_probability,
        1e-15,
        1.0,
    )

    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        row_index,
        truth,
    ] = 1.0

    return {
        "accuracy": float(
            accuracy
        ),
        "balanced_accuracy": float(
            np.mean(
                recalls
            )
        ),
        "macro_f1": float(
            np.mean(
                f1_values
            )
        ),
        "log_loss": float(
            -np.mean(
                np.log(
                    clipped
                )
            )
        ),
        "brier_score": float(
            np.mean(
                np.sum(
                    (
                        probabilities
                        - one_hot
                    )
                    ** 2,
                    axis=1,
                )
            )
        ),
        "mean_true_probability": float(
            np.mean(
                true_probability
            )
        ),
    }


def simulate_one_depth(
    rng,
    depth,
    n_participants,
    templates,
    population_means,
    stable_var,
    transient_var,
    day_var,
    priors,
):
    """Simulate one replicate at one calibration depth."""

    truths = []
    population_probabilities = []
    personalized_probabilities = []

    population_predictive_var = (
        stable_var
        + transient_var
        + day_var
    )

    for _ in range(
        n_participants
    ):
        stable_effect = centered_effect(
            rng,
            stable_var,
        )

        (
            estimated_effect,
            posterior_var,
        ) = simulate_calibration(
            rng=rng,
            depth=depth,
            templates=templates,
            population_means=population_means,
            stable_effect=stable_effect,
            stable_var=stable_var,
            transient_var=transient_var,
            day_var=day_var,
        )

        (
            test_values,
            test_truth,
        ) = simulate_test_cycle(
            rng=rng,
            templates=templates,
            population_means=population_means,
            stable_effect=stable_effect,
            transient_var=transient_var,
            day_var=day_var,
        )

        population_prob = (
            predict_probabilities(
                observations=test_values,
                means=population_means,
                predictive_variance=(
                    population_predictive_var
                ),
                priors=priors,
            )
        )

        personalized_means = (
            population_means
            + estimated_effect
        )

        personalized_predictive_var = (
            posterior_var
            + transient_var[
                np.newaxis,
                :
            ]
            + day_var[
                np.newaxis,
                :
            ]
        )

        personalized_prob = (
            predict_probabilities(
                observations=test_values,
                means=personalized_means,
                predictive_variance=(
                    personalized_predictive_var
                ),
                priors=priors,
            )
        )

        truths.append(
            test_truth
        )

        population_probabilities.append(
            population_prob
        )

        personalized_probabilities.append(
            personalized_prob
        )

    truth = np.concatenate(
        truths
    )

    population_probabilities = np.vstack(
        population_probabilities
    )

    personalized_probabilities = np.vstack(
        personalized_probabilities
    )

    population_metrics = evaluate(
        truth,
        population_probabilities,
    )

    personalized_metrics = evaluate(
        truth,
        personalized_probabilities,
    )

    output = {
        "depth": depth,
        "n_test_days": len(
            truth
        ),
    }

    for metric in HIGHER_IS_BETTER:
        output[
            f"population_{metric}"
        ] = population_metrics[
            metric
        ]

        output[
            f"personalized_{metric}"
        ] = personalized_metrics[
            metric
        ]

        output[
            f"delta_{metric}"
        ] = (
            personalized_metrics[
                metric
            ]
            - population_metrics[
                metric
            ]
        )

    return output


def summarize(
    replicate_results,
):
    """Summarize Monte Carlo differences."""

    rows = []

    for depth in DEPTHS:
        subset = replicate_results[
            replicate_results[
                "depth"
            ]
            == depth
        ]

        for metric, higher_is_better in (
            HIGHER_IS_BETTER.items()
        ):
            column = (
                f"delta_{metric}"
            )

            values = subset[
                column
            ].to_numpy()

            if higher_is_better:
                improvement_probability = np.mean(
                    values > 0
                )
            else:
                improvement_probability = np.mean(
                    values < 0
                )

            rows.append(
                {
                    "depth": depth,
                    "metric": metric,
                    "mean_difference": (
                        values.mean()
                    ),
                    "median_difference": (
                        np.median(
                            values
                        )
                    ),
                    "ci_2.5": np.quantile(
                        values,
                        0.025,
                    ),
                    "ci_97.5": np.quantile(
                        values,
                        0.975,
                    ),
                    "probability_personalized_better": (
                        improvement_probability
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def find_log_loss_thresholds(
    summary,
):
    """Identify prespecified primary crossover depths."""

    log_loss = summary[
        summary["metric"]
        == "log_loss"
    ].sort_values(
        "depth"
    )

    expected = log_loss[
        log_loss[
            "mean_difference"
        ] < 0
    ]

    reliable = log_loss[
        log_loss[
            "ci_97.5"
        ] < 0
    ]

    expected_depth = (
        int(
            expected.iloc[0][
                "depth"
            ]
        )
        if not expected.empty
        else None
    )

    reliable_depth = (
        int(
            reliable.iloc[0][
                "depth"
            ]
        )
        if not reliable.empty
        else None
    )

    return (
        expected_depth,
        reliable_depth,
    )


def make_primary_figure(
    summary,
    output_dir,
):
    """Plot personalized-minus-population log loss."""

    data = summary[
        summary["metric"]
        == "log_loss"
    ].sort_values(
        "depth"
    )

    x = data[
        "depth"
    ].to_numpy()

    y = data[
        "mean_difference"
    ].to_numpy()

    low = data[
        "ci_2.5"
    ].to_numpy()

    high = data[
        "ci_97.5"
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(7.5, 4.8)
    )

    line = ax.plot(
        x,
        y,
        marker="o",
    )[0]

    ax.fill_between(
        x,
        low,
        high,
        alpha=0.2,
        color=line.get_color(),
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        DEPTHS
    )

    ax.set_xlabel(
        "Prior calibration cycles"
    )

    ax.set_ylabel(
        "Personalized − population log loss"
    )

    ax.set_title(
        "When does personalization improve "
        "next-cycle prediction?"
    )

    ax.text(
        0.01,
        0.02,
        "Below zero favors personalization",
        transform=ax.transAxes,
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "log_loss_crossover.png",
        dpi=300,
    )

    plt.close(
        fig
    )


def main():
    """Run the cycle-depth simulation."""

    args = parse_args()

    if args.replicates < 1:
        raise ValueError(
            "--replicates must be >= 1."
        )

    if args.participants < 1:
        raise ValueError(
            "--participants must be >= 1."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        population_means,
        stable_var,
        transient_var,
        day_var,
        repeatability,
    ) = load_parameters()

    (
        templates,
        template_ids,
        priors,
    ) = load_cycle_templates()

    rng = np.random.default_rng(
        args.seed
    )

    rows = []

    total_jobs = (
        args.replicates
        * len(DEPTHS)
    )

    job = 0

    for replicate in range(
        args.replicates
    ):
        for depth in DEPTHS:
            result = simulate_one_depth(
                rng=rng,
                depth=depth,
                n_participants=(
                    args.participants
                ),
                templates=templates,
                population_means=(
                    population_means
                ),
                stable_var=stable_var,
                transient_var=(
                    transient_var
                ),
                day_var=day_var,
                priors=priors,
            )

            result[
                "replicate"
            ] = replicate

            rows.append(
                result
            )

            job += 1

        if (
            (replicate + 1) % 50 == 0
            or replicate + 1
            == args.replicates
        ):
            print(
                "Completed",
                job,
                "/",
                total_jobs,
                "simulation cells",
            )

    replicate_results = pd.DataFrame(
        rows
    )

    summary = summarize(
        replicate_results
    )

    (
        expected_depth,
        reliable_depth,
    ) = find_log_loss_thresholds(
        summary
    )

    replicate_results.to_csv(
        args.output_dir
        / "replicate_metrics.csv",
        index=False,
    )

    summary.to_csv(
        args.output_dir
        / "simulation_summary.csv",
        index=False,
    )

    parameter_snapshot = (
        repeatability.reset_index()[
            [
                "signal",
                "profile_repeatability",
                "stable_person_phase_variance",
                "transient_cycle_phase_variance",
                "day_variance",
            ]
        ]
    )

    parameter_snapshot.to_csv(
        args.output_dir
        / "parameter_snapshot.csv",
        index=False,
    )

    make_primary_figure(
        summary,
        args.output_dir,
    )

    primary = summary[
        summary["metric"].isin(
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "log_loss",
                "brier_score",
            ]
        )
    ].copy()

    print("\n" + "=" * 82)
    print("AIM 2 CYCLE-DEPTH SIMULATION")
    print("=" * 82)

    print(
        "Replicates:",
        args.replicates,
    )

    print(
        "Synthetic participants per replicate:",
        args.participants,
    )

    print(
        "Empirical complete-cycle templates:",
        len(
            template_ids
        ),
    )

    print(
        "Calibration depths:",
        DEPTHS,
    )

    print("\nPARAMETERS")
    print(
        parameter_snapshot.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print("\nPERSONALIZED MINUS POPULATION")
    print(
        primary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print(
        "\nPRIMARY ENDPOINT: LOG LOSS"
    )

    if expected_depth is None:
        print(
            "Expected crossover: "
            "not reached by 8 cycles"
        )
    else:
        print(
            "Expected crossover:",
            expected_depth,
            "cycles",
        )

    if reliable_depth is None:
        print(
            "Reliable crossover (95% interval < 0): "
            "not reached by 8 cycles"
        )
    else:
        print(
            "Reliable crossover (95% interval < 0):",
            reliable_depth,
            "cycles",
        )

    print(
        "\nOutputs:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
