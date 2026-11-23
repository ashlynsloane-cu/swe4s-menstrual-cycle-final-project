"""Plot Aim 1 cycle-signature distances in the Aim 2 figure style.

Panel A contrasts same- and different-person pairs from the same study
interval. Panel B makes the corresponding cross-interval comparison.

These are descriptive distributions, NOT independent observations: each
cycle can contribute to many pairs. Do not infer p-values or confidence
intervals from the displayed boxes or pair counts.

The distance calculation and summary table retain the original method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mcphases.aim1_panel_inference import compare_panels
from mcphases.similarity import (
    calculate_pairwise_distances,
    select_repeated_cycle_cohort,
)

# Use the same blue/orange accents, typography, panel labels, and export
# settings used by Aim 2 Figures 1–3.
SAME_PERSON_COLOR = "#0072B2"
DIFFERENT_PERSON_COLOR = "#D55E00"
EDGE_COLOR = "#404040"

GROUPS = (
    (True, True, "Same person\nsame interval"),
    (True, False, "Same person\ncross interval"),
    (False, True, "Different people\nsame interval"),
    (False, False, "Different people\ncross interval"),
)

PANELS = (
    (True, "Within the same study interval"),
    (False, "Across study intervals"),
)


def summarize_distance_pairs(
    signatures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute original pairwise distances and four descriptive groups."""
    complete = (
        select_repeated_cycle_cohort(signatures)
        .dropna()
        .reset_index(drop=True)
    )
    if len(complete) < 3:
        raise ValueError("Not enough complete repeated-cycle signatures")

    pairs = calculate_pairwise_distances(complete)
    years = complete["study_interval"].to_numpy()
    pairs["same_interval"] = (
        years[pairs["index_a"].to_numpy()]
        == years[pairs["index_b"].to_numpy()]
    )

    rows = []
    for same_person, same_interval, label in GROUPS:
        subset = pairs.loc[
            (pairs["same_person"] == same_person)
            & (pairs["same_interval"] == same_interval),
            "distance",
        ].dropna()
        rows.append(
            {
                "category": label.replace("\n", " / "),
                "same_person": bool(same_person),
                "same_interval": bool(same_interval),
                "pair_count_not_independent_n": int(len(subset)),
                "mean_distance": (
                    float(subset.mean()) if len(subset) else np.nan
                ),
                "median_distance": (
                    float(subset.median()) if len(subset) else np.nan
                ),
                "q25": (
                    float(subset.quantile(0.25))
                    if len(subset) else np.nan
                ),
                "q75": (
                    float(subset.quantile(0.75))
                    if len(subset) else np.nan
                ),
            }
        )
    return pairs, pd.DataFrame(rows)


def panel_label(axis, label: str) -> None:
    """Use the bold, upper-left panel lettering of Aim 2 figures."""
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def draw_interval_panel(
    axis,
    pairs: pd.DataFrame,
    same_interval: bool,
    title: str,
    matched_cross_only: bool = False,
) -> None:
    """Draw comparably scaled, colored boxes for one interval stratum."""
    names = []
    allowed_pairs = pairs["same_interval"] == same_interval
    if not same_interval and matched_cross_only:
        same_cross = pairs.loc[
            ~pairs["same_interval"] & pairs["same_person"]
        ]
        shared_ids = same_cross["id_a"].unique()
        allowed_pairs &= pairs["id_a"].isin(shared_ids) & (
            pairs["id_b"].isin(shared_ids)
        )
    for index, (same_person, color, name) in enumerate(
        (
            (True, SAME_PERSON_COLOR, "Same person"),
            (False, DIFFERENT_PERSON_COLOR, "Different people"),
        ),
        start=1,
    ):
        values = pairs.loc[
            (pairs["same_person"] == same_person) & allowed_pairs,
            "distance",
        ].dropna().to_numpy()
        names.append(f"{name}\n{len(values):,} pairs")
        if not len(values):
            continue

        box = axis.boxplot(
            [values],
            positions=[index],
            widths=0.50,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            medianprops={"color": "#111111", "linewidth": 2.2},
            boxprops={"edgecolor": EDGE_COLOR, "linewidth": 1.1},
            whiskerprops={"color": EDGE_COLOR, "linewidth": 1.1},
            capprops={"color": EDGE_COLOR, "linewidth": 1.1},
        )
        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_alpha(0.80)

    axis.set_xticks([1, 2])
    axis.set_xticklabels(names, fontsize=10)
    axis.set_xlim(0.4, 2.6)
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.tick_params(axis="y", labelsize=10)
    axis.grid(axis="y", color="#D5D5D5", alpha=0.48, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def make_figure(pairs: pd.DataFrame, comparisons=None):
    """Build the two-panel figure without changing pairwise statistics."""
    if pairs.empty or not pairs["distance"].notna().any():
        raise ValueError("No valid pairwise distances to plot")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6.3),
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.78,
        bottom=0.20,
        wspace=0.23,
    )

    for axis, (same_interval, title), label in zip(
        axes,
        PANELS,
        ("A", "B"),
    ):
        draw_interval_panel(
            axis, pairs, same_interval, title,
            matched_cross_only=(comparisons is not None),
        )
        panel_label(axis, label)

    if comparisons is not None:
        if set(comparisons["contrast"]) != {
            "within_interval", "cross_interval"
        }:
            raise ValueError("Expected exactly the two prespecified contrasts")
        low, high = axes[0].get_ylim()
        height = high - low
        # Keep both annotation lines clear of the upper whisker without
        # extending the plot's actual distance scale differently by panel.
        axes[0].set_ylim(low, high + 0.27 * height)
        for axis, name in zip(
            axes, ("within_interval", "cross_interval")
        ):
            row = comparisons.loc[
                comparisons["contrast"] == name
            ].iloc[0]
            p_value = float(row["p_holm_two_contrasts"])
            advantage = float(row["similarity_advantage"])
            bracket_y = high + 0.030 * height
            bracket_height = 0.022 * height
            axis.plot(
                [1, 1, 2, 2],
                [
                    bracket_y,
                    bracket_y + bracket_height,
                    bracket_y + bracket_height,
                    bracket_y,
                ],
                color="#555555", linewidth=0.8, clip_on=False,
            )
            axis.text(
                1.5, high + 0.143 * height,
                f"Holm-adjusted p = {p_value:.4f}",
                ha="center", va="bottom", fontsize=8.6,
                fontweight="semibold", color="#333333",
            )
            axis.text(
                1.5, high + 0.095 * height,
                f"Mean advantage = {advantage:+.3f}",
                ha="center", va="bottom", fontsize=8.5,
                color="#555555",
            )

    axes[0].set_ylabel("Standardized Euclidean distance", fontsize=11)
    fig.suptitle(
        "Are physiological cycle signatures more similar within a person?",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    fig.supxlabel(
        "Lower distance = greater similarity  ·  "
        "Pair counts are not independent",
        fontsize=9.5,
        y=0.075,
    )
    return fig


def build_figure_caption(comparisons=None) -> str:
    """Keep methodological detail beside the figure, not in its footer."""
    paragraphs = [
        "**Aim 1 — Physiological cycle-signature distances.** "
        "Panel A compares cycles within the same study interval; "
        "Panel B compares cycles across the 2022 and 2024 intervals.",
        "Boxes indicate interquartile ranges, black lines mark medians, "
        "and whiskers extend to the most extreme observations within "
        "1.5 × IQR of each quartile. Outliers are hidden, not excluded "
        "from the underlying calculations. Lower standardized Euclidean "
        "distances indicate greater physiological similarity. Cycle pairs "
        "are dependent because individual cycles appear in multiple pairs; "
        "the displayed pair counts are not independent sample sizes.",
    ]
    if comparisons is not None:
        by_name = comparisons.set_index("contrast")
        a = by_name.loc["within_interval"]
        b = by_name.loc["cross_interval"]
        paragraphs.append(
            "The exploratory mean-distance advantage is defined as "
            "the mean different-person distance minus the mean "
            "same-person distance; positive values favor within-person "
            "similarity. Brackets show Holm-adjusted p-values for two "
            "prespecified one-sided randomization contrasts. "
            f"Panel A: advantage = {a['similarity_advantage']:+.3f}, "
            f"adjusted p = {a['p_holm_two_contrasts']:.4f}, "
            f"{int(a['n_participants_in_contrast'])} participants, "
            "with participant labels permuted among individual cycles "
            "within each study interval. "
            f"Panel B: advantage = {b['similarity_advantage']:+.3f}, "
            f"adjusted p = {b['p_holm_two_contrasts']:.4f}, "
            f"{int(b['n_participants_in_contrast'])} participants "
            "observed in both intervals, with entire later-interval "
            "participant profiles reassigned as blocks."
        )
        rejected = [
            name
            for name, row in (("A", a), ("B", b))
            if float(row["p_holm_two_contrasts"]) < 0.05
        ]
        if rejected:
            decision = (
                "Adjusted p < 0.05 for panel(s) "
                + ", ".join(rejected)
                + "; interpret only under the stated null assumptions."
            )
        else:
            decision = (
                "Neither adjusted p-value is below 0.05. "
                "This does not demonstrate equivalence or absence "
                "of a participant-specific effect."
            )
        paragraphs.append(
            "Both tests are exploratory and require the stated "
            "exchangeability assumptions. Multiplicity adjustment does "
            "not resolve dependence or confounding. The cross-interval "
            "matched-participant sample differs from the broader "
            "within-interval cohort, so the effect sizes should not be "
            "interpreted as a direct test of temporal change. "
            + decision
        )
    else:
        paragraphs.append(
            "This version is descriptive only; no significance test "
            "has been overlaid on the plotted distributions."
        )
    return "\n\n".join(paragraphs) + "\n"


def save_figure(fig, output_dir: Path) -> list[Path]:
    """Match Aim 2's 300-dpi PNG and vector PDF/SVG deliverables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "aim1_distance_boxes.png",
        output_dir / "aim1_distance_boxes.pdf",
        output_dir / "aim1_distance_boxes.svg",
    ]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    for path in paths[1:]:
        fig.savefig(path, bbox_inches="tight")
    return paths


def main() -> None:
    """Regenerate the original Aim 1 outputs with Aim 2-style visuals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/aim1"),
    )
    parser.add_argument(
        "--permutations", type=int, default=0,
        help=("Run conditional tests; 0 leaves plot descriptive"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--annotate-p", action="store_true",
        help="Display Holm-adjusted p-values; requires --permutations > 0",
    )
    args = parser.parse_args()
    if args.annotate_p and args.permutations < 1:
        parser.error("--annotate-p requires --permutations >= 1")

    # Headless, reproducible rendering; not placed among imports (PEP 8).
    plt.switch_backend("Agg")
    signatures = pd.read_csv(args.signatures)
    pairs, summary = summarize_distance_pairs(signatures)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "distance_box_summary.csv", index=False)

    comparisons = None
    if args.permutations > 0:
        comparisons, nulls = compare_panels(
            signatures, pairs, permutations=args.permutations,
            seed=args.seed,
        )
        comparisons.to_csv(
            args.output_dir / "aim1_panel_comparisons.csv", index=False
        )
        nulls.to_csv(
            args.output_dir / "aim1_panel_null_distributions.csv",
            index=False,
        )
        print("Exploratory panel comparisons:")
        print(comparisons.to_string(index=False))

    annotated = comparisons if args.annotate_p else None
    fig = make_figure(pairs, annotated)
    paths = save_figure(fig, args.output_dir)
    plt.close(fig)
    caption_path = args.output_dir / "aim1_distance_boxes_caption.md"
    caption_path.write_text(build_figure_caption(annotated))
    print(summary.to_string(index=False))
    print("Saved:")
    for path in paths:
        print(path)
    print(args.output_dir / "distance_box_summary.csv")
    print(caption_path)


if __name__ == "__main__":
    main()
