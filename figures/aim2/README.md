# Aim 2: phase prediction and personalization

This directory contains version-controlled exports of three figures.
The underlying data, posterior draws, and generated run outputs are kept
outside Git; this directory is not an output destination for model fitting.

## Figures

| Figure | Question | Files |
| --- | --- | --- |
| 1 — Near-term personalization | Does adding two earlier cycles improve subsequent menstrual-phase prediction within a study interval? | [PNG](figure1_near_term_personalization.png), [PDF](figure1_near_term_personalization.pdf) |
| 2 — Longitudinal personalization | How well do individualized physiological profiles learned in 2022 transfer to 2024, and what mechanisms explain limitations? | [PNG](figure2_longitudinal_mechanisms.png), [PDF](figure2_longitudinal_mechanisms.pdf) |
| 3 — Calibration-cycle simulation | How do stable person-specific signal and cycle variability affect the amount of calibration needed? | [PNG](figure3_personalization_thresholds.png), [PDF](figure3_personalization_thresholds.pdf) |

## Reproduction

The source scripts are:

- `scripts/aim2/make_figure1_near_term_personalization.py`
- `scripts/aim2/make_figure2_longitudinal_mechanisms.py`
- `scripts/aim2/make_figure3_personalization_thresholds.py`

These scripts normally write derived figures under
`results/figures/aim2/`, which is intentionally Git-ignored. The files
in this directory are the selected version-controlled exports, not
outputs that Snakemake writes into the repository. To reproduce them,
follow [`docs/AIM2_WORKFLOW.md`](../../docs/AIM2_WORKFLOW.md) and the
[main README](../../README.md), using authorized local mcPHASES data.
The opt-in Bayesian model runs and simulations may be computationally
expensive; `--dry-run` inspects their Snakemake dependencies without
refitting models.

## Interpretation

The empirical Aim 2 evaluations did not demonstrate an overall advantage
from two-cycle personalization. The threshold simulations describe
conditional behavior under their specified data-generating assumptions,
not a universal number of calibration cycles. Full methods and limitations
are documented in [`docs/ANALYSIS_METHODS.md`](../../docs/ANALYSIS_METHODS.md).
