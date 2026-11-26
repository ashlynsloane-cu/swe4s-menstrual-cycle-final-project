# Beyond the Average Cycle

**Person-specific physiological signatures across the menstrual cycle using mcPHASES**

This repository contains the final project for **CSCI 6118 / Software Engineering for Scientists**. It uses wearable and hormone-derived data from the mcPHASES dataset to study whether menstrual-cycle physiology is reproducible within individuals and whether person-specific information improves prediction of future menstrual phase.

## Project status

The scientific analyses for this project are complete.

You **do not need to rerun the Bayesian models or simulations** to use, review, or grade this repository. The final analysis code, figures, tests, and reproducibility workflow are version-controlled here. Computationally expensive model outputs are intentionally not committed.

The repository contains:

- finalized preprocessing and cycle-quality-control code;
- Aim 1 analyses of within-person physiological reproducibility;
- Aim 1 distance-metric sensitivity analyses using regularized Mahalanobis distance;
- Aim 2 population and personalized phase-prediction models;
- longitudinal 2022-to-2024 personalization analyses;
- simulation experiments examining how many calibration cycles are needed for personalization;
- final presentation-ready figures;
- tests and GitHub Actions configuration.

## Scientific questions

### Aim 1

**Are menstrual-cycle physiological signatures reproducible within the same person across cycles?**

The analysis focuses on four wearable-derived physiological signals:

- resting heart rate;
- HRV RMSSD;
- computed nightly temperature (exact source unit/convention unverified);
- respiratory rate.

Hormone measurements are used to define menstrual-cycle phase, but they are not used as predictors.

Each menstrual cycle is represented by a 16-dimensional physiological signature constructed from four physiological signals across four menstrual cycle phases. Phase-specific means are centered relative to the corresponding cycle-level mean.

Standardized Euclidean distance is used as the primary similarity metric, accounting for differences in feature variance. Regularized Mahalanobis distance is used as a sensitivity analysis, accounting for covariance among physiological features using Ledoit-Wolf shrinkage.

Similarity advantage is defined as:

    mean(between-person distance) - mean(within-person distance)

Positive values indicate greater within-person physiological similarity.

### Aim 2

**When is person-specific wearable information useful for predicting future menstrual phase?**

Aim 2 compares:

1. a **population-only model**, trained without data from the target participant; and
2. a **personalized hybrid model**, which adds information from the target participant's prior cycles.

The four prediction classes are Menstrual, Follicular, Fertility, and Luteal.

## Main Aim 2 findings

The empirical analyses did **not** show that two-cycle personalization improved overall prediction of future menstrual phase.

In the near-term within-interval analysis, personalization did not improve overall discrimination and produced worse probabilistic calibration.

In the longitudinal analysis, person-specific information learned from 2022 did not transfer well to the same participants in 2024. Baseline physiological drift and changes in phase-specific patterns contributed to this failure.

Simulation experiments provide an important qualification: under a stationary data-generating process resembling the observed mcPHASES variance structure, approximately **two calibration cycles** were sufficient for a reliable personalization benefit in the nominal scenario. However, the required number of cycles changed substantially when the strength of stable person-specific signal or cycle-to-cycle variability changed.

Therefore, **two cycles is not a universal rule**. More calibration data can reduce transient cycle noise, but it cannot solve physiological drift when the personalized signature itself changes over time.

## Repository structure

```text
.
├── .github/workflows/test.yml
├── assets/aim2/
├── data/README.md
├── docs/
├── figures/
│   ├── aim1/
│   │   └── mahalanobis/
│   └── aim2/
├── scripts/
│   ├── aim1/
│   ├── aim2/
│   └── review/
├── src/mcphases/
├── tests/
├── workflow/Snakefile
├── environment.yml
├── pyproject.toml
└── README.md
```

## Figures

Figures from both aims are collected under [`figures/`](figures/README.md).
These are version-controlled exports; generated intermediate files and
participant-level outputs remain outside Git in `results/`.

| Aim | Figure | Formats |
| --- | --- | --- |
| Aim 1 | [Physiological signature distances](figures/aim1/aim1_distance_boxes.png) within and across study intervals | [PNG](figures/aim1/aim1_distance_boxes.png), [PDF](figures/aim1/aim1_distance_boxes.pdf), [SVG](figures/aim1/aim1_distance_boxes.svg) |
| Aim 1 | [Regularized Mahalanobis distance sensitivity analysis](figures/aim1/mahalanobis/aim1_distance_boxes.png) within and across study intervals | [PNG](figures/aim1/mahalanobis/aim1_distance_boxes.png), [PDF](figures/aim1/mahalanobis/aim1_distance_boxes.pdf), [SVG](figures/aim1/mahalanobis/aim1_distance_boxes.svg) |
| Aim 1 | [Signal-specific distance metric comparison](figures/aim1/aim1_metric_comparison.png) | [PNG](figures/aim1/aim1_metric_comparison.png), [PDF](figures/aim1/aim1_metric_comparison.pdf), [SVG](figures/aim1/aim1_metric_comparison.svg) |
| Aim 1 | [Physiological feature correlation heatmap](figures/aim1/aim1_feature_correlations.png) | [PNG](figures/aim1/aim1_feature_correlations.png), [PDF](figures/aim1/aim1_feature_correlations.pdf), [SVG](figures/aim1/aim1_feature_correlations.svg) |
| Aim 2 | [Figure 1: Near-term personalization](figures/aim2/figure1_near_term_personalization.png) | [PNG](figures/aim2/figure1_near_term_personalization.png), [PDF](figures/aim2/figure1_near_term_personalization.pdf) |
| Aim 2 | [Figure 2: Longitudinal personalization](figures/aim2/figure2_longitudinal_mechanisms.png) | [PNG](figures/aim2/figure2_longitudinal_mechanisms.png), [PDF](figures/aim2/figure2_longitudinal_mechanisms.pdf) |
| Aim 2 | [Figure 3: Calibration-cycle simulation](figures/aim2/figure3_personalization_thresholds.png) | [PNG](figures/aim2/figure3_personalization_thresholds.png), [PDF](figures/aim2/figure3_personalization_thresholds.pdf) |

**Aim 1** compares standardized Euclidean distances between cycle
signatures from the same and different participants. Exploratory mean
similarity advantages are positive within intervals (+0.112; Holm-adjusted
p = 0.2282) and across intervals (+0.269; adjusted p = 0.2188), but neither
contrast meets a 0.05 threshold. See the [caption](figures/aim1/aim1_distance_boxes_caption.md),
[comparison table](figures/aim1/aim1_panel_comparisons.csv), and
[reproduction notes](figures/aim1/README.md) for cohort sizes, permutation
units, dependency, and exchangeability assumptions.

To assess sensitivity to the choice of distance metric, the analysis was
repeated using regularized Mahalanobis distance on the same complete-case
repeated-cycle cohort. The direction of the within-person similarity
advantage was preserved within and across study intervals, although
neither comparison reached statistical significance after Holm correction.

See the [Mahalanobis figure caption](figures/aim1/mahalanobis/aim1_distance_boxes_caption.md)
and [comparison table](figures/aim1/mahalanobis/aim1_panel_comparisons.csv)
for the corresponding interval-specific results.

Exploratory signal-specific comparisons showed consistent ordering among
individual physiological signals across both distance metrics. Resting
heart rate exhibited the largest observed similarity advantage, followed
by nightly temperature and HRV RMSSD. Respiratory rate exhibited a negative
similarity advantage under both metrics. None of the signal-specific
comparisons reached an unadjusted 0.05 significance threshold.

The [signal-specific comparison table](figures/aim1/aim1_metric_comparison.csv)
contains the corresponding similarity advantages and permutation p-values.
The [feature correlation matrix](figures/aim1/aim1_feature_correlations.csv)
illustrates covariance among the 16 centered physiological features and
motivates the use of a covariance-aware sensitivity analysis.

Raw distance advantages obtained using different metrics are not directly
comparable in magnitude.

**Aim 2** evaluates population-only versus personalized phase prediction
within an interval (Figure 1), longitudinal transfer across study years
(Figure 2), and simulation-based calibration requirements (Figure 3).
See [Aim 2 figure descriptions](figures/aim2/README.md) and the
[computational workflow](docs/AIM2_WORKFLOW.md) for reproduction details.

## Data and generated results

The mcPHASES source data are not committed to this repository.

Generated analysis outputs under `results/` are also excluded from Git by default. This includes large or restricted artifacts such as participant-level prediction files, posterior NetCDF files, bootstrap replicate tables, simulation replicate tables, temporary model outputs, and run logs.

This keeps the repository small and avoids committing restricted or reproducible derived data. The final figures and the code required to regenerate the analyses are version-controlled.

## Reproducibility and software engineering

The repository was structured around reproducible scientific software practices, including:

- modular preprocessing and analysis code;
- explicit train/calibration/test separation;
- leakage-safe population training;
- deterministic simulation seeds;
- unit tests for core data-splitting logic;
- modular distance-metric implementations;
- regression tests for covariance estimation, distance calculations, permutation inference, and figure generation;
- automated analysis and figure generation through Snakemake;
- PEP 8 style checks;
- Python compilation checks;
- GitHub Actions continuous integration;
- explicit environment configuration;
- version-controlled final figures;
- separation of generated results from source code.

The primary environment target is **Python 3.12**. The final analyses were also executed successfully in a local development environment using Python 3.14.

## Installation

Installation is only needed for development or reproduction.

### Conda

```bash
conda env create -f environment.yml
conda activate beyond-average-cycle
```

### pip

```bash
python -m pip install -e ".[dev]"
```

To install the optional Snakemake workflow dependencies:

```bash
python -m pip install -e ".[dev,workflow]"
```

## Running tests

The normal repository validation step is lightweight and does **not** rerun the Bayesian analyses.

```bash
python -m pytest -q
python -m pycodestyle src tests scripts/aim1/*.py scripts/aim2/*.py scripts/review/*.py
python -m py_compile scripts/aim1/*.py scripts/aim2/*.py scripts/review/*.py
git diff --check
```

GitHub Actions performs the corresponding lightweight checks under Python 3.12.

Following the addition of the Aim 1 distance-metric sensitivity analysis
and visualization tests, the local test suite passed all **111 tests**.

## Optional: full scientific reproduction

The commands in this section are **only** for someone who intentionally wants to regenerate the scientific analyses from the required local mcPHASES data.

**These steps are not required for normal repository use, code review, grading, or figure inspection.**

### Aim 1 analysis

The complete Aim 1 analysis and figure workflow is available through the
optional `aim1_figures` Snakemake target.

```bash
python -m pip install -e ".[dev,workflow]"

snakemake -s workflow/Snakefile --cores 1 aim1_figures --dry-run

snakemake -s workflow/Snakefile --cores 1 aim1_figures
```

This target generates:

- preprocessing and cycle-quality-control outputs;
- centered physiological cycle signatures;
- standardized Euclidean distance analysis;
- regularized Mahalanobis distance sensitivity analysis;
- signal-specific permutation comparisons;
- interval-specific distance boxplots for both metrics;
- signal-specific distance metric comparison figure;
- physiological feature correlation heatmap;
- aggregate statistical tables and figure captions.

Generated outputs are stored under:

```text
results/
├── preprocessed/
├── aim1/
├── aim1_mahalanobis/
└── figures/
    └── aim1/
        ├── euclidean/
        └── mahalanobis/
```

The signal-specific comparison and feature correlation figures are
generated directly under `results/figures/aim1/`.

Final presentation-ready exports and aggregate supporting tables are
version-controlled under `figures/aim1/`. Generated intermediate files
remain outside Git.

The analyses can also be reproduced individually using the Python
analysis modules and plotting scripts.

#### Primary standardized Euclidean analysis

```bash
python -m mcphases.run_aim1
```

#### Regularized Mahalanobis sensitivity analysis

```bash
python -m mcphases.run_aim1_mahalanobis \
  --output-dir results/aim1_mahalanobis \
  --permutations 10000
```

#### Standardized Euclidean distance boxplots

```bash
python scripts/aim1/plot_aim1_distance_boxes.py \
  --signatures results/aim1/cycle_signatures.csv \
  --distance-metric standardized_euclidean \
  --output-dir results/figures/aim1/euclidean \
  --permutations 9999 \
  --seed 42 \
  --annotate-p
```

#### Regularized Mahalanobis distance boxplots

```bash
python scripts/aim1/plot_aim1_distance_boxes.py \
  --signatures results/aim1/cycle_signatures.csv \
  --distance-metric regularized_mahalanobis \
  --output-dir results/figures/aim1/mahalanobis \
  --permutations 9999 \
  --seed 42 \
  --annotate-p
```

Both distance metrics use the same primary signature table to ensure that
the interval-specific comparisons are evaluated on an identical cohort.

#### Signal-specific distance metric comparison

```bash
python scripts/aim1/plot_aim1_metric_comparison.py \
  --euclidean results/aim1/signal_specific_reproducibility.csv \
  --mahalanobis results/aim1_mahalanobis/signal_specific_reproducibility.csv \
  --output-dir results/figures/aim1
```

#### Physiological feature correlation heatmap

```bash
python scripts/aim1/plot_aim1_feature_correlations.py \
  --signatures results/aim1/cycle_signatures.csv \
  --output-dir results/figures/aim1
```

Each plotting script exports figures in PNG, PDF, and SVG formats,
together with the corresponding aggregate statistical tables.

### Near-term Aim 2 analysis

```bash
python3 scripts/aim2/build_within_interval_manifest.py
python3 scripts/aim2/build_centered_primary_folds.py
PYTENSOR_FLAGS="cxx=" python3 scripts/aim2/run_primary_hierarchical_models.py
python3 scripts/aim2/evaluate_primary_predictions.py
python3 scripts/aim2/make_figure1_near_term_personalization.py
```

### Longitudinal Aim 2 analysis

```bash
python3 scripts/aim2/build_longitudinal_manifest.py
python3 scripts/aim2/build_longitudinal_folds.py
PYTENSOR_FLAGS="cxx=" python3 scripts/aim2/run_longitudinal_hierarchical_models.py
python3 scripts/aim2/evaluate_longitudinal_predictions.py
python3 scripts/aim2/estimate_phase_profile_repeatability.py
python3 scripts/aim2/diagnose_empirical_profile_transfer.py
python3 scripts/aim2/diagnose_personalization_failure.py
PYTENSOR_FLAGS="cxx=" python3 scripts/aim2/oracle_recenter_longitudinal.py
python3 scripts/aim2/diagnose_residual_phase_drift.py
python3 scripts/aim2/make_figure2_longitudinal_mechanisms.py
```

`oracle_recenter_longitudinal.py` is a retrospective diagnostic that uses the observed 2024 baseline. It is not a prospective prediction model.

### Simulation analysis

```bash
python3 scripts/aim2/estimate_simulation_components.py
python3 scripts/aim2/estimate_phase_profile_repeatability.py
python3 scripts/aim2/run_cycle_depth_simulation.py
python3 scripts/aim2/run_cycle_depth_sensitivity.py
python3 scripts/aim2/run_variance_cycle_thresholds.py
python3 scripts/aim2/run_nominal_precision_check.py
python3 scripts/aim2/make_figure3_personalization_thresholds.py
```

On the development Mac, PyTensor model fitting used:

```bash
PYTENSOR_FLAGS="cxx=" python3 <script>
```

This is a platform-specific workaround and may not be necessary on other systems.

## Reproducibility verification

The final longitudinal reproducibility rerun regenerated all nine longitudinal folds. The resulting aggregate fold-results file and prediction file were **byte-for-byte identical** to the preserved original outputs under SHA-256 comparison.

The longitudinal rerun produced:

- 18 fitted models across 9 target participants;
- 0 divergences;
- worst observed maximum R-hat of 1.0155;
- minimum bulk ESS of 698.6;
- minimum tail ESS of 736.5;
- 701 held-out 2024 evaluation days.

The Aim 1 sensitivity analysis was executed locally using authorized
derived mcPHASES data. All four Aim 1 figure sets were generated,
the expanded test suite passed 111 tests, and the `aim1_figures`
Snakemake dry run successfully resolved the required workflow
dependencies.

## Development scripts

Exploratory and pilot scripts are retained under:

```text
scripts/aim2/development/
```

They are preserved for provenance but are not part of the final supported analysis workflow.

## Interpretation

The project distinguishes among three different questions:

1. whether physiological features vary systematically across menstrual phase;
2. whether those patterns are reproducible within individuals;
3. whether learning an individual's prior pattern improves prediction of future phase.

These are not equivalent. A participant can show a recognizable phase-associated physiological pattern while still providing little transferable predictive benefit if the magnitude, baseline, or direction of that pattern changes across cycles or years.

## License and data access

This repository contains analysis code and derived figures for the course project. Access to the original mcPHASES dataset is governed separately by the dataset's distribution and data-use terms.


## Code-review improvements (Moore–Fairborn)

The review response records completed changes, evidence, and remaining
limitations. Start with [the itemized response](docs/CODE_REVIEW_RESPONSE.md)
and [analysis methods](docs/ANALYSIS_METHODS.md); the
[statistical audit](docs/STATISTICAL_AUDIT.md) distinguishes code
regression tests from empirical checks and assumptions. The
[Aim 2 workflow guide](docs/AIM2_WORKFLOW.md) explains expensive model
dependencies and expected output paths.

Improvements include shared Aim 2 modeling helpers, label-aligned
multiclass log-loss calculations, stricter paired-prediction matching,
participant-level leakage audits, bootstrap concentration diagnostics,
Aim 1 null-sensitivity checks, exploratory power simulations, and
additional tests. The tests and audits are not substitutes for
re-estimating Bayesian posteriors.

### Aim 1 distance-metric sensitivity analysis

In response to the reviewers' recommendation to evaluate alternative
standardized distance metrics, regularized Mahalanobis distance was
implemented as a sensitivity analysis while retaining standardized
Euclidean distance as the primary analysis.

Standardized Euclidean distance accounts for differences in variance
among physiological features but does not account for covariance between
features. Regularized Mahalanobis distance incorporates the covariance
structure of cohort-standardized physiological signatures using
Ledoit-Wolf shrinkage, improving the stability of covariance estimation.

The primary and sensitivity analyses use the same complete-case
repeated-cycle cohort. The original primary analysis and its default
permutation procedure remain unchanged.

The implementation includes:

- a modular regularized Mahalanobis distance implementation in
  `src/mcphases/mahalanobis.py`;
- a separate sensitivity-analysis runner in
  `src/mcphases/run_aim1_mahalanobis.py`;
- matched interval-specific boxplots using both distance metrics;
- signal-specific comparisons of similarity advantages;
- a correlation heatmap of the 16 centered physiological features;
- unit tests for regularized covariance estimation, distance calculation,
  cohort consistency, permutation inference, and visualization;
- reproducible analysis and figure generation through Snakemake.

The Aim 1 plotting scripts are organized under `scripts/aim1/`.

The original distance boxplot script was extended to support both
distance metrics while retaining standardized Euclidean distance as its
default. This avoids duplicating the plotting and statistical inference
implementations.

### Aim 1 statistical inference

Two complementary permutation-analysis approaches are retained.

The pooled signal-specific analysis uses 10,000 permutations with
cycle-level participant-label shuffling. These analyses are exploratory
and report unadjusted permutation p-values.

The interval-specific figure contrasts use 9,999 permutations and
random seed 42, with Holm correction across the two comparisons
within each distance metric.

The within-interval comparison permutes cycle-level participant labels
separately within study intervals. The cross-interval comparison
reassigns complete later-interval participant profiles as blocks
among participants observed in both study intervals.

The same interval-specific permutation procedures are applied to
standardized Euclidean and regularized Mahalanobis distances.

Pairwise distances are not independent observations. The permutation
procedures preserve the structures specified by their respective null
hypotheses, and their validity depends on the associated
exchangeability assumptions.

### Empirical validation of saved results

After the review changes, saved local Aim 2 fold scores were checked
without refitting: **38 near-term and 18 longitudinal fold-level
log-loss values (56 total), zero mismatches** against recalculation
from the explicitly named phase-probability columns (within numerical
precision). The split-leakage audit passed **19/19 near-term and 9/9
longitudinal** folds. The primary and longitudinal cluster-bootstrap
concentration checks used 5,000 resamples each.

The Aim 1 interval-sensitivity analysis and the two exploratory
figure contrasts were also run on locally available restricted
derived signatures. The small power grid (five simulation replicates
and 99 permutations per condition) is **a functional smoke test,
not a calibrated power study**. These checks establish only the
specific properties tested; they do not prove exchangeability,
prospective generalization, or equivalence after Bayesian refitting.

The additional Aim 1 distance-metric sensitivity analysis was also
executed on locally available restricted derived signatures. Positive
within-person similarity advantages were observed within and across
study intervals using both distance metrics, although neither
interval-specific comparison reached statistical significance after
Holm correction.

The exploratory signal-specific results were:

| Physiological signal | Standardized Euclidean advantage | Unadjusted p | Regularized Mahalanobis advantage | Unadjusted p |
| --- | ---: | ---: | ---: | ---: |
| Resting heart rate | +0.1694 | 0.0660 | +0.1551 | 0.0534 |
| Nightly temperature | +0.1218 | 0.1493 | +0.1155 | 0.1498 |
| HRV RMSSD | +0.1044 | 0.2555 | +0.0764 | 0.3016 |
| Respiratory rate | -0.0613 | 0.6638 | -0.0546 | 0.6566 |
| All signals | +0.1197 | 0.2318 | +0.0544 | 0.3417 |

The ordering of individual physiological signals and the direction of
their observed similarity advantages were consistent across both
distance metrics. These results are descriptive and do not establish
statistically significant within-person reproducibility.

Numerical advantages from different distance metrics are presented for
transparency but should not be interpreted as directly comparable
effect sizes.

The expanded local test suite passed **111 tests**, including tests
for the regularized Mahalanobis implementation, analysis runner,
matched cycle pairs, interval-specific inference, and figure
generation. PEP 8 checks of the newly added Aim 1 code also passed.

### Snakemake: preprocessing and Aim 1 by default

All Snakemake targets are optional and require authorized local
mcPHASES source files. Running Snakemake with **no target** invokes
preprocessing and Aim 1 only; it does not fit the Bayesian Aim 2
models. A dry-run plans work without executing it, but still
requires the expected source filenames.

```bash
python -m pip install -e ".[dev,workflow]"
snakemake --snakefile workflow/Snakefile --cores 1 --config data_dir=data/mcphases --list-target-rules
snakemake --snakefile workflow/Snakefile --cores 1 --config data_dir=data/mcphases --dry-run
```

The complete Aim 1 sensitivity analysis and figure workflow is
available through the optional `aim1_figures` target.

```bash
snakemake --snakefile workflow/Snakefile --cores 1 --config data_dir=data/mcphases --dry-run aim1_figures
```

To execute the Aim 1 analysis and figure workflow:

```bash
snakemake --snakefile workflow/Snakefile --cores 1 --config data_dir=data/mcphases aim1_figures
```

The additional workflow rules include:

- `aim1_mahalanobis`: regularized Mahalanobis sensitivity analysis;
- `aim1_distance_boxplots`: interval-specific boxplots and inference
  for both distance metrics;
- `aim1_metric_comparison`: signal-specific distance metric comparison;
- `aim1_feature_correlations`: physiological feature correlation heatmap;
- `aim1_figures`: aggregate target for the complete Aim 1 figure workflow.

The default Snakemake target remains unchanged. The additional Aim 1
figure workflow does not require fitting the Bayesian Aim 2 models.

Aim 2 is explicitly opt-in through `aim2_primary`,
`aim2_longitudinal`, `aim2_simulation`, or `aim2`:

```bash
snakemake --snakefile workflow/Snakefile --cores 1 --config data_dir=data/mcphases --dry-run aim2
```

Use `--config data_dir=/authorized/path` when files are stored
elsewhere. **Do not omit `--dry-run` for Aim 2 unless you intend to
refit expensive Bayesian models and simulations.** No target
downloads the restricted data.

### Outstanding scientific limitations

- The source unit and absolute-versus-relative convention of
  `computed_temperature.csv:nightly_temperature` remain unverified.
- Aim 1 cycle-label permutation tests have exchangeability assumptions
  not established by unit tests or the exploratory figure.
- Historical fold-score agreement does not establish posterior refit
  equivalence. The Bayesian reruns require a separate comparison of
  model outputs and numerical diagnostics.
- Cluster-bootstrap intervals use participant resampling with
  replacement; this is appropriate to the chosen estimator but
  does not eliminate small-cohort uncertainty.
- Raw similarity advantages from different distance metrics are not
  directly comparable in magnitude.
- Centering physiological phase means relative to the cycle-level mean
  can induce correlations among phase-specific features.
- The feature correlation heatmap motivates covariance-aware distance
  estimation but does not establish a causal explanation for differences
  between distance metrics.
- Signal-specific permutation analyses are exploratory and report
  unadjusted p-values.
- Consistency across distance metrics does not establish statistical
  significance or equivalence.
- The within-interval and cross-interval comparisons involve different
  participant cohorts and should not be interpreted as a direct test
  of temporal physiological drift.

See [the statistical audit](docs/STATISTICAL_AUDIT.md) for scope,
qualification, and instructions for additional data-dependent checks.