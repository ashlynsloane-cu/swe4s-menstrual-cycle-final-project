# Response to Moore–Fairborn SWE4S code review

Reviewed files: `features.py`, `similarity.py`, `aim2_split.py`, two
hierarchical-model drivers, and the primary evaluation script.

| Review finding | Response | Scope |
| --- | --- | --- |
| **Required:** identical probability extraction / result aggregation duplicated in Aim 2 drivers | Shared model-independent helpers in `scripts/aim2/hierarchical_shared.py`; preserved different folder globs and distinct fold construction | Refactor only; no scientific model changes |
| **Required:** tests of actual statistical properties of Aim 1 permutation test | Added fixed-seed null calibration and strong-effect detection simulation tests; validation of within-interval label multiplicities | Synthetic tests, not empirical estimates of real-data power |
| **Suggestion:** automate repeatability with Snakemake | Added `workflow/Snakefile` with preprocessing/Aim 1 default and separately opt-in, expensive Aim 2 targets. Subsequently expanded the workflow to include Aim 1 distance-metric sensitivity analysis and automated figure generation through `aim1_figures` | Ordinary test suite does not fit Bayesian models; default workflow remains unchanged |
| **Suggestion:** define units, metrics, and model objective | Added `docs/ANALYSIS_METHODS.md` explaining physical signal units (with uncertainty clearly flagged), dimensionless z-scores, likelihood, metrics and bootstrap | Documentation only |
| **Suggestion:** weighted standardized distances / Mahalanobis | Initially deferred to preserve the frozen primary analysis. Subsequently implemented regularized Mahalanobis distance with Ledoit-Wolf covariance shrinkage as a separate sensitivity analysis, retaining standardized Euclidean distance as the primary metric | Completed exploratory sensitivity analysis; original primary metric and default permutation procedure preserved |
| **Suggestion:** violin or box plots | Generated a two-panel Aim 1 box plot from locally available derived mcPHASES signatures. Extended the plotting implementation to support matched standardized Euclidean and regularized Mahalanobis boxplots, a signal-specific metric comparison, and a feature correlation heatmap | Versioned PNG/PDF/SVG, exploratory adjusted p-values, aggregate tables, and captions in `figures/aim1/` |
| **Suggestion:** bootstrap unique-fraction tuning | Explained why ordinary clustered bootstrap intentionally contains duplicated and omitted participants | No change to uncertainty estimator |
| **Preference:** flatten nested helper functions | Deferred where harmless; changes should preserve test coverage and frozen logic | Low priority |

The default analysis outputs and experimental definitions are preserved. The
new tests exercise functions in isolation and do not rerun computationally
expensive Bayesian models. A full rerun is required to validate scientific
output equivalence, not just the tests.

## Follow-up work on the code review

- Expanded Snakemake to optional, fully connected Aim 2 primary, longitudinal,
  oracle-diagnostics and simulation targets, while leaving the lightweight
  Aim 1 default unchanged.
- Added regression checks for fold-local standardization, participant-interval
  model group IDs, multiclass Brier/log-loss, phase-column ordering and
  posterior-free prediction-table construction for **both** hierarchical
  model drivers.
- Added a no-refit verification script that recomputes saved fold-level
  log loss from the explicitly named phase-probability columns and flags
  mismatches in a local aggregate report.
- Added an explicit probability-column-label alignment guard for fold-level
  log loss, plus an unsorted-label regression test. This may change prior
  fold-level log-loss numbers if their Bambi coordinate order was not sorted;
  subsequent audits found zero fold-score mismatches in the 38 primary
  and 18 longitudinal saved fold scores. This does not establish posterior
  refit equivalence.
- Added tests for the actual participant-cluster bootstrap estimator and
  caught/fixed asymmetric matched-prediction validation in primary Aim 2.
- Added runnable aggregate value summaries and global-versus-stratified
  Aim 1 permutation sensitivity diagnostics. These were subsequently
  executed locally on restricted derived mcPHASES signatures; no raw
  participant-level data are included in the repository.
- Documented that the dataset provides computed- and wrist-temperature
  metrics separately; exact units and absolute/deviation conventions remain
  pending confirmation from authorized source metadata.

See [`AIM2_WORKFLOW.md`](AIM2_WORKFLOW.md) and
[`STATISTICAL_AUDIT.md`](STATISTICAL_AUDIT.md).

### Aim 1 distance-metric sensitivity analysis

In response to the reviewers' suggestion to evaluate alternative
standardized distance metrics, regularized Mahalanobis distance was
implemented as a sensitivity analysis.

Standardized Euclidean distance accounts for differences in feature
variance but does not account for covariance between features.
Regularized Mahalanobis distance incorporates the covariance structure
of cohort-standardized physiological signatures using Ledoit-Wolf
shrinkage to stabilize covariance estimation.

This sensitivity analysis preserves the original standardized Euclidean
analysis and evaluates both metrics using the same complete-case
repeated-cycle cohort.

Implementation and software engineering improvements include:

- Added `src/mcphases/mahalanobis.py` for modular regularized covariance
  estimation and pairwise Mahalanobis distance calculations.
- Added `src/mcphases/run_aim1_mahalanobis.py` as an independent
  sensitivity-analysis runner, preserving the primary analysis outputs.
- Extended `scripts/aim1/plot_aim1_distance_boxes.py` to support both
  distance metrics while retaining standardized Euclidean distance
  as the default.
- Added `scripts/aim1/plot_aim1_metric_comparison.py` to visualize
  signal-specific similarity advantages across both distance metrics.
- Added `scripts/aim1/plot_aim1_feature_correlations.py` to visualize
  correlations among the 16 centered physiological signature features.
- Organized the Aim 1 plotting scripts under `scripts/aim1/` and
  updated the corresponding test references.
- Added `tests/test_mahalanobis.py` to test covariance estimation,
  distance calculations, and signal-specific analysis.
- Added `tests/test_run_aim1_mahalanobis.py` to test the sensitivity
  analysis runner and generated output files.
- Added `tests/test_aim1_sensitivity_figures.py` to test cohort
  consistency, interval-specific inference, metric comparison plots,
  and correlation heatmap generation.
- Expanded `workflow/Snakefile` to include the sensitivity analysis
  and all Aim 1 figures through the optional `aim1_figures` target.

### Aim 1 statistical inference

The original pooled permutation procedure is retained for the primary
analysis and applied to the additional distance metric.

Signal-specific comparisons use 10,000 permutations and report
exploratory, unadjusted p-values.

The interval-specific figure contrasts use 9,999 permutations with
random seed 42. Holm correction is applied across the two interval
comparisons within each distance metric.

The within-interval comparison permutes cycle-level participant labels
separately within study intervals. The cross-interval comparison
reassigns complete later-interval participant profiles as blocks among
participants observed in both study intervals.

Both distance metrics use the same interval-specific permutation
procedures. Pairwise distances are not independent observations, and
validity depends on the exchangeability assumptions associated with
each permutation procedure.

## Post-review local verification

The final figure is recorded under [`../figures/aim1/`](../figures/aim1/)
with the caption, aggregate p-values, and reproducibility command.
The within-interval Holm-adjusted p-value was 0.2282 (25 participants);
the cross-interval Holm-adjusted p-value was 0.2188 (14 matched
participants). These are exploratory and neither is below 0.05.

Local checks on saved, untracked derived results found **0/56 fold-level
log-loss mismatches** (38 primary, 18 longitudinal) and **0/28
population-training/target-ID leakage failures** (19 primary and nine
longitudinal folds). These checks do not guarantee that every upstream
source of leakage was excluded, or that Bayesian refitting preserves
all past outputs. A separate small permutation power grid was a
functional smoke test, not evidence of well-estimated power.

### Aim 1 sensitivity-analysis verification

The additional regularized Mahalanobis sensitivity analysis was
executed locally using restricted derived mcPHASES signatures.

Both distance metrics produced positive within-person similarity
advantages within and across study intervals. Neither interval-specific
comparison reached statistical significance after Holm correction
under either distance metric.

Exploratory signal-specific comparisons produced consistent directions
and ordering among individual physiological signals across both metrics.
Resting heart rate exhibited the largest observed similarity advantage,
followed by nightly temperature and HRV RMSSD. Respiratory rate
exhibited a negative similarity advantage under both metrics.

None of the signal-specific permutation comparisons reached an
unadjusted 0.05 significance threshold.

The final Aim 1 figure exports include:

- [Standardized Euclidean distance boxplots](../figures/aim1/aim1_distance_boxes.png)
- [Regularized Mahalanobis distance boxplots](../figures/aim1/mahalanobis/aim1_distance_boxes.png)
- [Signal-specific distance metric comparison](../figures/aim1/aim1_metric_comparison.png)
- [Physiological feature correlation heatmap](../figures/aim1/aim1_feature_correlations.png)

The corresponding aggregate statistical outputs are available in:

- [Standardized Euclidean interval comparisons](../figures/aim1/aim1_panel_comparisons.csv)
- [Regularized Mahalanobis interval comparisons](../figures/aim1/mahalanobis/aim1_panel_comparisons.csv)
- [Signal-specific distance metric comparisons](../figures/aim1/aim1_metric_comparison.csv)
- [Physiological feature correlation matrix](../figures/aim1/aim1_feature_correlations.csv)

### Software verification

Following implementation of the distance-metric sensitivity analysis:

| Validation | Result |
| --- | --- |
| Full local pytest suite | 111 tests passed |
| PEP 8 checks of additional Aim 1 code | Passed |
| Regularized covariance and distance calculations | Unit tested |
| Cycle-pair consistency across distance metrics | Unit tested |
| Mahalanobis interval-specific permutation inference | Unit tested |
| Aim 1 figure generation | All four figure sets generated |
| Snakemake DAG construction | Dry run passed |

On November 26, 2026, the Aim 1 sensitivity analysis and figure
generation workflow was successfully rerun using Snakemake 9.22.0
under Python 3.12.16. All six scheduled workflow jobs completed
successfully, reproducing the interval-specific statistical results
and regenerating all four Aim 1 figure sets from existing local
preprocessed data and cycle signatures. No Bayesian Aim 2 models
were refitted.

The additional analysis and figures can be reproduced from authorized
local mcPHASES source data through the `aim1_figures` Snakemake target.

The default Snakemake workflow remains unchanged, and the additional
Aim 1 sensitivity analysis does not require fitting the Bayesian Aim 2
models.

### Interpretation and limitations

The consistency of observed similarity patterns across distance metrics
provides descriptive evidence that the direction of the findings is not
specific to standardized Euclidean distance.

However:

- Neither interval-specific contrast reached statistical significance
  after Holm correction.
- Signal-specific analyses are exploratory and report unadjusted
  permutation p-values.
- Raw similarity advantages obtained using different distance metrics
  are not directly comparable in magnitude.
- Cycle centering can induce correlations among phase-specific features.
- The correlation heatmap motivates covariance-aware distance estimation
  but does not establish a causal explanation for differences between
  distance metrics.
- The within-interval and cross-interval analyses involve different
  participant cohorts and do not provide a direct test of temporal
  physiological drift.
- Synthetic tests and consistency across distance metrics do not
  establish exchangeability, statistical equivalence, or empirical power.

Exact computed-temperature units remain unverified. See
[`STATISTICAL_AUDIT.md`](STATISTICAL_AUDIT.md) for assumptions and limits.