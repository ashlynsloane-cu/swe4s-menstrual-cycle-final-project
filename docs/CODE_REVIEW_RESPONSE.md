# Response to Moore–Fairborn SWE4S code review

Reviewed files: `features.py`, `similarity.py`, `aim2_split.py`, two
hierarchical-model drivers, and the primary evaluation script.

| Review finding | Response | Scope |
| --- | --- | --- |
| **Required:** identical probability extraction / result aggregation duplicated in Aim 2 drivers | Shared model-independent helpers in `scripts/aim2/hierarchical_shared.py`; preserved different folder globs and distinct fold construction | Refactor only; no scientific model changes |
| **Required:** tests of actual statistical properties of Aim 1 permutation test | Added fixed-seed null calibration and strong-effect detection simulation tests; validation of within-interval label multiplicities | Synthetic tests, not empirical estimates of real-data power |
| **Suggestion:** automate repeatability with Snakemake | Added `workflow/Snakefile` with preprocessing/Aim 1 default and separately opt-in, expensive Aim 2 targets | Ordinary test suite does not fit Bayesian models |
| **Suggestion:** define units, metrics, and model objective | Added `docs/ANALYSIS_METHODS.md` explaining physical signal units (with uncertainty clearly flagged), dimensionless z-scores, likelihood, metrics and bootstrap | Documentation only |
| **Suggestion:** weighted standardized distances / Mahalanobis | Not adopted: changing metric would change the frozen scientific question and requires leak-free selection and reanalysis | Future sensitivity analysis |
| **Suggestion:** violin or box plots | Created and approved a two-panel Aim 1 box plot from locally available derived mcPHASES signatures | Versioned PNG/PDF/SVG, exploratory adjusted p-values and caption in `figures/aim1/` |
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

## Post-review local verification

The final figure is recorded under [`../figures/aim1/`](../figures/aim1/)
with the approved caption, aggregate p-values, and reproducibility command.
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

Exact computed-temperature units remain unverified. See
[`STATISTICAL_AUDIT.md`](STATISTICAL_AUDIT.md) for assumptions and limits.
