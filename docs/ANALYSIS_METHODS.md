# Analysis methods, units, and validation

This page makes assumptions and evaluation metrics explicit without changing
any frozen scientific result. For code and dataset provenance, see
[`src/mcphases/preprocess.py`](../src/mcphases/preprocess.py) and the
[mcPHASES 1.0.0 data record](https://physionet.org/content/mcphases/1.0.0/).

## Physiological features and units

| Analysis column | mcPHASES input column | Meaning / units |
| --- | --- | --- |
| `resting_heart_rate` | `resting_heart_rate.csv:value` | Resting heart rate; normally beats/minute (bpm) |
| `hrv_rmssd` | `heart_rate_variability_details.csv:rmssd` | Root mean square of successive beat-interval differences; normally milliseconds (ms) |
| `nightly_temperature` | `computed_temperature.csv:nightly_temperature` | Nightly wearable temperature quantity; **verify the dataset's exact unit and whether this is absolute temperature or a deviation** before labeling it in physical units |
| `respiratory_rate` | `respiratory_rate_summary.csv:full_sleep_breathing_rate` | Full-sleep breathing rate; normally breaths/minute |

The preprocessing function takes the median available signal value per
participant and day. It does **not** perform unit conversions or fill in
missing values. The exact provider-specific units must be checked against the
restricted source data dictionary before reporting raw values as physical
measurements. Aim 1 phase signatures are **phase mean minus cycle mean**,
so their 16 feature coordinates retain the units of their parent signal.
`cycle_number` and `study_interval` are identifiers, not continuous features.

Aim 2 fits z-scaled predictors **within each training fold**. These scaled
predictors are dimensionless and should not be described with physical units.

## Aim 1 hypothesis test

`src/mcphases/similarity.py` computes standardized Euclidean distance over
the 16 cycle-centered phase/signature coordinates, using the cohort-level
sample variance of each feature. The similarity advantage is

    mean(between-person distance) - mean(within-person distance)

so **positive** values indicate greater within-person reproducibility.
The one-sided permutation p-value is

    (1 + number of permuted advantages >= observed advantage) / (B + 1)

with B=10,000 by default and random seed 42. Permutations operate on
cycle-level participant labels, holding observed distances fixed. The
frozen primary analysis permutes across all complete signatures; an opt-in
`stratify_by_study_interval=True` sensitivity test permutes separately within
study intervals and holds interval-specific participant-label counts fixed.
This sensitivity null is **not** interchangeable with the original null,
especially when the same participant appears in both intervals.

Tests now check fixed-seed null false-positive behavior, detection of a
strong planted person effect, reproducibility, input validation, and
interval-specific label multiplicities. These synthetic checks do not prove
exchangeability for the real data, do not estimate the study's actual power,
and do not justify substituting the alternative null without reanalysis.

## Aim 2 fitting, evaluation, and uncertainty

The models are hierarchical **categorical Bayesian regressions** implemented
with [Bambi](https://bambinos.github.io/bambi/),
[PyMC](https://www.pymc.io/), and
[ArviZ](https://python.arviz.org/). They use a four-class categorical
likelihood and regularizing Normal / HalfNormal priors; they do not use a
neural-network training loss. Their posteriors are sampled with MCMC, not
optimized through a standalone gradient-based loss function.

The held-out evaluation reports:

- **Accuracy:** fraction of correct top-probability phase predictions.
- **Balanced accuracy:** unweighted mean class recall.
- **Macro-F1:** unweighted mean of class F1 scores.
- **Log loss:** negative average log probability assigned to the true phase;
  lower is better.
- **Multiclass Brier score:** average sum of squared differences between
  predicted class probabilities and one-hot observed phases; lower is better.
- **Mean true-phase probability:** average predicted probability assigned to
  the observed class; higher is better.

Aim 2 uncertainty intervals use a participant-cluster bootstrap: 5,000 draws
with replacement of target participant IDs, with **all evaluation rows for
all sampled participants** included each time. As in any ordinary bootstrap,
a given resample is *not* guaranteed to include a fixed fraction of the
unique participants; duplicates are expected. The results are percentile
intervals (2.5% and 97.5%) of the paired hybrid-minus-population metric.
Changing that scheme to demand a fixed unique-participant fraction would be
a distinct resampling estimator requiring methodological justification.

## Scope of refactor

Only posterior probability extraction and completed-fold aggregation were
moved into `scripts/aim2/hierarchical_shared.py`. Primary folds use
`id*_*` directories (target ID and interval), while longitudinal folds use
`id*` directories. Priors, models, split logic, fold contents, and the default
Aim 1 permutation procedure were not changed. Prior published or submitted
numeric results therefore must still be reproduced before claiming
bit-for-bit equivalence after code changes.

## Follow-up verification and caveats

The full opt-in Snakemake DAG and additional model-utility regression tests
are documented in [`AIM2_WORKFLOW.md`](AIM2_WORKFLOW.md) and
[`STATISTICAL_AUDIT.md`](STATISTICAL_AUDIT.md).

The source descriptor distinguishes **computed temperature** from
**wrist-temperature deviations**; these are not interchangeable without
checking the actual `computed_temperature.csv:nightly_temperature`
dictionary entry. We have **not** verified its precise units. The original
Aim 1 cycle-label permutation and the interval-stratified sensitivity test
invoke different exchangeability assumptions; neither is automatically
established by the synthetic tests. See the audit document for data-dependent
checks and other limitations, including potential changes to wearable
firmware/software between study intervals.

The new `audit_fold_log_loss.py` script checks prior saved fold metrics
without rerunning posterior fits. Invoke separately for `--analysis primary`
and `--analysis longitudinal` if the local ignored predictions and fold
results are available.

The final hierarchical model runners now score log loss in the class order
returned by the Bambi posterior coordinate, rather than assuming that
scikit-learn uses the same order. **This is a correctness fix** and may
change previously generated fold-level metrics if category order differed.
Those outputs need a real-data audit before claiming preservation.

The primary Aim 2 evaluator now verifies *bidirectional* population/hybrid
match completeness (all population rows match hybrid rows and vice versa).
This is an input-integrity check; it does not change correctly paired metrics.
