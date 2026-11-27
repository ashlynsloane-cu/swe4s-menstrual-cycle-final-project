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

### Aim 1 distance-metric sensitivity analysis

In response to the Moore–Fairborn code review, regularized
Mahalanobis distance was implemented as an exploratory sensitivity
analysis while retaining standardized Euclidean distance as the
primary metric.

Standardized Euclidean distance accounts for differences in feature
variance but does not account for covariance between features.

For two cycle signatures x and y:

    d(x, y) = sqrt(sum((x_j - y_j)^2 / s_j^2))

where s_j^2 is the cohort-level sample variance of feature j.

For regularized Mahalanobis distance, features are standardized
using cohort-level means and population standard deviations.
Ledoit-Wolf shrinkage is then used to estimate a regularized
covariance matrix.

The distance between standardized signatures is:

    d(z_x, z_y) = sqrt((z_x - z_y)^T P (z_x - z_y))

where P is the inverse of the regularized covariance matrix.

Both metrics use the same complete-case repeated-cycle cohort.
Similarity advantage is defined as:

    mean(between-person distance) - mean(within-person distance)

Positive values indicate greater within-person similarity.
Raw similarity advantages from different distance metrics are
not directly comparable in magnitude.

The implementation is provided in:

- `src/mcphases/mahalanobis.py`
- `src/mcphases/run_aim1_mahalanobis.py`

The sensitivity analysis is executed separately, preserving
the original primary analysis outputs.

### Aim 1 interval-specific statistical inference

The within-interval permutation test shuffles cycle-level
participant labels separately within each study interval.

The cross-interval permutation test reassigns complete
later-interval participant profiles as blocks among participants
observed in both study intervals.

Both distance metrics use the same permutation procedures,
with 9,999 permutations and random seed 42.

Holm correction is applied across the two interval-specific
comparisons separately for each distance metric.

| Metric | Contrast | Similarity advantage | Holm-adjusted p |
| --- | --- | ---: | ---: |
| Standardized Euclidean | Within interval | +0.111913 | 0.2282 |
| Standardized Euclidean | Across intervals | +0.268941 | 0.2188 |
| Regularized Mahalanobis | Within interval | +0.053512 | 0.3163 |
| Regularized Mahalanobis | Across intervals | +0.217161 | 0.2374 |

Neither interval-specific comparison reached statistical
significance after Holm correction under either distance metric.

Pairwise distances are not independent observations.
Permutation inference depends on the exchangeability assumptions
associated with each null hypothesis.

The within-interval and cross-interval comparisons involve
different participant cohorts and do not provide a direct
test of temporal physiological drift.

See [AIM1_PANEL_INFERENCE.md](AIM1_PANEL_INFERENCE.md)
for additional methodological details.

### Aim 1 signal-specific comparisons

Signal-specific analyses evaluate resting heart rate, HRV RMSSD,
nightly temperature, and respiratory rate separately, alongside
the combined 16-feature physiological signature.

The pooled comparisons use 10,000 permutations with cycle-level
participant-label shuffling and random seed 42.
Permutation p-values are exploratory and unadjusted.

The direction and ordering of individual signal-specific
similarity advantages were consistent across both distance
metrics. None of the comparisons reached an unadjusted
0.05 significance threshold.

### Aim 1 physiological feature correlations

Pearson correlations among the 16 centered physiological
signature features are visualized using a correlation heatmap.

The correlation structure motivates covariance-aware distance
estimation. However, cycle centering can itself induce
correlations among phase-specific features.

The heatmap is descriptive and does not establish a causal
explanation for differences between distance metrics.

### Aim 1 software validation and reproducibility

Additional unit tests evaluate regularized covariance estimation,
Mahalanobis distance calculations, cohort consistency, permutation
inference, and figure generation.

The expanded local test suite passed all 111 tests, along with
repository-wide PEP 8 and Python compilation checks.

The optional `aim1_figures` Snakemake target integrates the
sensitivity analysis and all four Aim 1 figure sets.

On November 26, 2026, all six scheduled Snakemake jobs completed
successfully using Snakemake 9.22.0 and Python 3.12.16,
regenerating the figures and reproducing the interval-specific
statistical results from existing local preprocessed data and
cycle signatures.

No Bayesian Aim 2 models were refitted.

Final figures, aggregate statistical results, reproduction
instructions, and provenance manifests are available in
[`figures/aim1/`](../figures/aim1/README.md).

Consistency across distance metrics does not establish statistical
significance, equivalence, exchangeability, or empirical power.

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
