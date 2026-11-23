# Scientific-methods audit after the Moore–Fairborn review

**Scope:** code, documentation, unit tests, and separately reported
local audits of saved derived mcPHASES outputs. These notes distinguish
verification of **implemented methods** and saved-score consistency from
validation of **scientific assumptions**. Bayesian fits were not rerun
as part of the review checks.

## Source mapping and physical units

The preprocessing source uses the following unconverted input columns:

| Output feature | Data source | Typical interpretation | Verification status |
| --- | --- | --- | --- |
| `resting_heart_rate` | `resting_heart_rate.csv:value` | beats/min | Typical Fitbit unit; exact export confirmation pending |
| `hrv_rmssd` | `heart_rate_variability_details.csv:rmssd` | ms | Typical RMSSD unit; exact export confirmation pending |
| `nightly_temperature` | `computed_temperature.csv:nightly_temperature` | nightly computed sleep-temperature metric | **Unit and absolute-vs-relative convention not verified** |
| `respiratory_rate` | `respiratory_rate_summary.csv:full_sleep_breathing_rate` | breaths/min | Typical rate unit; exact export confirmation pending |

mcPHASES describes both **Computed Temperature** and **Wrist Temperature**
as separate tables. The wearable vendor's *wrist temperature* can be
reported as a deviation from a personal baseline; that fact **does not**
prove that `computed_temperature.csv:nightly_temperature` uses the same
quantity or unit. Do not label this feature as a body temperature in °C,
° F, or a baseline deviation without checking the restricted data dictionary.

Sources:
- Dataset: https://physionet.org/content/mcphases/1.0.0/
- Dataset descriptor: https://www.nature.com/articles/s41597-026-06805-3
- Vendor skin-temperature context (a *different* measurement description):
  https://developers.google.com/health/data-types/skin-temperature

The `scripts/aim2/audit_signal_values.py` helper writes *aggregate* value
ranges without participant IDs. **Range checks cannot establish units.**
No raw data are copied into this repository.

## Aim 1: assumptions and sensitivity

1. The 16 signature columns are phase-specific means centered to each
   cycle-wide mean. Their features **retain** original signal units, then
   standardized Euclidean distance rescales by cohort variances. This is a
   descriptive similarity statistic, not a phase-prediction loss function.
2. The participant-label permutation null is **cycle-level**, holding
   distances fixed. Its one-sided statistic is
   `mean(between-person distance) - mean(within-person distance)`.
3. The frozen original permutes across **all** complete cycle signatures.
   Within-interval permutation is a **different null** that preserves each
   interval's participant-label multiplicities.
4. Equal-cycle and equal-interval sampling are not assured. Within-person
   and between-person pairs are *dependent* because each cycle participates
   in multiple pairs. Do not treat the number of pairwise distances as
   independent observations or base a classical t-test on them.
5. Validity requires exchangeability under the specified null; unequal
   study intervals, participant differences in cycle count, repeated
   participation across 2022/2024, drift, and unequal phase coverage could
   complicate it. Simulation unit tests demonstrate behavior under their
   specific data-generating processes; they do not prove these assumptions
   on mcPHASES.
6. Pair count, intervals, cross-interval same-person comparisons, and
   global versus stratified p-values can be inspected by running:

   ```bash
   snakemake --snakefile workflow/Snakefile --cores 1 \
     --config data_dir=data/mcphases aim1_assumption_audit
   ```

   Output: `results/aim1/audit/assumptions.json` (ignored by Git).
   Compare the two p-values as sensitivity estimates; do not choose the
   smaller or change the primary null post hoc.

## Aim 2: no-leakage and uncertainty checks

1. For both final hierarchical model scripts, `standardize_fold` fits
   **both scalers on population-training rows only**, then transforms
   calibration and held-out test rows. New tests compare output to an
   independently defined training distribution and check no input mutation.
2. `add_model_group` uses `source_id` **and** `study_interval`, preventing
   accidental sharing of a fitted participant-level group identity across
   years.
3. The Bambi formula is a Bayesian categorical regression with regularizing
   priors; there is **no neural-network loss function**. Predictions are
   posterior-mean category probabilities. The tests validate scoring and
   table assembly without fitting posterior samples.
4. **Additional correctness finding:** both hierarchical runner `evaluate`
   functions previously called `sklearn.metrics.log_loss` with a probability
   matrix and an explicit category list. Scikit-learn assumes the probability
   columns follow sorted class order. This is not guaranteed by an arbitrary
   Bambi category coordinate. The extension replaces the fold-level call
   with a checked `aligned_log_loss` that maps **each observed phase directly
   to its actual probability column**. This is regression-tested using
   unsorted labels. If prior predictions were in sorted order, results are
   expected to agree (subject to numerical precision); if not, fold-level
   log-loss results **could change** and prior figures/interpretations need
   verification against saved class coordinates. **No empirical equivalence
   has been established.**
5. The paired predictions must match in **both directions**: every
   population row has exactly one hybrid row and vice versa. The previous
   evaluator only checked population completeness; an extra hybrid row could
   be discarded silently. A symmetric check is introduced and regression-
   tested; this guard should not affect previously correctly paired files.
6. Aim 2 participant-cluster bootstrap draws participant IDs with
   replacement and includes all corresponding held-out rows. Duplicates and
   omissions are intrinsic to the bootstrap. Unequal numbers of days per
   participant mean pooled-day metrics give participants with more days
   more influence; this is an **estimand choice**. An equal-participant
   estimand would need its own prespecified analysis.
7. Correlation and temporal drift complicate interpretation of 2022-to-2024
   transfer. The dataset descriptor explicitly cautions that manufacturer
   firmware/software changes may affect measurement precision across
   intervals. This is a limitation, **not a demonstrated explanation** for
   model performance.
8. The oracle recentering test uses held-out 2024 observations. It remains
   retrospective and must never be presented as prospective model accuracy.

## Local empirical checks after the original review

The following checks were executed separately on authorized, untracked
local derived results. They do not require publishing participant-level
records and do not constitute a Bayesian refit:

| Audit | Recorded result | Scope |
| --- | --- | --- |
| Saved fold-level log loss | 38/38 primary and 18/18 longitudinal fold scores matched recomputation from named class-probability columns; 0 mismatches | Saved predictions and fold-score tables; does not validate refitting |
| Population/target-ID exclusion | 19/19 primary and 9/9 longitudinal fold checks passed | Checks tested split boundaries; not every potential upstream leakage source |
| Cluster-bootstrap resampling | 5,000 replicates for each of the 16-participant primary and 9-participant longitudinal analyses | Concentration diagnostics, not validity thresholds |
| Aim 1 original/global vs interval-stratified sensitivity | One-sided p = 0.2318 vs 0.2077 with 9,999 permutations | Different exploratory null assumptions; neither is significant |
| Aim 1 approved panel A/B comparisons | Holm-adjusted p = 0.2282 / 0.2188 with 9,999 permutations, seed 42 | Pair distances dependent; panel A 25 participants, panel B 14 matched participants |
| Synthetic power smoke grid | Five simulations and 99 permutations per scenario | Too few replicates for reliable power or type-I error estimates |

The prior unsorted-class log-loss safeguard is retained even though
these specific stored score tables show no mismatches. The approved
figure, caption, and aggregate comparisons are tracked under
[`../figures/aim1/`](../figures/aim1/).

## Verification still pending

- Confirm all four source-column units, especially nightly computed
  temperature's unit and absolute/deviation convention, against
  authorized dataset metadata.
- Assess permutation-test type-I error under realistic participant-level
  dependence, unequal cycle counts, and interval heterogeneity.
- Check participant-level class composition and per-fold phase support.
- Validate posterior diagnostics and output equivalence after the
  shared-helper refactor with an explicitly controlled Bayesian rerun.
  The 56 saved-score matches do not establish this.
- Explore weighted standardized distances or regularized covariance only
  as a prespecified sensitivity analysis with proper leakage control.

To rerun the **no-refit** saved-score checks on authorized local files:

```bash
python scripts/aim2/audit_fold_log_loss.py --analysis primary
python scripts/aim2/audit_fold_log_loss.py --analysis longitudinal
```

Never publish restricted participant-level prediction files or raw
mcPHASES records as part of this audit.
