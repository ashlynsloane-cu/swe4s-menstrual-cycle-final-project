# Follow-up Moore–Fairborn review: methodological gates

This supplement documents the reviewer's additional verbal notes and a
reproducible audit plan. It does **not** alter the frozen primary scientific
analysis, reported p-values, Bayesian priors, or saved model outputs.

## Empirical Aim 1 audit (run 2026-11-19, user-provided log)

Using **9,999** participant-label permutations on the 79 complete-case
repeated-cycle signatures (25 participants, 16 features), the within-person
similarity advantage (between mean distance minus within mean distance) was
**0.1196925**. Global label-permutation **p = 0.2318**; interval-stratified
label-permutation **p = 0.2077**. Neither rejects its own null at α = .05.
There were 113 within-person pairs, 2,968 between-person pairs, 1,560
cross-interval pairs, and 60 cross-interval within-person pairs. The pair
counts are descriptive, **not independent sample sizes**. The full output
should remain locally in `results/aim1/audit/assumptions.json` and must not
be mistaken for a replacement of the frozen primary result.

The stratified null shuffles cycle-level participant labels independently
*within each study interval*. It preserves each interval's distribution of
participant-label counts but does **not** preserve longitudinal pairing
across intervals or participant-specific dependence across cycles. The
global scheme also assumes cycle-level exchangeability under its null.
Neither p-value establishes validity of those assumptions.

## Synthetic power and calibration

The additional `scripts/review/simulate_permutation_power.py` simulates
synthetic 16-dimensional, within-cycle phase-centered signals, with stable
person-by-phase effects, independent cycle noise, and optional 2024 drift.
It estimates rejection frequency at α=.05 using a fixed, explicit number
of Monte Carlo replicates and permutations and reports a Wilson binomial
interval. For effect size 0 with no drift, it is a **type-I error** scenario;
for nonzero effects, **power** is conditional on the generative model.
Do not claim this is empirical or retrospective power for the 25 analyzed
mcPHASES participants. Test 8, 25, and 50 participants (and two versus
four cycles) because the reviewers specifically requested larger datasets.

For efficient simulations, the implementation uses `scipy.spatial.distance`
and NumPy to reproduce the *same* standardized Euclidean statistic and ID
shuffle as Aim 1. The test code checks known positive and null cases; any
scientific publication should also compare the optimized p-values with the
original `mcphases.similarity.permutation_test` on the same small synthetic
dataset and random seed.

## Bootstrap participant diversity

`evaluate_primary_predictions.cluster_bootstrap()` and the longitudinal
`participant_bootstrap()` sample participant **clusters** with replacement,
which is standard for a nonparametric cluster bootstrap. Duplicate IDs are
expected and must not be prohibited without changing the estimator.
`audit_bootstrap_diversity.py` reproduces the ID-drawing mechanism, reporting
unique participants per draw, maximum multiplicity, and an effective cluster
count `n_eff = n^2 / sum(count_i^2)`. The finite-sample implications of
small participant counts should be discussed; distributional diagnostics
are not proof of valid confidence interval coverage.

The primary and longitudinal participant-level bootstrap confidence intervals
are not automatically valid for arbitrary cluster heterogeneity or very
small numbers of independent participants. If this matters materially, run a
predeclared simulation-based coverage check using the actual analysis design,
not an arbitrary minimum-unique-participants rejection rule.

## Train/calibration/test integrity

The required exclusion is **target ID absent from population training**.
In personalized phase prediction, the held-out target's earlier cycles are
*intentionally present* in personal calibration and the same person's later
cycles are held out for testing. Checking that the target is absent from
**all training**, including calibration, would make the scientific design
impossible. The new fold audit verifies target exclusion from population
training, exact participant-day nonoverlap, per-split uniqueness, and primary
within-interval or longitudinal across-interval chronology. It does not
prove independent labels or eliminate leakage from global preprocessing,
model/feature selection, or dataset-specific collection artifacts.

## Figures and alternative distance measures

Use **box plots**, as recommended by the reviewer, rather than kernel-smoothed
violin plots. `plot_aim1_distance_boxes.py` separates within-/between-person
and within-/cross-interval distances, and writes a summary table. Multiple
pairs reuse cycles: quartiles and whiskers are descriptive, not valid
independent-pair uncertainty intervals. Avoid counting pair comparisons as
independent participants.

The original standardized Euclidean distance weights each standardized
feature equally. A reliability-weighted analysis would require prespecified,
comparable and defensible channel-level measurement uncertainty (not an
assumption that heart-rate sensors are always more precise). For a 16-feature
regularized Mahalanobis metric, estimate/shrink covariance **within each
training fold** (or using an independent training cohort), fix regularization
in advance, and evaluate stability and overfitting, especially at n=79
complete signatures. Both alternatives are sensitivity studies, not changes
required for code quality; do not tune weights on the reported test set.

## Workflow, data access, security, and remaining work

The existing `workflow/Snakefile` is the reproducible runner for the entire
Aim 2 DAG. New `check_workflow.sh` is a **dry-run-only** safe convenience
wrapper; `check_mcphases_inputs.py` verifies that five authorized local
mcPHASES files have readable CSV headers. It does not make restricted data
openly downloadable, request an unverified "API key", print credentials,
commit data, or transmit raw observations.

PhysioNet identifies mcPHASES 1.0.0 as restricted-access data. Obtain access
through its official requirements: https://physionet.org/content/mcphases/1.0.0/
The current license forbids sharing access to the restricted dataset.

Outstanding: verify the exact unit/processing of `nightly_temperature` from
source metadata; inspect unusual temperature minima; run empirical fold
leakage and participant-diversity audits against the original saved results;
review same- versus cross-interval descriptive box plots; test model output
parity if feasible. Synthetic power results and coverage checks should be
reported as explicitly exploratory, not after-the-fact justification for a
specific p-value.
