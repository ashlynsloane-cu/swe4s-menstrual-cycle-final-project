# Aim 1: exploratory within- and cross-interval comparisons

## Scientific question

Do two cycles from the **same person** have a smaller standardized Euclidean
signature distance than cycles from **different people**, (A) within one
study interval or (B) across the 2022 and 2024 intervals?

The prespecified effect is `mean_distance_different_people -
mean_distance_same_person`. Positive values indicate closer same-person
signatures. The plotted boxes summarize *pairs*, which are not independent
because the same cycle appears in several pairs. No pair-level t test or
Mann–Whitney p-value is used.

## Two separately defined randomization nulls

**A — within-interval repeatability.** Independently permute participant
labels *across individual cycles within each study interval*, preserving
participant-label counts within each interval. Recompute the positive
similarity advantage among same-interval pairs on every permutation. This
requires the eligible cycle signatures to be exchangeable within study
intervals under the null of no person-specific structure. Participants may
contribute different numbers of eligible cycles; chronological/batch effects
or additional within-participant structure can violate this assumption.

**B — cross-interval identity stability.** Keep 2022 labels fixed and
randomly reassign the **entire 2024 participant-profile blocks** among
participants observed in **both 2022 and 2024**. Each participant's cycles
remain grouped within 2024. Both same-person and different-person cross-
interval box plots are restricted to these shared participants when the
comparison annotation is enabled. This
tests whether matching participant identities across intervals confers
additional similarity over arbitrary matching, conditional on the observed
profile blocks. Its interpretation requires exchangeability of later-year
participant profiles with respect to their earlier-year identities. Unequal
participation across years, selection, demographic covariates, or
measurement differences may violate that assumption.

Both tests use a prespecified **one-sided** alternative (same person more
similar). Monte Carlo p-values use `(1 + # permuted stats >= observed) /
(1 + number of permutations)`. The two primary p-values are adjusted using
Holm's method; numerically reported Monte Carlo adjusted p-values, rather than stars, are annotated.
The tests are **exploratory sensitivity analyses**, not replacements for
the original pooled primary permutation test. Holm adjustment does not
repair a failure of exchangeability.

## Influence and uncertainty

Each panel also reports a leave-one-participant-out range of the descriptive
similarity advantage and the fraction of those exclusions retaining a
positive advantage. **This is not a confidence interval** and does not
replace a participant-level uncertainty analysis.

For the full set of 79 complete-case signatures reported in the earlier
audit, the **descriptive** mean advantages (from the provided box-summary)
were 5.362199 - 5.250286 = **0.111913** within intervals and
5.490781 - 5.358022 = **0.132759** across intervals **for all
eligible participants**. These are not p-values and do not establish an
effect. Panel B's inferential *matched-participant subset* can have a
different between-person mean and therefore a different effect size.
The original four-group descriptive CSV remains unchanged; interpret
it separately from the matched-panel inferential output. Final p-values
must be computed from the original signature rows; aggregated quartiles
are insufficient.

## Reproducible command

From the repository root in the `beyond-average-cycle` environment:

```bash
python scripts/review/plot_aim1_distance_boxes.py \
  --signatures results/aim1/cycle_signatures.csv \
  --output-dir results/figures/aim1 \
  --permutations 9999 \
  --seed 42 \
  --annotate-p
```

**Produces** the existing `distance_box_summary.csv`, plus
`aim1_panel_comparisons.csv` and `aim1_panel_null_distributions.csv`.
Also regenerates the existing `aim1_distance_boxes.png`, `.pdf`, `.svg`,
and writes `aim1_distance_boxes_caption.md` with a reproducible figure
legend containing the comparison-specific participant counts, effect sizes,
randomization procedures, multiplicity correction, and limitations.
The graphic itself has a brief footer and lighter, stacked p-value/effect-size
annotations. The unannotated original plot remains available by omitting the
last three flags.

## Reproducibility and data-sharing safeguards

Do not commit source participant files, cycle-signature rows, or other
restricted mcPHASES data. The code and synthetic unit tests are suitable
for the repository. Assess dataset usage terms before deciding whether
aggregate derived summary tables and figures can be publicly released.
Report the number of participants and the exact null assumptions with
any numerical p-values; do not use the count of cycle *pairs* as an
independent sample size. Record all prespecified comparisons and disclose
any additional exploratory variants.
