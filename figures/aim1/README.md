# Aim 1: Cycle-signature distance comparisons

This directory contains the two-panel Aim 1 distance figure,
distance-metric sensitivity figures, and aggregate statistical
comparisons. It does not contain individual participant records
or cycle-level signatures.

## Figure files

### Primary standardized Euclidean analysis

- `aim1_distance_boxes.png`: raster figure
- `aim1_distance_boxes.pdf`: vector PDF
- `aim1_distance_boxes.svg`: vector SVG
- `aim1_distance_boxes_caption.md`: caption and interpretation notes
- `aim1_panel_comparisons.csv`: aggregate permutation comparisons

### Figure provenance

- `figure_manifest.json`: original figure checksums and analysis parameters
- `sensitivity_manifest.json`: checksums and analysis parameters for the
  additional distance-metric sensitivity figures.

The original `figure_manifest.json` is retained as a historical provenance
record. Its README and plotting-script checksums refer to the original
versions, before the distance-metric sensitivity analysis was added.

### Regularized Mahalanobis sensitivity analysis

- `mahalanobis/aim1_distance_boxes.png`: raster figure
- `mahalanobis/aim1_distance_boxes.pdf`: vector PDF
- `mahalanobis/aim1_distance_boxes.svg`: vector SVG
- `mahalanobis/aim1_distance_boxes_caption.md`: caption and interpretation notes
- `mahalanobis/aim1_panel_comparisons.csv`: aggregate permutation comparisons

### Signal-specific distance metric comparison

- `aim1_metric_comparison.png`
- `aim1_metric_comparison.pdf`
- `aim1_metric_comparison.svg`
- `aim1_metric_comparison.csv`: similarity advantages and permutation p-values

### Physiological feature correlation heatmap

- `aim1_feature_correlations.png`
- `aim1_feature_correlations.pdf`
- `aim1_feature_correlations.svg`
- `aim1_feature_correlations.csv`: Pearson correlation matrix

## Reproduction

Using authorized local mcPHASES data, the complete Aim 1 analysis
and figure workflow can be reproduced using Snakemake:

```bash
python -m pip install -e ".[dev,workflow]"

snakemake -s workflow/Snakefile --cores 1 aim1_figures --dry-run

snakemake -s workflow/Snakefile --cores 1 aim1_figures
```

Run these commands from the repository root.

Generated figures and statistical tables are written to
`results/figures/aim1/`.

### Standardized Euclidean distance boxplots

```bash
python scripts/aim1/plot_aim1_distance_boxes.py \
  --signatures results/aim1/cycle_signatures.csv \
  --distance-metric standardized_euclidean \
  --output-dir results/figures/aim1/euclidean \
  --permutations 9999 --seed 42 --annotate-p
```

### Regularized Mahalanobis distance boxplots

```bash
python scripts/aim1/plot_aim1_distance_boxes.py \
  --signatures results/aim1/cycle_signatures.csv \
  --distance-metric regularized_mahalanobis \
  --output-dir results/figures/aim1/mahalanobis \
  --permutations 9999 --seed 42 --annotate-p
```

Both distance metrics use the same complete-case repeated-cycle cohort.

### Signal-specific distance metric comparison

First generate the primary and sensitivity analysis outputs:

```bash
python -m mcphases.run_aim1

python -m mcphases.run_aim1_mahalanobis \
  --output-dir results/aim1_mahalanobis \
  --permutations 10000
```

Then generate the comparison figure:

```bash
python scripts/aim1/plot_aim1_metric_comparison.py \
  --euclidean results/aim1/signal_specific_reproducibility.csv \
  --mahalanobis results/aim1_mahalanobis/signal_specific_reproducibility.csv \
  --output-dir results/figures/aim1
```

### Physiological feature correlation heatmap

```bash
python scripts/aim1/plot_aim1_feature_correlations.py \
  --signatures results/aim1/cycle_signatures.csv \
  --output-dir results/figures/aim1
```

The plotting and inference implementations are in:

- `scripts/aim1/plot_aim1_distance_boxes.py`
- `scripts/aim1/plot_aim1_metric_comparison.py`
- `scripts/aim1/plot_aim1_feature_correlations.py`
- `src/mcphases/aim1_panel_inference.py`
- `src/mcphases/similarity.py`
- `src/mcphases/mahalanobis.py`
- `src/mcphases/run_aim1_mahalanobis.py`

## Results

### Primary standardized Euclidean analysis

| Contrast | Same-person pairs | Different-person pairs | Participants | Mean-distance advantage | One-sided p | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Within interval | 53 | 1,468 | 25 | +0.111913 | 0.2282 | 0.2282 |
| Across intervals | 60 | 715 | 14 | +0.268941 | 0.1094 | 0.2188 |

The distance advantage is the difference between different-person and
same-person mean standardized Euclidean distances; positive values indicate
closer same-person signatures. Both tests used 9,999 permutations and seed 42.
Neither comparison was statistically significant after Holm correction.

The displayed pair counts are not independent sample sizes. The within-interval
test permutes cycle-level participant labels within study intervals; the
cross-interval test reassigns complete later-interval profiles among individuals
observed in both intervals. Their validity depends on the associated
exchangeability assumptions. The participant cohorts differ, so effect-size
magnitudes should not be interpreted as a direct test of temporal drift.

See `aim1_distance_boxes_caption.md` and
`docs/AIM1_PANEL_INFERENCE.md` for the full methodological description.

### Regularized Mahalanobis sensitivity analysis

Regularized Mahalanobis distance was implemented in response to the
Moore–Fairborn code review to assess sensitivity to feature covariance.

Ledoit-Wolf shrinkage is used to stabilize covariance estimation
for the cohort-standardized physiological features.

The same interval-specific permutation procedures and Holm correction
are applied to both distance metrics.

Positive within-person similarity advantages were observed within
and across study intervals using regularized Mahalanobis distance.
Neither comparison reached statistical significance after Holm
correction.

See the [Mahalanobis caption](mahalanobis/aim1_distance_boxes_caption.md)
and [aggregate comparison table](mahalanobis/aim1_panel_comparisons.csv)
for the corresponding numerical results.

### Signal-specific reproducibility

| Physiological signal | Standardized Euclidean advantage | Unadjusted p | Regularized Mahalanobis advantage | Unadjusted p |
| --- | ---: | ---: | ---: | ---: |
| Resting heart rate | +0.1694 | 0.0660 | +0.1551 | 0.0534 |
| Nightly temperature | +0.1218 | 0.1493 | +0.1155 | 0.1498 |
| HRV RMSSD | +0.1044 | 0.2555 | +0.0764 | 0.3016 |
| Respiratory rate | -0.0613 | 0.6638 | -0.0546 | 0.6566 |
| All signals | +0.1197 | 0.2318 | +0.0544 | 0.3417 |

The pooled signal-specific comparisons use 10,000 permutations
and report exploratory, unadjusted p-values.

The direction and ordering of the observed individual signal-specific
similarity advantages were consistent across both distance metrics.
Resting heart rate exhibited the largest observed similarity advantage
among the four individual physiological signals.

None of the signal-specific comparisons reached an unadjusted
0.05 significance threshold.

### Physiological feature correlations

Pearson correlations among the 16 centered physiological signature
features provide a descriptive visualization of feature relationships.

The correlation structure motivates the use of a covariance-aware
distance metric. However, cycle centering can itself induce
correlations among phase-specific features.

The heatmap does not establish a causal explanation for differences
between distance metrics.

## Interpretation and limitations

- The primary standardized Euclidean analysis is preserved.
- Regularized Mahalanobis distance is an exploratory sensitivity analysis.
- Raw similarity advantages from different metrics are not directly
  comparable in magnitude.
- Signal-specific permutation p-values are unadjusted.
- Interval-specific p-values are Holm-adjusted across two comparisons
  separately for each distance metric.
- Consistency across distance metrics does not establish statistical
  significance or equivalence.

See [analysis methods](../../docs/ANALYSIS_METHODS.md) and
[code-review response](../../docs/CODE_REVIEW_RESPONSE.md)
for additional methodological details.

Source files and participant-level outputs remain excluded from Git.
