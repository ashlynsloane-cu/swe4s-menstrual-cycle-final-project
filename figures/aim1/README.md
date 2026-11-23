# Aim 1: Cycle-signature distance comparisons

This directory contains the two-panel Aim 1 distance figure and aggregate
statistical comparisons. It does not contain individual participant records
or cycle-level signatures.

## Figure files

- `aim1_distance_boxes.png`: raster figure
- `aim1_distance_boxes.pdf`: vector PDF
- `aim1_distance_boxes.svg`: vector SVG
- `aim1_distance_boxes_caption.md`: caption and interpretation notes
- `aim1_panel_comparisons.csv`: aggregate permutation comparisons
- `figure_manifest.json`: checksums and analysis parameters

## Reproduction

Using locally generated cycle signatures from authorized mcPHASES data:

```bash
python scripts/review/plot_aim1_distance_boxes.py \
  --signatures results/aim1/cycle_signatures.csv \
  --output-dir results/figures/aim1 \
  --permutations 9999 --seed 42 --annotate-p
```

The plotting and inference implementations are in
`scripts/review/plot_aim1_distance_boxes.py`,
`src/mcphases/aim1_panel_inference.py`, and
`src/mcphases/similarity.py`.

## Results

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
Source files and participant-level outputs remain excluded from Git.
