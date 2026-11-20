# Approved Aim 1 distance comparison figure

**Status:** Figure styling approved on 2026-11-20. This is an exploratory figure, not a claim of statistical significance. Preserve the underlying analysis and caption when reusing it.

This folder contains the exact user-approved **reference image** (`aim1_distance_boxes_approved_reference.png` after installation), plus an aggregate two-row comparison table. It holds no participant-level rows or participant identifiers. The PNG is a visual reference uploaded by the user; the original 300-dpi PNG and vector PDF/SVG are copied from the local reproducible output when this package is installed.

## Reproduction

From the repository root in the `beyond-average-cycle` environment:

```bash
python scripts/review/plot_aim1_distance_boxes.py \
  --signatures results/aim1/cycle_signatures.csv \
  --output-dir results/figures/aim1 \
  --permutations 9999 --seed 42 --annotate-p
```

Save the original PDF/SVG and PNG alongside the approved reference image when preserving this version. Required source code: `scripts/review/plot_aim1_distance_boxes.py`, `src/mcphases/aim1_panel_inference.py`, and `src/mcphases/similarity.py`. Source data and per-cycle signatures remain restricted and are not included here.

## Approved comparisons

- **A — Within interval:** mean-distance advantage +0.111913; 53 same-person and 1,468 different-person pairs; 25 participants; one-sided permutation p = 0.2282; Holm-adjusted p = **0.2282**.
- **B — Across intervals:** mean-distance advantage +0.268941; 60 same-person and 715 different-person pairs; 14 matched participants; one-sided permutation p = 0.1094; Holm-adjusted p = **0.2188**.
- Both used **9,999 permutations**, random seed **42**. Effect is mean different-person minus mean same-person distance. Neither adjusted p-value is below 0.05.

**Interpretation constraints:** Pair counts are not independent sample sizes. The within-interval test permutes cycle identity labels within intervals, and the cross-interval test reassigns whole later-interval participant profile blocks among participants observed in both intervals. These exploratory procedures depend on exchangeability assumptions; multiplicity correction cannot fix violations. Because panels use different participant cohorts, their effect magnitudes should not be directly compared as evidence of drift.

## Provenance

- Approved reference image SHA-256: `bcd8a32c5535133b0741a3ac4b9ef7cd1d86cff7916ded1b361bfd6edf1445cb`
- Expected finalized plotting-script SHA-256: `8df3977fee5f562dec90280e3966a536955ad28a2caf09cc2f2b1833e6d6713d`
- Comparison table: `aim1_panel_comparisons.csv`, as submitted with the user-run analysis.

No new statistical calculations were performed when freezing this figure. Before pushing derived figures to a **public** GitHub repository, check the mcPHASES data-use terms.
