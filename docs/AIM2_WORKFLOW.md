# Opt-in Snakemake reproduction: Aim 2

This workflow adds explicit, inspectable dependencies to the **frozen**
standalone release scripts. It does not alter the model formulae, priors,
fold definitions, random seeds, bootstrap rules, or simulation parameters.

## Setup and safe defaults

From the repository root, in `beyond-average-cycle`:

```bash
python -m pip install -e '.[dev,workflow]'
snakemake --snakefile workflow/Snakefile --cores 1 \
  --config data_dir=data/mcphases --list-target-rules
```

**Running Snakemake without an explicit target invokes only preprocessing
and Aim 1**, not Bayesian fitting. To inspect the complete Aim 2 dependency
graph **without executing it**, run:

```bash
snakemake --snakefile workflow/Snakefile --cores 1 \
  --config data_dir=data/mcphases --dry-run aim2
```

The dry run still requires filenames for the five restricted source CSVs to
be present at the path supplied in `data_dir`; Snakemake does not download
the raw data. See `data/README.md` for authorized data access. Input and
output directories should remain untracked/ignored. Never upload restricted
participant-level results to GitHub or attach them to a public issue.

## Explicit targets

| Target | Script branches reached | Outputs |
| --- | --- | --- |
| `all` (default) | Preprocessing -> Aim 1 | Aim 1 CSVs only |
| `aim2_primary` | Within-interval manifest -> folds -> Bayesian population/hybrid models -> evaluation -> Figure 1 | Near-term metrics and Figure 1 PNG/PDF |
| `aim2_longitudinal` | 2022-2024 manifest -> folds -> Bayesian models -> evaluation; profile-transfer, repeatability, failure, oracle recentering and residual-drift diagnostics -> Figure 2 | Longitudinal metrics and Figure 2 PNG/PDF |
| `aim2_simulation` | Variance components and repeatability -> depth, sensitivity, variance-grid and nominal-precision simulations -> Figure 3 | Simulation tables and Figure 3 PNG/PDF |
| `aim2` | All three branches | Figures 1-3 and primary/longitudinal metrics |
| `aim1_assumption_audit` | Existing Aim 1 signatures -> frozen global vs interval-stratified sensitivity nulls | `results/aim1/audit/assumptions.json` |

Example (computationally expensive; **run only when intentionally refitting**):

```bash
snakemake --snakefile workflow/Snakefile --cores 1 \
  --config data_dir=data/mcphases aim2_primary
```

The model rules preserve the existing release-script defaults of 1,000
posterior draws, 1,000 tuning steps, four chains, and seed 20261014. They
use the project's established `PYTENSOR_FLAGS='cxx='` workaround. On other
platforms, the PyTensor setting may need adaptation. `--cores 1` does not
change the runner's configured 4 model-fitting cores; plan resource use
accordingly.

## Completion and reproducibility caveats

* The fold generator writes many per-person files into a Snakemake
  `directory()` output. Regenerating that output **replaces** the old fold
  directory. Back up any locally valuable derived files before forcing it.
* The existing hierarchical runners aggregate completed per-fold files to
  `all_fold_results.csv` and `all_predictions.csv`. Each runner has a manual
  `--resume` flag but considers `fold_result.csv` alone sufficient to skip a
  fold. Snakemake deliberately **does not** add `--resume` automatically:
  an incomplete fold could otherwise be incorrectly skipped.
* Snakemake tracks the named generated files, not the full contents of every
  posterior NetCDF. The oracle diagnostic assumes posterior files were
  generated successfully in each longitudinal fold. Inspect files before
  interpreting an oracle rerun.
* If the user already has generated results, a dry-run may show fewer jobs.
  This is expected: it plans only missing/out-of-date outputs.
* If saved model outputs already exist, verify their recorded fold-level
  log loss against named phase-probability columns with
  `python scripts/aim2/audit_fold_log_loss.py --analysis primary` and the
  equivalent `--analysis longitudinal`. The audit is **not** wired as a
  Snakemake dependent target to avoid unexpectedly triggering model fits.
* The fold-level class-probability log loss has a new explicit label-
  alignment safeguard. Any previously generated fold-level log-loss values
  must be audited before claiming numerical preservation.
* Bayesian stochastic results should be compared using saved prediction
  tables, diagnostics, and numerical tolerance, not automatic assertions of
  byte-for-byte identity. Exact equivalence is **not** established by the
  lightweight test suite.
* The oracle uses observed 2024 baselines retrospectively. It is a
  diagnostic, **not** leakage-free prospective prediction.

## Validation sequence

```bash
pytest -q
python -m pycodestyle src tests scripts/aim2/*.py
python -m py_compile scripts/aim2/*.py
snakemake --snakefile workflow/Snakefile --cores 1 \
  --config data_dir=data/mcphases --dry-run aim2
```

For a faster check, dry-run just `aim2_primary`, `aim2_longitudinal`, and
`aim2_simulation` separately. Only after reviewing the planned jobs and
availability of authorized input data should you execute an Aim 2 target.
