# Beyond the Average Cycle

**Person-specific physiological signatures across the menstrual cycle using mcPHASES**

This repository contains the final project for **CSCI 6118 / Software Engineering for Scientists**. It uses wearable and hormone-derived data from the mcPHASES dataset to study whether menstrual-cycle physiology is reproducible within individuals and whether person-specific information improves prediction of future menstrual phase.

## Project status

The scientific analyses for this project are complete.

You **do not need to rerun the Bayesian models or simulations** to use, review, or grade this repository. The final analysis code, figures, tests, and reproducibility workflow are version-controlled here. Computationally expensive model outputs are intentionally not committed.

The repository contains:

- finalized preprocessing and cycle-quality-control code;
- Aim 1 analyses of within-person physiological reproducibility;
- Aim 2 population and personalized phase-prediction models;
- longitudinal 2022-to-2024 personalization analyses;
- simulation experiments examining how many calibration cycles are needed for personalization;
- final presentation-ready figures;
- tests and GitHub Actions configuration.

## Scientific questions

### Aim 1

**Are menstrual-cycle physiological signatures reproducible within the same person across cycles?**

The analysis focuses on four wearable-derived physiological signals:

- resting heart rate;
- HRV RMSSD;
- nightly skin temperature;
- respiratory rate.

Hormone measurements are used to define menstrual-cycle phase, but they are not used as predictors.

### Aim 2

**When is person-specific wearable information useful for predicting future menstrual phase?**

Aim 2 compares:

1. a **population-only model**, trained without data from the target participant; and
2. a **personalized hybrid model**, which adds information from the target participant's prior cycles.

The four prediction classes are Menstrual, Follicular, Fertility, and Luteal.

## Main Aim 2 findings

The empirical analyses did **not** show that two-cycle personalization improved overall prediction of future menstrual phase.

In the near-term within-interval analysis, personalization did not improve overall discrimination and produced worse probabilistic calibration.

In the longitudinal analysis, person-specific information learned from 2022 did not transfer well to the same participants in 2024. Baseline physiological drift and changes in phase-specific patterns contributed to this failure.

Simulation experiments provide an important qualification: under a stationary data-generating process resembling the observed mcPHASES variance structure, approximately **two calibration cycles** were sufficient for a reliable personalization benefit in the nominal scenario. However, the required number of cycles changed substantially when the strength of stable person-specific signal or cycle-to-cycle variability changed.

Therefore, **two cycles is not a universal rule**. More calibration data can reduce transient cycle noise, but it cannot solve physiological drift when the personalized signature itself changes over time.

## Repository structure

```text
.
├── .github/workflows/test.yml
├── assets/aim2/
├── data/README.md
├── docs/figures/aim2/
├── scripts/aim2/
│   └── development/
├── src/
├── tests/
├── environment.yml
├── pyproject.toml
└── README.md
```

## Final Aim 2 figures

Version-controlled copies of the final figures are stored in:

```text
docs/figures/aim2/
```

- **Figure 1 — Near-term personalization:** population-only versus two-cycle personalized prediction for future cycles within the same study interval.
- **Figure 2 — Longitudinal personalization:** why person-specific information learned in 2022 does not transfer reliably to 2024, including baseline drift and phase-profile transfer diagnostics.
- **Figure 3 — Personalization thresholds:** simulation showing how the number of calibration cycles required for reliable personalization depends on stable person-specific signal and cycle-to-cycle variability.

## Data and generated results

The mcPHASES source data are not committed to this repository.

Generated analysis outputs under `results/` are also excluded from Git by default. This includes large or restricted artifacts such as participant-level prediction files, posterior NetCDF files, bootstrap replicate tables, simulation replicate tables, temporary model outputs, and run logs.

This keeps the repository small and avoids committing restricted or reproducible derived data. The final figures and the code required to regenerate the analyses are version-controlled.

## Reproducibility and software engineering

The repository was structured around reproducible scientific software practices, including:

- modular preprocessing and analysis code;
- explicit train/calibration/test separation;
- leakage-safe population training;
- deterministic simulation seeds;
- unit tests for core data-splitting logic;
- PEP 8 style checks;
- Python compilation checks;
- GitHub Actions continuous integration;
- explicit environment configuration;
- version-controlled final figures;
- separation of generated results from source code.

The primary environment target is **Python 3.12**. The final analyses were also executed successfully in a local development environment using Python 3.14.

## Installation

Installation is only needed for development or reproduction.

### Conda

```bash
conda env create -f environment.yml
conda activate beyond-average-cycle
```

### pip

```bash
python -m pip install -e ".[dev]"
```

## Running tests

The normal repository validation step is lightweight and does **not** rerun the Bayesian analyses.

```bash
pytest -q
python3 -m pycodestyle src tests scripts/aim2/*.py
python3 -m py_compile scripts/aim2/*.py
git diff --check
```

GitHub Actions performs the corresponding lightweight checks under Python 3.12.

## Optional: full scientific reproduction

The commands in this section are **only** for someone who intentionally wants to regenerate the scientific analyses from the required local mcPHASES data.

**These steps are not required for normal repository use, code review, grading, or figure inspection.**

### Near-term Aim 2 analysis

```bash
python3 scripts/aim2/build_within_interval_manifest.py
python3 scripts/aim2/build_centered_primary_folds.py
PYTENSOR_FLAGS="cxx=" python3 scripts/aim2/run_primary_hierarchical_models.py
python3 scripts/aim2/evaluate_primary_predictions.py
python3 scripts/aim2/make_figure1_near_term_personalization.py
```

### Longitudinal Aim 2 analysis

```bash
python3 scripts/aim2/build_longitudinal_manifest.py
python3 scripts/aim2/build_longitudinal_folds.py
PYTENSOR_FLAGS="cxx=" python3 scripts/aim2/run_longitudinal_hierarchical_models.py
python3 scripts/aim2/evaluate_longitudinal_predictions.py
python3 scripts/aim2/estimate_phase_profile_repeatability.py
python3 scripts/aim2/diagnose_empirical_profile_transfer.py
python3 scripts/aim2/diagnose_personalization_failure.py
PYTENSOR_FLAGS="cxx=" python3 scripts/aim2/oracle_recenter_longitudinal.py
python3 scripts/aim2/diagnose_residual_phase_drift.py
python3 scripts/aim2/make_figure2_longitudinal_mechanisms.py
```

`oracle_recenter_longitudinal.py` is a retrospective diagnostic that uses the observed 2024 baseline. It is not a prospective prediction model.

### Simulation analysis

```bash
python3 scripts/aim2/estimate_simulation_components.py
python3 scripts/aim2/estimate_phase_profile_repeatability.py
python3 scripts/aim2/run_cycle_depth_simulation.py
python3 scripts/aim2/run_cycle_depth_sensitivity.py
python3 scripts/aim2/run_variance_cycle_thresholds.py
python3 scripts/aim2/run_nominal_precision_check.py
python3 scripts/aim2/make_figure3_personalization_thresholds.py
```

On the development Mac, PyTensor model fitting used:

```bash
PYTENSOR_FLAGS="cxx=" python3 <script>
```

This is a platform-specific workaround and may not be necessary on other systems.

## Reproducibility verification

The final longitudinal reproducibility rerun regenerated all nine longitudinal folds. The resulting aggregate fold-results file and prediction file were **byte-for-byte identical** to the preserved original outputs under SHA-256 comparison.

The longitudinal rerun produced:

- 18 fitted models across 9 target participants;
- 0 divergences;
- worst observed maximum R-hat of 1.0155;
- minimum bulk ESS of 698.6;
- minimum tail ESS of 736.5;
- 701 held-out 2024 evaluation days.

## Development scripts

Exploratory and pilot scripts are retained under:

```text
scripts/aim2/development/
```

They are preserved for provenance but are not part of the final supported analysis workflow.

## Interpretation

The project distinguishes among three different questions:

1. whether physiological features vary systematically across menstrual phase;
2. whether those patterns are reproducible within individuals;
3. whether learning an individual's prior pattern improves prediction of future phase.

These are not equivalent. A participant can show a recognizable phase-associated physiological pattern while still providing little transferable predictive benefit if the magnitude, baseline, or direction of that pattern changes across cycles or years.

## License and data access

This repository contains analysis code and derived figures for the course project. Access to the original mcPHASES dataset is governed separately by the dataset's distribution and data-use terms.
