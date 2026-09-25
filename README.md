# Beyond the Average Cycle

## Person-specific physiological signatures across the menstrual cycle

This repository contains reproducible code for a CSCI 6118 / Software Engineering for Scientists group project using the **mcPHASES** dataset.

### Scientific question

Population-level menstrual-cycle changes in wearable physiology are well established, but population averages may not accurately describe an individual person's response. We ask whether menstrual-cycle-associated physiological patterns are reproducible within individuals and whether personalization improves interpretation of wearable physiology.

### Aims

**Aim 1 — Reproducibility of person-specific physiology**  
Test whether repeated cycles from the same participant are more physiologically similar than cycles from different participants.

**Aim 2 — Personalized versus population models**  
Test whether participant-specific calibration improves prediction of hormone-defined menstrual phase compared with a pooled population model.

The two aims share a common preprocessing pipeline but can be completed independently.

## Dataset

We use the restricted-access **mcPHASES** dataset from PhysioNet:

https://physionet.org/content/mcphases/1.0.0/

The dataset contains repeated menstrual-cycle measurements including hormone/self-report data and wearable physiology. Raw participant-level data are **not included in this repository**.

### Data-use rule

Do **not** commit downloaded mcPHASES files to GitHub. The `data/` directory is ignored except for `data/README.md`.

## Repository structure

```text
beyond-average-cycle/
├── .github/
│   └── workflows/
│       └── test.yml
├── data/
│   └── README.md
├── results/
│   └── README.md
├── src/
│   └── mcphases/
│       ├── __init__.py
│       └── preprocess.py
├── tests/
│   └── test_preprocess.py
├── .gitignore
├── environment.yml
├── pyproject.toml
└── README.md
```

## Reproduce the environment

Using `mamba` or `conda`:

```bash
mamba env create -f environment.yml
mamba activate beyond-average-cycle
python -m pip install -e .
```

## Obtain the data

1. Create a PhysioNet account.
2. Complete the mcPHASES data-use agreement.
3. Download the dataset.
4. Place the CSV files locally under `data/mcphases/`.

For example:

```text
data/mcphases/
├── hormones_and_selfreport.csv
├── resting_heart_rate.csv
├── computed_temperature.csv
└── ...
```

## Run preprocessing

```bash
python -m mcphases.preprocess \
    --data-dir data/mcphases \
    --output-dir results/preprocessed
```

The current pipeline:

- validates required columns;
- standardizes mcPHASES phase labels;
- parses LH, estrogen/E3G, and PdG measurements;
- identifies menstrual cycles within study intervals;
- calculates cycle day and normalized cycle position;
- flags complete cycles and invalid phase order;
- aggregates resting heart rate and nightly temperature to participant-day values;
- merges physiology with hormone/self-report data;
- identifies repeated-cycle and long-term-analysis candidates; and
- writes participant- and cycle-level QC summaries.

No imputation or model scaling is performed during shared preprocessing. Any scaling used for prediction must be fitted only on training data to prevent data leakage.

## Outputs

The preprocessing command generates:

```text
results/preprocessed/
├── mcphases_daily_master.csv
├── mcphases_daily_complete_cycles.csv
├── mcphases_cycle_qc.csv
└── mcphases_participant_qc.csv
```

These files may contain restricted participant-level information and are ignored by Git.

## Tests

Run:

```bash
pytest -q
```

Check PEP 8 style:

```bash
pycodestyle src tests
```

GitHub Actions runs both checks automatically on pushes and pull requests.

## Reproducible collaboration workflow

For each change:

```bash
git switch main
git pull
git switch -c descriptive-branch-name
# make changes
git add <files>
git commit -m "Describe the change"
git push -u origin descriptive-branch-name
```

Then open a pull request into `main` and have another group member review it before merging.

## Planned analyses

### Aim 1

1. Define interpretable cycle-level physiological features.
2. Quantify within-person and between-person cycle similarity.
3. Test the within-person similarity advantage with participant-label permutation tests.
4. Evaluate signal-specific reproducibility.
5. If feasible, test long-term stability across study intervals.

### Aim 2

1. Define hormone-derived phase as the prediction target.
2. Establish a simple calendar-based baseline.
3. Fit population models with participant-level cross-validation.
4. Fit personalized models using prior cycles from the same participant.
5. Compare balanced accuracy / macro-F1 within participants.

## Reproducibility principles

- Keep raw restricted data out of Git.
- Keep preprocessing deterministic.
- Record package versions in `environment.yml`.
- Write reusable functions instead of analysis-only notebook code.
- Add unit tests whenever preprocessing behavior changes.
- Use participant-level train/test separation for predictive analyses.
- Generate final figures from scripts so they can be reproduced from the processed inputs.
