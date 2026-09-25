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

## Preprocessing and quality-control decisions

The primary analysis uses four wearable physiological signals:

- resting heart rate
- HRV, represented by nightly median RMSSD
- nightly skin temperature
- respiratory rate during full sleep

### Physiological preprocessing

Resting heart rate, RMSSD, and respiratory-rate values less than or equal to
zero are treated as invalid device values and excluded before daily/nightly
aggregation.

HRV is supplied as 5-minute RMSSD windows during sleep. The primary pipeline
uses the median of positive RMSSD windows within each participant-night. A
sensitivity check comparing this approach with RMSSD calculated only from
windows with Fitbit coverage >= 0.90 showed minimal impact on nightly values,
so no additional HRV coverage filter or arbitrary upper RMSSD cutoff is used.

No missing physiological measurements are imputed.

### Menstrual-cycle definition

The mcPHASES reference analysis identifies 192 sequences containing all four
phase labels:

- Menstrual
- Follicular
- Fertility
- Luteal

For the present project, the primary biological analysis uses a stricter,
menses-anchored definition. A cycle begins at an observed Menstrual phase and
must subsequently contain all four phases in the expected order.

This produces 147 menses-anchored complete cycles.

For Aim 1, a cycle must additionally:

1. be closed by an observed subsequent menstrual onset;
2. have valid phase order; and
3. have >=70% daily coverage for each of the four core physiological signals.

A stricter >=80% coverage definition is retained as a sensitivity-analysis
flag rather than replacing the primary threshold.

### Aim 1 cycle eligibility and analysis cohort

Cycle eligibility and participant eligibility are distinct.

At the **cycle level**, a cycle is flagged `aim1_primary_eligible` if it:

1. is a complete menses-anchored cycle;
2. has the expected phase order;
3. is closed by an observed subsequent menstrual onset; and
4. has >=70% daily coverage for each of the four core physiological signals.

The preprocessing pipeline identifies:

- 42 total participants
- 5,659 participant-days
- 147 menses-anchored complete cycles
- 93 cycles passing the Aim 1 cycle-level QC criteria
- 37 participants contributing at least one such cycle

However, the primary Aim 1 question concerns **within-person reproducibility
across repeated cycles**. A participant therefore needs at least two eligible
cycles to contribute to the primary analysis.

The 93 cycle-level eligible cycles are distributed as follows:

- 25 participants have >=2 eligible cycles and contribute 81 cycles
- 12 participants have exactly 1 eligible cycle and contribute 12 cycles
- 5 participants have no eligible cycles

Therefore, the **primary Aim 1 repeated-cycle analysis cohort is 25
participants contributing 81 cycles**, not all 93 cycle-level eligible cycles.

Among these data, 16 participants have primary-eligible cycles represented in
both study intervals and can contribute to the secondary long-term stability
analysis.

Participants, rather than individual cycles, are treated as the independent
biological units in downstream inference.

## Analysis handoff

The shared preprocessing and QC pipeline is complete before Aim 1 and Aim 2
development begins. Both analyses must start from the same cleaned data and
must not independently redefine the shared preprocessing rules.

### Shared starting data

The main shared daily dataset is:

`results/preprocessed/mcphases_daily_master.csv`

Cycle-level QC information is stored in:

`results/preprocessed/mcphases_cycle_qc.csv`

Participant-level QC information is stored in:

`results/preprocessed/mcphases_participant_qc.csv`

These generated files are not committed to Git because they are derived from
restricted mcPHASES data. Each collaborator should obtain the mcPHASES dataset
independently and regenerate these files using the documented preprocessing
command.

### Aim 1: physiological signature reproducibility

Aim 1 asks whether menstrual-cycle physiological signatures are reproducible
within individuals.

A cycle can pass cycle-level Aim 1 QC without necessarily being usable for the
repeated-cycle analysis. The pipeline currently identifies:

- 93 cycles passing cycle-level Aim 1 QC
- 37 participants contributing at least one eligible cycle
- 25 participants contributing at least two eligible cycles
- 81 eligible cycles from those 25 repeated-cycle participants
- 16 participants with eligible cycles represented in both study intervals

Therefore, the primary Aim 1 repeated-cycle analysis cohort is 25 participants
contributing 81 cycles.

### Aim 2: personalized phase prediction

Aim 2 asks whether personalized wearable physiology improves prediction of
hormone-defined menstrual phase relative to population-level approaches.

Aim 2 starts from the same cleaned daily master dataset but defines its own
analysis-specific eligibility criteria. It should not automatically restrict
the dataset to `aim1_primary_eligible` cycles because prediction does not have
the same requirement for repeated, closed cycles as Aim 1.

Participant-level separation between training and testing must be preserved
when evaluating population-level prediction models to prevent data leakage.

### Shared preprocessing contract

After this handoff, Aim-specific branches should not independently change
shared preprocessing choices such as:

- signal definitions or merge keys
- invalid-value filtering
- nightly HRV aggregation
- menstrual phase normalization
- menses-anchored cycle definition
- cycle closure logic
- physiological coverage calculations

If a genuine preprocessing bug is discovered, it should be fixed in a
dedicated branch with an accompanying test, merged into `main`, and then
incorporated into both Aim branches.
