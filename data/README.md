# Data directory

The mcPHASES dataset is restricted-access PhysioNet data and must **not** be committed to this repository.

After obtaining access from PhysioNet, place the downloaded CSV files in a local directory such as:

```text
data/mcphases/
├── hormones_and_selfreport.csv
├── resting_heart_rate.csv
├── computed_temperature.csv
└── ...
```

The repository `.gitignore` excludes everything under `data/` except this README.

Dataset: https://physionet.org/content/mcphases/1.0.0/
