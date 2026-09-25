# Results directory

Generated analysis outputs belong here and are ignored by Git by default.

The preprocessing script writes files such as:

```text
results/preprocessed/
├── mcphases_daily_master.csv
├── mcphases_daily_complete_cycles.csv
├── mcphases_cycle_qc.csv
└── mcphases_participant_qc.csv
```

Do not commit any result that contains restricted participant-level data unless the data-use agreement explicitly permits it.
