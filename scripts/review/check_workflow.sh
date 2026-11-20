#!/usr/bin/env bash
# Conservative one-command, preflighted Aim 2 Snakemake dry-run.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_DIR="${1:-data/mcphases}"
TARGET="${2:-aim2}"
python scripts/review/check_mcphases_inputs.py --data-dir "$DATA_DIR"
# Never remove --dry-run here. A full model run requires separate authorization.
snakemake --snakefile workflow/Snakefile --cores 1 \
  --config "data_dir=$DATA_DIR" --dry-run "$TARGET"
