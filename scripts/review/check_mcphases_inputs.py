"""Check local authorized mcPHASES files before scheduling long workflows.

PhysioNet mcPHASES 1.0.0 is restricted-access data. This script never
prompts for or stores passwords, API keys, tokens, or raw-data contents.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RAW_FILES = (
    "hormones_and_selfreport.csv",
    "resting_heart_rate.csv",
    "heart_rate_variability_details.csv",
    "computed_temperature.csv",
    "respiratory_rate_summary.csv",
)
SOURCE = "https://physionet.org/content/mcphases/1.0.0/"


def check_source_directory(directory: Path):
    """Validate presence, nonempty headers, and data-file readability."""
    directory = Path(directory).expanduser()
    problems = []
    for name in RAW_FILES:
        file = directory / name
        if not file.is_file():
            problems.append(f"Missing: {name}")
            continue
        try:
            with file.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.reader(stream)
                first = next(reader, None)
                second = next(reader, None)
        except (OSError, UnicodeError, csv.Error) as exc:
            problems.append(f"Unreadable {name}: {type(exc).__name__}")
            continue
        if not first or len(first) < 2 or not second:
            problems.append(f"Empty/malformed CSV or no data rows: {name}")
    return {
        "ok": not problems,
        "local_data_dir": str(directory.resolve()),
        "expected_file_count": len(RAW_FILES),
        "problems": problems,
        "source": SOURCE,
        "access_policy": (
            "Credentialed/restricted; obtain authorized access "
            "via PhysioNet"
        ),
        "warning": "Never commit raw CSVs or credentials to GitHub",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    result = check_source_directory(args.data_dir)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
