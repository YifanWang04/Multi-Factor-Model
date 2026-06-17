"""Inspect a rebalance-day Excel report from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qqq_core.paths import ProjectPaths


EXPECTED_SHEETS = {
    "Rebalance_Config_Status": True,
    "Daily_Returns": True,
    "Cumulative_Returns": True,
    "Rebalance_Day_Status": False,
}


def find_latest_rebalance_report(paths: ProjectPaths | None = None) -> Path:
    """Return the newest rebalance-day report from standard and legacy layouts."""

    paths = paths or ProjectPaths.from_env()
    candidates: list[Path] = []
    candidates.extend(paths.rebalance_runs_dir.glob("*/reports/rebalance_day_report.xlsx"))
    candidates.extend(paths.output_dir.glob("rebalance_day_*/rebalance_day_report.xlsx"))
    candidates.extend(paths.output_dir.glob("rebalance_day_*/reports/rebalance_day_report.xlsx"))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "No rebalance_day_report.xlsx found under output/rebalance_runs "
            "or legacy output/rebalance_day_* directories."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def analyze_report(file_path: str | Path, preview_rows: int = 2) -> None:
    """Print workbook sheets, preview rows, and basic rebalance report checks."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Report file does not exist: {path}")

    xl = pd.ExcelFile(path)
    all_sheets = xl.sheet_names

    print("=" * 80)
    print("EXCEL FILE ANALYSIS")
    print("=" * 80)
    print(f"\nFile: {path}\n")
    print("ALL SHEETS FOUND:")
    for i, name in enumerate(all_sheets, 1):
        print(f"  {i}. {name}")

    print("\n" + "=" * 80)
    print("DETAILED SHEET ANALYSIS")
    print("=" * 80)

    for sheet_name in all_sheets:
        df = pd.read_excel(path, sheet_name=sheet_name)
        print(f"\n--- Sheet: {sheet_name} ---")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"First {preview_rows} rows:")
        print(df.head(preview_rows).to_string())

    print("\n" + "=" * 80)
    print("CONFIRMATION CHECKS")
    print("=" * 80)

    for i, (sheet, expected) in enumerate(EXPECTED_SHEETS.items(), 1):
        exists = sheet in all_sheets
        result = "PASS" if exists is expected else "FAIL"
        print(f'\n{i}. "{sheet}" exists: {exists}')
        print(f"   -> Expected: {expected}")
        print(f"   -> Result: {result}")

    _check_weight_threshold(path, all_sheets, "Current_Operations", 5)
    _check_weight_threshold(path, all_sheets, "All_Operations", 6)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)


def _check_weight_threshold(
    path: Path,
    all_sheets: list[str],
    sheet_name: str,
    check_number: int,
    threshold: float = 0.0001,
) -> None:
    if sheet_name not in all_sheets:
        print(f'\n{check_number}. "{sheet_name}" sheet not found -> SKIP')
        return

    df = pd.read_excel(path, sheet_name=sheet_name)
    if "Weight" not in df.columns:
        print(f'\n{check_number}. "{sheet_name}" - No "Weight" column found -> SKIP')
        return

    min_weight = df["Weight"].min()
    all_above_threshold = (df["Weight"] >= threshold).all()
    print(f'\n{check_number}. "{sheet_name}" - All Weight >= {threshold}:')
    print(f"   Min Weight: {min_weight}")
    print(f"   All >= {threshold}: {all_above_threshold}")
    print(f'   -> Result: {"PASS" if all_above_threshold else "FAIL"}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a rebalance-day Excel report")
    parser.add_argument(
        "report",
        nargs="?",
        help="Path to rebalance_day_report.xlsx. Defaults to the newest standard report.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=2,
        help="Number of rows to print from each sheet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = Path(args.report) if args.report else find_latest_rebalance_report()
    analyze_report(report, preview_rows=args.preview_rows)


if __name__ == "__main__":
    main()
