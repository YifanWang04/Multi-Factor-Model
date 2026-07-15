import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.price_snapshot import (  # noqa: E402
    PRICE_BASE_RUN_DIR_ENV_VAR,
    apply_preserved_price_scale,
    load_manifest,
    manifest_adjustments_frame,
    manifest_path_for_run,
    write_manifest,
)
from qqq_core.paths import price_filename  # noqa: E402


class PriceSnapshotTests(unittest.TestCase):
    def test_appended_rows_are_scaled_back_to_previous_price_scale(self):
        factor = 0.9535
        base = _price_frame("2026-06-15", periods=10, price_start=100.0, volume_start=1000.0)
        fresh_overlap = base.copy()
        for col in _PRICE_COLUMNS:
            fresh_overlap[col] = fresh_overlap[col] / factor
        fresh_overlap["Volume"] = fresh_overlap["Volume"] * factor

        fresh_new = _price_frame("2026-06-29", periods=2, price_start=120.0, volume_start=2000.0)
        for col in _PRICE_COLUMNS:
            fresh_new[col] = fresh_new[col] / factor
        fresh_new["Volume"] = fresh_new["Volume"] * factor
        fresh = pd.concat([fresh_overlap, fresh_new], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmp:
            base_run = Path(tmp) / "2026-06-24_155408_strategy11_offset0"
            current_run = Path(tmp) / "2026-07-09_152712_strategy11_offset0"
            (base_run / "data").mkdir(parents=True)
            current_run.mkdir(parents=True)
            with pd.ExcelWriter(base_run / "data" / price_filename(0)) as writer:
                base.to_excel(writer, sheet_name="HON", index=False)

            old_override = os.environ.get(PRICE_BASE_RUN_DIR_ENV_VAR)
            os.environ[PRICE_BASE_RUN_DIR_ENV_VAR] = str(base_run)
            try:
                merged, result = apply_preserved_price_scale(
                    {"HON": fresh},
                    run_dir=current_run,
                    profile_name="Strategy11",
                    offset=0,
                )
            finally:
                if old_override is None:
                    os.environ.pop(PRICE_BASE_RUN_DIR_ENV_VAR, None)
                else:
                    os.environ[PRICE_BASE_RUN_DIR_ENV_VAR] = old_override

        got = merged["HON"].reset_index(drop=True)
        self.assertEqual(len(got), 12)
        pd.testing.assert_series_equal(got.loc[:9, "Close"], base["Close"], check_names=False)
        self.assertAlmostEqual(float(got.loc[10, "Close"]), float(fresh_new.loc[0, "Close"] * factor))
        self.assertAlmostEqual(float(got.loc[10, "Volume"]), float(fresh_new.loc[0, "Volume"] / factor))
        self.assertEqual(len(result.adjustments), 1)
        self.assertAlmostEqual(result.adjustments[0].price_factor, factor, places=8)

    def test_appended_rows_keep_fresh_scale_when_no_stable_adjustment_exists(self):
        base = _price_frame("2026-06-15", periods=10, price_start=100.0, volume_start=1000.0)
        fresh_new = _price_frame("2026-06-29", periods=2, price_start=120.0, volume_start=2000.0)
        fresh = pd.concat([base, fresh_new], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmp:
            base_run = Path(tmp) / "2026-06-24_155408_strategy11_offset0"
            current_run = Path(tmp) / "2026-07-09_152712_strategy11_offset0"
            (base_run / "data").mkdir(parents=True)
            current_run.mkdir(parents=True)
            with pd.ExcelWriter(base_run / "data" / price_filename(0)) as writer:
                base.to_excel(writer, sheet_name="AAPL", index=False)

            old_override = os.environ.get(PRICE_BASE_RUN_DIR_ENV_VAR)
            os.environ[PRICE_BASE_RUN_DIR_ENV_VAR] = str(base_run)
            try:
                merged, result = apply_preserved_price_scale(
                    {"AAPL": fresh},
                    run_dir=current_run,
                    profile_name="Strategy11",
                    offset=0,
                )
            finally:
                if old_override is None:
                    os.environ.pop(PRICE_BASE_RUN_DIR_ENV_VAR, None)
                else:
                    os.environ[PRICE_BASE_RUN_DIR_ENV_VAR] = old_override

        got = merged["AAPL"].reset_index(drop=True)
        self.assertEqual(len(got), 12)
        self.assertEqual(result.adjustments, [])
        pd.testing.assert_series_equal(got.loc[10:, "Close"].reset_index(drop=True), fresh_new["Close"], check_names=False)

    def test_explicit_base_run_dir_takes_precedence_over_environment_override(self):
        env_base = _price_frame("2026-06-15", periods=10, price_start=500.0, volume_start=5000.0)
        profile_base = _price_frame("2026-06-15", periods=10, price_start=100.0, volume_start=1000.0)
        fresh = pd.concat(
            [profile_base, _price_frame("2026-06-29", periods=1, price_start=120.0, volume_start=2000.0)],
            ignore_index=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            env_run = Path(tmp) / "env_base"
            profile_run = Path(tmp) / "profile_base"
            current_run = Path(tmp) / "current"
            for run_dir, data in ((env_run, env_base), (profile_run, profile_base)):
                (run_dir / "data").mkdir(parents=True)
                with pd.ExcelWriter(run_dir / "data" / price_filename(0)) as writer:
                    data.to_excel(writer, sheet_name="AAPL", index=False)
            current_run.mkdir(parents=True)

            old_override = os.environ.get(PRICE_BASE_RUN_DIR_ENV_VAR)
            os.environ[PRICE_BASE_RUN_DIR_ENV_VAR] = str(env_run)
            try:
                merged, result = apply_preserved_price_scale(
                    {"AAPL": fresh},
                    run_dir=current_run,
                    profile_name="Strategy11",
                    offset=0,
                    base_run_dir=profile_run,
                )
            finally:
                if old_override is None:
                    os.environ.pop(PRICE_BASE_RUN_DIR_ENV_VAR, None)
                else:
                    os.environ[PRICE_BASE_RUN_DIR_ENV_VAR] = old_override

        self.assertEqual(result.base_run_dir, str(profile_run.resolve()))
        pd.testing.assert_series_equal(
            merged["AAPL"].reset_index(drop=True).loc[:9, "Close"],
            profile_base["Close"],
            check_names=False,
        )

    def test_manifest_can_be_loaded_and_expanded_for_reports(self):
        factor = 1.057
        base = _price_frame("2026-06-15", periods=10, price_start=100.0, volume_start=1000.0)
        fresh_overlap = base.copy()
        for col in _PRICE_COLUMNS:
            fresh_overlap[col] = fresh_overlap[col] / factor
        fresh_new = _price_frame("2026-06-29", periods=1, price_start=120.0, volume_start=2000.0)
        for col in _PRICE_COLUMNS:
            fresh_new[col] = fresh_new[col] / factor
        fresh = pd.concat([fresh_overlap, fresh_new], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmp:
            base_run = Path(tmp) / "2026-06-24_155408_strategy11_offset0"
            current_run = Path(tmp) / "2026-07-09_152712_strategy11_offset0"
            (base_run / "data").mkdir(parents=True)
            current_run.mkdir(parents=True)
            with pd.ExcelWriter(base_run / "data" / price_filename(0)) as writer:
                base.to_excel(writer, sheet_name="SPGI", index=False)

            old_override = os.environ.get(PRICE_BASE_RUN_DIR_ENV_VAR)
            os.environ[PRICE_BASE_RUN_DIR_ENV_VAR] = str(base_run)
            try:
                _, result = apply_preserved_price_scale(
                    {"SPGI": fresh},
                    run_dir=current_run,
                    profile_name="Strategy11",
                    offset=0,
                )
            finally:
                if old_override is None:
                    os.environ.pop(PRICE_BASE_RUN_DIR_ENV_VAR, None)
                else:
                    os.environ[PRICE_BASE_RUN_DIR_ENV_VAR] = old_override

            write_manifest(manifest_path_for_run(current_run), result)
            manifest = load_manifest(current_run)
            adjustments = manifest_adjustments_frame(manifest)

        self.assertEqual(len(adjustments), 1)
        self.assertEqual(adjustments.loc[0, "Ticker"], "SPGI")
        self.assertAlmostEqual(float(adjustments.loc[0, "Price_Factor"]), factor, places=8)


_PRICE_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Adj Open",
    "Adj High",
    "Adj Low",
)


def _price_frame(start: str, periods: int, price_start: float, volume_start: float) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    rows = []
    for i, date in enumerate(dates):
        price = price_start + i
        rows.append(
            {
                "Date": date,
                "Open": price,
                "High": price + 1.0,
                "Low": price - 1.0,
                "Close": price + 0.5,
                "Adj Close": price + 0.25,
                "Adj Open": price - 0.1,
                "Adj High": price + 0.9,
                "Adj Low": price - 0.9,
                "Volume": volume_start + i * 10,
            }
        )
    return pd.DataFrame(rows)
