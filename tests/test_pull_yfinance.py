import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas as pd

from data import pull_yfinance_data as puller
from data import data_config


class PullYfinanceTests(unittest.TestCase):
    def test_profile_download_start_overrides_default_base_and_offset(self):
        with patch.dict(
            os.environ,
            {
                "REBALANCE_DATA_START_DATE": "2020-01-01",
                "REBALANCE_OFFSET_DAYS": "1000",
            },
            clear=False,
        ):
            self.assertEqual(data_config.yfinance_pull_start_date(), "2020-01-01")

    def test_empty_or_failed_symbol_is_skipped_with_gbk_safe_notice(self):
        good_frame = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-07-14")],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Adj Close": [100.5],
                "Volume": [1_000_000],
                "Ticker": ["GOOD"],
            }
        )

        def fake_download(symbol, start_date, end_date):
            if symbol == "EMPTY":
                return pd.DataFrame()
            if symbol == "ERROR":
                raise RuntimeError("simulated download failure")
            return good_frame.copy()

        old_run_dir = os.environ.get("REBALANCE_RUN_DIR")
        old_preserve = os.environ.get(puller.PRESERVE_PRICE_SCALE_ENV_VAR)
        output_bytes = io.BytesIO()
        gbk_stdout = io.TextIOWrapper(output_bytes, encoding="gbk", errors="strict")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["REBALANCE_RUN_DIR"] = tmp
                os.environ[puller.PRESERVE_PRICE_SCALE_ENV_VAR] = "0"
                with (
                    patch.object(puller, "_download_one_symbol", side_effect=fake_download),
                    patch.object(puller, "_backfill_completed_close_bars", return_value={}),
                    redirect_stdout(gbk_stdout),
                ):
                    output_path = puller.main(tickers=["EMPTY", "ERROR", "GOOD"])

                self.assertTrue(os.path.isfile(output_path))
                with pd.ExcelFile(output_path) as workbook:
                    self.assertEqual(workbook.sheet_names, ["GOOD"])
        finally:
            if old_run_dir is None:
                os.environ.pop("REBALANCE_RUN_DIR", None)
            else:
                os.environ["REBALANCE_RUN_DIR"] = old_run_dir
            if old_preserve is None:
                os.environ.pop(puller.PRESERVE_PRICE_SCALE_ENV_VAR, None)
            else:
                os.environ[puller.PRESERVE_PRICE_SCALE_ENV_VAR] = old_preserve

        gbk_stdout.flush()
        output = output_bytes.getvalue().decode("gbk")
        self.assertIn("[SKIP] EMPTY: no data returned", output)
        self.assertIn("[SKIP] ERROR: download failed (RuntimeError: simulated download failure)", output)
        self.assertIn("Skipped 2/3 tickers", output)


if __name__ == "__main__":
    unittest.main()
