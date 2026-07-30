import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BuildFactorsOutputTests(unittest.TestCase):
    @staticmethod
    def _sample_data_dict():
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        close = pd.DataFrame(
            {"AAA": range(30), "BBB": range(30, 60)},
            index=dates,
            dtype=float,
        )
        return {
            "close": close,
            "returns": close.pct_change(),
        }

    def test_failed_factor_write_preserves_previous_workbook_and_fails_run(self):
        from factor_pipeline import build_factors

        data_dict = self._sample_data_dict()

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            target = Path(tmp_dir) / "factor_alpha001.xlsx"
            previous_contents = b"previous-valid-workbook"
            target.write_bytes(previous_contents)

            def fail_after_direct_target_is_opened(destination, *args, **kwargs):
                if isinstance(destination, (str, os.PathLike)):
                    Path(destination).write_bytes(b"partial-workbook")
                raise OSError(22, "simulated transient write failure")

            with (
                patch.object(build_factors, "FACTOR_RAW_DIR", tmp_dir),
                patch.dict(
                    os.environ,
                    {"REBALANCE_SELECTED_FACTORS": "alpha001", "QQQ_MAX_WORKERS": "1"},
                ),
                patch.object(
                    pd.DataFrame,
                    "to_excel",
                    new=fail_after_direct_target_is_opened,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "alpha001"):
                    build_factors.build_and_save_all_factors(data_dict)

            self.assertEqual(target.read_bytes(), previous_contents)

    def test_transient_excel_write_error_is_retried(self):
        from factor_pipeline import build_factors

        real_to_excel = pd.DataFrame.to_excel
        call_count = 0

        def fail_first_write(destination, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError(22, "simulated transient write failure")
            return real_to_excel(destination, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            target = Path(tmp_dir) / "factor_alpha001.xlsx"
            with (
                patch.object(build_factors, "FACTOR_RAW_DIR", tmp_dir),
                patch.dict(
                    os.environ,
                    {"REBALANCE_SELECTED_FACTORS": "alpha001", "QQQ_MAX_WORKERS": "1"},
                ),
                patch.object(pd.DataFrame, "to_excel", new=fail_first_write),
            ):
                built = build_factors.build_and_save_all_factors(
                    self._sample_data_dict()
                )

            self.assertEqual(built, [("alpha001", str(target))])
            self.assertEqual(call_count, 2)
            self.assertTrue(target.is_file())
            saved = pd.read_excel(target, sheet_name="factor", index_col=0)
            self.assertEqual(saved.shape, (30, 2))


if __name__ == "__main__":
    unittest.main()
