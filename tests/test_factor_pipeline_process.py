import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DataProcessExcelWriteTests(unittest.TestCase):
    @staticmethod
    def _write_input(path: Path) -> None:
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        factor = pd.DataFrame(
            {
                "AAA": [1.0, 2.0, 3.0, 4.0, 5.0],
                "BBB": [2.0, 4.0, 6.0, 8.0, 10.0],
                "CCC": [3.0, 6.0, 9.0, 12.0, 15.0],
            },
            index=dates,
        )
        factor.index.name = "Date"
        factor.to_excel(path, sheet_name="factor")

    def test_transient_excel_write_error_is_retried(self):
        from factor_pipeline import process_factors

        real_atomic_excel_writer = process_factors.atomic_excel_writer
        call_count = 0

        @contextmanager
        def fail_first_open(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError(22, "simulated transient write failure")
            with real_atomic_excel_writer(*args, **kwargs) as writer:
                yield writer

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            input_path = Path(tmp_dir) / "factor_alpha001.xlsx"
            output_path = Path(tmp_dir) / "factor_alpha001_processed.xlsx"
            self._write_input(input_path)

            with patch.object(
                process_factors,
                "atomic_excel_writer",
                new=fail_first_open,
            ):
                process_factors.process_factor_excel(input_path, output_path)

            self.assertEqual(call_count, 2)
            saved = pd.read_excel(
                output_path,
                sheet_name="factor",
                index_col=0,
            )
            self.assertEqual(saved.shape, (5, 3))

    def test_persistent_write_error_preserves_previous_workbook(self):
        from factor_pipeline import process_factors

        call_count = 0

        @contextmanager
        def always_fail_open(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise OSError(22, "simulated persistent write failure")
            yield

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp_dir:
            input_path = Path(tmp_dir) / "factor_alpha001.xlsx"
            output_path = Path(tmp_dir) / "factor_alpha001_processed.xlsx"
            self._write_input(input_path)
            previous_contents = b"previous-valid-workbook"
            output_path.write_bytes(previous_contents)

            with patch.object(
                process_factors,
                "atomic_excel_writer",
                new=always_fail_open,
            ):
                with self.assertRaises(OSError):
                    process_factors.process_factor_excel(input_path, output_path)

            self.assertEqual(call_count, 3)
            self.assertEqual(output_path.read_bytes(), previous_contents)


if __name__ == "__main__":
    unittest.main()
