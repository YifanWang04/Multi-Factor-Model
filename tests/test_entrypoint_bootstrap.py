import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EntrypointBootstrapTests(unittest.TestCase):
    def _assert_entrypoint_loads_outside_project_root(self, script: Path) -> None:
        probe = (
            "import runpy; "
            f"runpy.run_path({str(script)!r}, run_name='entrypoint_import_probe')"
        )

        completed = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            cwd=ROOT.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_multi_factor_entrypoint_can_load_outside_project_root(self):
        self._assert_entrypoint_loads_outside_project_root(
            ROOT / "analysis" / "single_factor" / "run_multi_factor_test.py"
        )

    def test_factor_and_data_entrypoints_can_load_outside_project_root(self):
        scripts = [
            ROOT / "data" / "pull_yfinance_data.py",
            ROOT / "factor_pipeline" / "build_factors.py",
            ROOT / "factor_pipeline" / "process_factors.py",
        ]
        for script in scripts:
            with self.subTest(script=script):
                self._assert_entrypoint_loads_outside_project_root(script)


if __name__ == "__main__":
    unittest.main()
