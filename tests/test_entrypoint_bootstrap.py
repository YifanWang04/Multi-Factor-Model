import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EntrypointBootstrapTests(unittest.TestCase):
    def test_multi_factor_entrypoint_can_load_outside_project_root(self):
        script = ROOT / "analysis" / "single_factor" / "run_multi_factor_test.py"
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


if __name__ == "__main__":
    unittest.main()
