import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "jab.py"


def test_script_runs_without_install(tmp_path):
    # -I ignores PYTHONPATH and user site-packages, so this only works if the
    # script finds the package next to itself.
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "status"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={},
    )
    assert result.returncode == 0, result.stderr
    assert "backend: not set" in result.stdout
