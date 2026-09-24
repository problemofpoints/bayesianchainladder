"""Execute the documentation notebooks end to end (slow)."""

import pathlib
import subprocess
import sys

import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "notebooks"


@pytest.mark.slow
def test_modus_operandi_notebook_builds_and_executes(tmp_path):
    subprocess.run(
        [sys.executable, str(NOTEBOOKS / "build_modus_operandi.py"), str(tmp_path)],
        check=True,
    )
    nb = tmp_path / "modus_operandi.ipynb"
    assert nb.exists()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=1800",
            "--output",
            "executed.ipynb",
            str(nb),
        ],
        check=True,
    )
    text = (tmp_path / "executed.ipynb").read_text()
    assert "DrPeterEngland/StochasticReserving" in text
    assert "Step 15" in text
