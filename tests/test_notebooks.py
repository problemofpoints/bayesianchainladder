"""Execute the documentation notebooks end to end (slow)."""

import pathlib
import subprocess
import sys

import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "notebooks"

BUILD_SCRIPTS = [
    ("build_modus_operandi.py", "modus_operandi.ipynb"),
    ("build_evw_2019.py", "evw_2019_one_year_view.ipynb"),
    ("build_ev_2006.py", "ev_2006_predictive_distributions.ipynb"),
]


@pytest.mark.slow
@pytest.mark.parametrize("build_script, notebook_name", BUILD_SCRIPTS)
def test_notebook_builds_and_executes(tmp_path, build_script, notebook_name):
    subprocess.run(
        [sys.executable, str(NOTEBOOKS / build_script), str(tmp_path)],
        check=True,
    )
    nb = tmp_path / notebook_name
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
        timeout=1800,
    )
    text = (tmp_path / "executed.ipynb").read_text()
    assert "DrPeterEngland/StochasticReserving" in text
    if notebook_name == "modus_operandi.ipynb":
        assert "Step 15" in text
