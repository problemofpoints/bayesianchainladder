"""Pytest configuration and shared fixtures."""

import chainladder as cl
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def quarterly_origin_triangle():
    """Cumulative paid triangle with quarterly origins and quarterly development.

    Eight origins (2020Q1 to 2021Q4), eight development quarters, valued at
    2021-12-31. Ultimates grow 5% per origin over a fixed paid pattern, so
    every origin has a distinct label and the chain ladder is well behaved.
    """
    origins = pd.period_range("2020Q1", "2021Q4", freq="Q")
    pattern = np.array([0.30, 0.55, 0.72, 0.84, 0.91, 0.96, 0.99, 1.00])
    valuation_date = pd.Timestamp("2021-12-31")
    rows = []
    for i, origin in enumerate(origins):
        ultimate = 1_000.0 * (1 + 0.05 * i)
        for j in range(len(pattern)):
            valuation = (origin + j).end_time.normalize()
            if valuation > valuation_date:
                continue
            rows.append(
                {
                    "origin": origin.start_time.strftime("%Y-%m-%d"),
                    "valuation": valuation.strftime("%Y-%m-%d"),
                    "paid": round(ultimate * pattern[j], 2),
                }
            )
    data = pd.DataFrame(rows)
    return cl.Triangle(
        data, origin="origin", development="valuation", columns="paid", cumulative=True
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (MCMC fitting)",
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow (MCMC fitting)")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is provided."""
    if config.getoption("--run-slow"):
        # Run all tests
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
