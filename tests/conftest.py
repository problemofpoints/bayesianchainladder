"""Pytest configuration and shared fixtures."""

import pytest


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
