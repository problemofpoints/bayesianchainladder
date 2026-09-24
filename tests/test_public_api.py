"""Every public name in the feature modules must be exported from the package."""

import importlib
import inspect

import bayesianchainladder as bcl

MODULES = [
    "bayesianchainladder.analytic",
    "bayesianchainladder.cdr",
    "bayesianchainladder.datasets",
    "bayesianchainladder.linkratio",
    "bayesianchainladder.riskmeasures",
    "bayesianchainladder.sensitivity",
]


def test_feature_modules_are_fully_exported():
    missing = []
    for name in MODULES:
        mod = importlib.import_module(name)
        for attr, obj in vars(mod).items():
            if attr.startswith("_") or not (
                inspect.isfunction(obj) or inspect.isclass(obj)
            ):
                continue
            if getattr(obj, "__module__", None) != name:
                continue  # re-exported import, not defined here
            if attr not in bcl.__all__:
                missing.append(f"{name}.{attr}")
    assert not missing, f"not in bayesianchainladder.__all__: {missing}"


def test_all_names_resolve():
    for name in bcl.__all__:
        assert hasattr(bcl, name), name
