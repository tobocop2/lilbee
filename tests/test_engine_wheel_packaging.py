"""The engine wheel must carry payload that sits in a subdirectory of bin/.

rocm is the first backend to put anything there: rocBLAS reads its Tensile kernels
from ``bin/rocblas/library`` at runtime rather than linking them. A single-level
``bin/*`` glob built a wheel that passed every other check and would have died on the
first matrix multiply, which is the failure the bundler's own comments warn about.

The behavioural check is ``tools/qa/assert_engine_bundle.py`` against a real wheel in
CI. This one is here so the mistake costs seconds rather than a 70-minute rocm build.
"""

from __future__ import annotations

import pathlib
import tomllib

_PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "packaging/engine-wheel/pyproject.toml"


def _package_data() -> list[str]:
    config = tomllib.loads(_PYPROJECT.read_text())
    return config["tool"]["setuptools"]["package-data"]["lilbee_engine"]


def test_package_data_reaches_into_subdirectories():
    """setuptools expands bin/* one level only; nested payload needs its own pattern."""
    assert any("**" in pattern for pattern in _package_data()), (
        "package-data has no recursive pattern, so bin/rocblas/library would be dropped"
    )


def test_package_data_still_covers_the_top_level():
    """bin/**/* alone does not match the binaries sitting directly in bin/."""
    assert "bin/*" in _package_data()
