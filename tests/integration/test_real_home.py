"""Integration tests and their fixtures resolve the developer's real global root."""

from pathlib import Path

import pytest

from lilbee.core.system import canonical_models_dir
from tests.conftest import REAL_GLOBAL_ROOT


@pytest.fixture(scope="module")
def models_dir_at_module_setup() -> Path:
    return canonical_models_dir()


def test_integration_tests_and_their_module_fixtures_use_the_real_models_dir(
    models_dir_at_module_setup,
):
    assert models_dir_at_module_setup == REAL_GLOBAL_ROOT / "models"
    assert canonical_models_dir() == REAL_GLOBAL_ROOT / "models"
