"""Integration test configuration — override unit test isolation for models_dir.

The root conftest._isolate_cfg sets models_dir to a temp dir for every test,
which breaks integration tests that need real models from the canonical location.
This fixture restores models_dir after the root fixture changes it.
"""

import pytest

from lilbee.config import cfg
from lilbee.platform import canonical_models_dir


@pytest.fixture(autouse=True)
def _preserve_models_dir():
    """Ensure models_dir stays at canonical location for integration tests."""
    cfg.models_dir = canonical_models_dir()
    yield
