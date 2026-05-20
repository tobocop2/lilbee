"""Test fixtures for provider-level tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_supports_tools_cache():
    """Reset ``_supports_tools_cached`` between tests so monkeypatched
    metadata readers don't leak True/False across cases that share a path.
    """
    from lilbee.providers.llama_cpp.provider import _supports_tools_cached

    _supports_tools_cached.cache_clear()
    yield
    _supports_tools_cached.cache_clear()
