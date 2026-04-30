"""list_chat_models curated vs all modes, plus discover_api_models bridging."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from lilbee.core.config import cfg
from lilbee.providers.curated_models import (
    CURATED_CHAT_MODELS,
    TOP_N_FALLBACK,
    curated_ids,
)
from lilbee.providers.litellm_sdk import LitellmSdkBackend


@pytest.fixture
def fake_litellm():
    """A stand-in litellm module: tiny catalog with mode metadata."""
    gemini_models = (
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-2.0-pro-preview",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.0-pro",
        "gemini-1.0-pro-vision",
    )
    novel_models = (
        "novel-large-v1",
        "novel-medium-v1",
        "novel-small-v1",
        "novel-experimental-v9",
        *(f"novel-extra-{i}" for i in range(20)),
    )
    return SimpleNamespace(
        models_by_provider={
            "gemini": set(gemini_models),
            "novelco": set(novel_models),
        },
        model_cost={name: {"mode": "chat"} for name in gemini_models + novel_models},
    )


def test_curated_mode_returns_curated_ids_only(fake_litellm):
    backend = LitellmSdkBackend()
    with mock.patch.dict("sys.modules", {"litellm": fake_litellm}):
        result = backend.list_chat_models("gemini", mode="curated")
    upstream = fake_litellm.models_by_provider["gemini"]
    expected = [mid for mid in curated_ids("gemini") if mid in upstream]
    assert result == expected
    # Curated set excludes -exp and dated previews even though the
    # upstream catalog carries them.
    assert "gemini-2.0-flash-exp" not in result
    assert "gemini-2.0-pro-preview" not in result


def test_all_mode_returns_full_upstream_catalog(fake_litellm):
    backend = LitellmSdkBackend()
    with mock.patch.dict("sys.modules", {"litellm": fake_litellm}):
        result = backend.list_chat_models("gemini", mode="all")
    assert set(result) == fake_litellm.models_by_provider["gemini"]


def test_uncurated_provider_uses_top_n_fallback(fake_litellm):
    """A provider with no curated entry gets the alphabetical top-N."""
    assert "novelco" not in CURATED_CHAT_MODELS
    backend = LitellmSdkBackend()
    with mock.patch.dict("sys.modules", {"litellm": fake_litellm}):
        result = backend.list_chat_models("novelco", mode="curated")
    assert len(result) == TOP_N_FALLBACK
    assert result == sorted(fake_litellm.models_by_provider["novelco"])[:TOP_N_FALLBACK]


def test_curated_default_mode_matches_curated_explicit(fake_litellm):
    backend = LitellmSdkBackend()
    with mock.patch.dict("sys.modules", {"litellm": fake_litellm}):
        default = backend.list_chat_models("gemini")
        explicit = backend.list_chat_models("gemini", mode="curated")
    assert default == explicit


def test_show_all_api_models_flag_flips_discover_default():
    """discover_api_models honors cfg.show_all_api_models when mode is None.

    When the user opts into "show all" globally the discovery default
    flips to ``"all"``; the catalog screen and the picker's
    "Show all" affordance can also pass ``mode="all"`` explicitly.
    """
    snapshot = cfg.show_all_api_models
    fake_provider = mock.MagicMock()
    fake_provider.list_chat_models.return_value = ["gemini-2.0-flash"]
    try:
        cfg.show_all_api_models = True
        with (
            mock.patch(
                "lilbee.modelhub.model_manager.discovery.PROVIDER_KEYS",
                (("gemini", "gemini_api_key", "GEMINI_API_KEY", "Gemini"),),
            ),
            mock.patch(
                "lilbee.modelhub.model_manager.discovery._has_provider_key",
                return_value=True,
            ),
            mock.patch(
                "lilbee.modelhub.model_manager.discovery.get_services",
                return_value=mock.MagicMock(provider=fake_provider),
            ),
        ):
            from lilbee.modelhub.model_manager.discovery import discover_api_models

            discover_api_models()
            fake_provider.list_chat_models.assert_called_with("gemini", mode="all")
    finally:
        cfg.show_all_api_models = snapshot
