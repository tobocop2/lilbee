"""The API key a hosted chat request carries, read at the client."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any
from unittest import mock

import httpx
import pytest

from lilbee.app.settings import apply_settings_update
from lilbee.catalog.types import KeyStatus
from lilbee.core.config import cfg
from lilbee.providers import key_check
from lilbee.providers.base import ProviderError
from lilbee.providers.routing_provider import RoutingProvider
from lilbee.providers.sdk_backend import PROVIDER_API_KEY_ENV
from tests import _fake_sdk as fakes

GEMINI_MODEL = "gemini/gemini-2.0-flash"
OPENAI_MODEL = "openai/gpt-4o-mini"
LM_STUDIO_MODEL = "lm_studio/local-model"


class _Wire:
    """Answers each chat like a provider and reports the key the client carried."""

    def __init__(self, fake: fakes.FakeSdk) -> None:
        self.fake = fake
        self.fake.chat_result = fakes.text_response("hi")

    @property
    def requests(self) -> list[dict[str, Any]]:
        return [fields for name, fields in self.fake.calls if name == "chat"]

    def last_key(self) -> str | None:
        return self.fake.clients[-1]["api_key"]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch) -> Iterator[None]:
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    for env_var in PROVIDER_API_KEY_ENV.values():
        monkeypatch.delenv(env_var, raising=False)
    yield
    for field_name in type(snapshot).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))


@pytest.fixture
def wire() -> Iterator[_Wire]:
    with fakes.installed(fakes.FakeSdk(require_key=True)) as fake:
        yield _Wire(fake)


def _chat(provider: RoutingProvider, model: str) -> None:
    provider.chat([{"role": "user", "content": "q"}], model=model)


@pytest.mark.parametrize("model", [GEMINI_MODEL, OPENAI_MODEL])
class TestWhichKeyIsSent:
    def test_provider_key_alone_is_sent(self, wire: _Wire, model: str) -> None:
        field = f"{model.split('/')[0]}_api_key"
        setattr(cfg, field, "provider-key")
        _chat(RoutingProvider(), model)
        assert wire.last_key() == "provider-key"

    def test_llm_api_key_alone_is_not_sent(self, wire: _Wire, model: str) -> None:
        provider = model.split("/")[0]
        cfg.llm_api_key = "generic-key"
        with pytest.raises(ProviderError):
            _chat(RoutingProvider(), model)
        assert not wire.requests
        assert key_check.provider_key_status(provider) is KeyStatus.MISSING_KEY

    def test_provider_key_wins_over_llm_api_key(self, wire: _Wire, model: str) -> None:
        provider = model.split("/")[0]
        setattr(cfg, f"{provider}_api_key", "provider-key")
        cfg.llm_api_key = "generic-key"
        _chat(RoutingProvider(), model)
        assert wire.last_key() == "provider-key"
        assert key_check.provider_api_key_in_use(provider) == "provider-key"

    def test_shell_env_var_wins_over_config(
        self, wire: _Wire, model: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = model.split("/")[0]
        monkeypatch.setenv(PROVIDER_API_KEY_ENV[provider], "shell-key")
        setattr(cfg, f"{provider}_api_key", "config-key")
        _chat(RoutingProvider(), model)
        assert wire.last_key() == "shell-key"
        assert key_check.provider_api_key_in_use(provider) == "shell-key"


class TestKeyChangedInSettings:
    def test_changed_key_is_checked_and_sent_without_a_restart(self, wire: _Wire) -> None:
        provider = RoutingProvider()
        checked: list[str] = []

        def _record(url: str, *, headers: dict[str, str]) -> httpx.Response:
            checked.append(headers["x-goog-api-key"])
            return httpx.Response(200, request=httpx.Request("GET", url))

        with mock.patch.object(key_check, "_http_get", _record):
            apply_settings_update({"gemini_api_key": "old-key"})
            assert key_check.provider_key_status("gemini") is KeyStatus.READY
            _chat(provider, GEMINI_MODEL)
            assert wire.last_key() == "old-key"

            apply_settings_update({"gemini_api_key": "new-key"})
            assert key_check.provider_key_status("gemini") is KeyStatus.READY
            _chat(provider, GEMINI_MODEL)

        assert checked == ["old-key", "new-key"]
        assert wire.last_key() == "new-key"

    def test_cleared_key_is_no_longer_sent(self, wire: _Wire) -> None:
        provider = RoutingProvider()
        apply_settings_update({"gemini_api_key": "old-key"})
        _chat(provider, GEMINI_MODEL)
        assert wire.last_key() == "old-key"

        apply_settings_update({"gemini_api_key": ""})
        sent_before = len(wire.requests)
        with pytest.raises(ProviderError):
            _chat(provider, GEMINI_MODEL)

        assert len(wire.requests) == sent_before
        assert key_check.provider_key_status("gemini") is KeyStatus.MISSING_KEY
        assert "GEMINI_API_KEY" not in os.environ

    def test_shell_env_var_survives_a_settings_change(
        self, wire: _Wire, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "shell-key")
        provider = RoutingProvider()
        apply_settings_update({"gemini_api_key": "config-key"})
        _chat(provider, GEMINI_MODEL)
        assert wire.last_key() == "shell-key"
        assert os.environ["GEMINI_API_KEY"] == "shell-key"

    def test_changed_llm_api_key_is_sent_without_a_restart(self, wire: _Wire) -> None:
        provider = RoutingProvider()
        apply_settings_update({"llm_api_key": "old-generic"})
        _chat(provider, LM_STUDIO_MODEL)
        assert wire.last_key() == "old-generic"
        apply_settings_update({"llm_api_key": "new-generic"})
        _chat(provider, LM_STUDIO_MODEL)
        assert wire.last_key() == "new-generic"
