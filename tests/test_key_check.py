"""Hosted-provider API-key checks and every surface that acts on a rejected key."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Iterator
from unittest import mock

import httpx
import pytest

from lilbee.app import services as svc_mod
from lilbee.catalog import CatalogResult
from lilbee.catalog.types import KeyStatus, ModelSource
from lilbee.core.config import cfg
from lilbee.modelhub.model_manager import (
    ValidationResult,
    discover_api_model_groups,
    discover_api_models,
    discovery,
    validate_persisted_model,
)
from lilbee.modelhub.model_manager.discovery import KnownModelCache
from lilbee.modelhub.model_manager.types import RemoteModel
from lilbee.providers import key_check
from lilbee.server import handlers
from lilbee.server.handlers import models as model_handlers
from lilbee.server.models import ModelsCatalogResponse

_GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-pro"]
_OPENAI_MODELS = ["gpt-4o"]
_BAD_KEY = "bad-gemini-key"
_GOOD_KEY = "good-gemini-key"
_PASTED_KEY = "good-gemini\u200bkey"
_LOCAL_REF = "org/repo/model.gguf"
# Captured at import, before the autouse seal replaces the seam.
_REAL_HTTP_GET = key_check._http_get


def _response(status: int, url: str, json: object | None = None) -> httpx.Response:
    return httpx.Response(status, json=json, request=httpx.Request("GET", url))


class _FakeProviders:
    """Stands in for every provider endpoint; records each call."""

    def __init__(self, answer: Callable[[str, dict[str, str]], httpx.Response]) -> None:
        self._answer = answer
        self._lock = threading.Lock()
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, headers: dict[str, str]) -> httpx.Response:
        with self._lock:
            self.calls.append((url, headers))
        return self._answer(url, headers)


def _gemini_rejects_bad_key(url: str, headers: dict[str, str]) -> httpx.Response:
    if headers.get("x-goog-api-key") == _BAD_KEY:
        body = {"error": {"code": 400, "details": [{"reason": "API_KEY_INVALID"}]}}
        return _response(400, url, body)
    return _response(200, url, {"models": []})


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakeProviders) -> _FakeProviders:
    monkeypatch.setattr(key_check, "_http_get", fake.get)
    key_check._checked_key_status.cache_clear()
    return fake


@pytest.fixture
def services(monkeypatch: pytest.MonkeyPatch) -> Iterator[mock.MagicMock]:
    from tests.conftest import make_mock_services

    svc = make_mock_services()
    catalog = {"gemini": _GEMINI_MODELS, "openai": _OPENAI_MODELS}
    svc.provider.list_chat_models.side_effect = lambda prov: catalog.get(prov, [])
    svc_mod.set_services(svc)
    for env in key_check.PROVIDER_API_KEY_ENV.values():
        monkeypatch.delenv(env, raising=False)
    yield svc
    svc_mod.set_services(None)


@pytest.fixture
def catalog_route(services: mock.MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """The catalog route with no native rows and no local servers."""
    monkeypatch.setattr(
        model_handlers,
        "get_catalog",
        lambda *a, **k: CatalogResult(total=None, limit=20, offset=0, models=[]),
    )
    monkeypatch.setattr(model_handlers, "classify_all_remote_models", lambda: [])
    services.registry.list_installed.return_value = []
    model_handlers._hosted_cache.clear()


def _frontier_names(resp: ModelsCatalogResponse) -> dict[str, KeyStatus | None]:
    return {m.display_name: m.key_status for m in resp.models if m.source == ModelSource.FRONTIER}


class TestCatalogRoute:
    async def test_rejected_key_omits_the_provider(self, catalog_route, monkeypatch) -> None:
        _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        cfg.gemini_api_key = _BAD_KEY
        cfg.openai_api_key = "sk-openai"
        resp = await handlers.models_catalog(task="chat")
        assert _frontier_names(resp) == {"gpt-4o": KeyStatus.READY}

    async def test_timeout_lists_the_provider_as_ready(
        self, catalog_route, monkeypatch, caplog
    ) -> None:
        def _offline(url: str, headers: dict[str, str]) -> httpx.Response:
            raise httpx.ConnectTimeout("offline")

        _install(monkeypatch, _FakeProviders(_offline))
        cfg.gemini_api_key = _BAD_KEY
        with caplog.at_level(logging.WARNING, logger=key_check.__name__):
            resp = await handlers.models_catalog(task="chat")
        assert _frontier_names(resp) == dict.fromkeys(_GEMINI_MODELS, KeyStatus.READY)
        assert "Could not verify the gemini API key: ConnectTimeout" in caplog.text
        assert _BAD_KEY not in caplog.text

    async def test_changed_key_checks_again(self, catalog_route, monkeypatch) -> None:
        fake = _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        cfg.gemini_api_key = _BAD_KEY
        first = await handlers.models_catalog(task="chat")
        cfg.gemini_api_key = _GOOD_KEY
        second = await handlers.models_catalog(task="chat")
        assert _frontier_names(first) == {}
        assert set(_frontier_names(second)) == set(_GEMINI_MODELS)
        assert [h["x-goog-api-key"] for _url, h in fake.calls] == [_BAD_KEY, _GOOD_KEY]

    async def test_concurrent_loads_check_the_key_once(self, catalog_route, monkeypatch) -> None:
        def _slow_accept(url: str, headers: dict[str, str]) -> httpx.Response:
            time.sleep(0.3)
            return _response(200, url, {"models": []})

        fake = _install(monkeypatch, _FakeProviders(_slow_accept))
        cfg.gemini_api_key = _GOOD_KEY
        first, second = await asyncio.gather(
            handlers.models_catalog(task="chat"), handlers.models_catalog(task="chat")
        )
        assert set(_frontier_names(first)) == set(_GEMINI_MODELS)
        assert set(_frontier_names(second)) == set(_GEMINI_MODELS)
        assert len(fake.calls) == 1

    async def test_unsendable_key_omits_the_provider_without_a_call(
        self, catalog_route, monkeypatch
    ) -> None:
        fake = _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        cfg.gemini_api_key = _PASTED_KEY
        cfg.openai_api_key = "sk-openai"
        resp = await handlers.models_catalog(task="chat")
        assert _frontier_names(resp) == {"gpt-4o": KeyStatus.READY}
        assert [url for url, _h in fake.calls] == ["https://api.openai.com/v1/models"]

    async def test_unexpected_check_error_lists_ready_and_warns(
        self, catalog_route, monkeypatch, caplog
    ) -> None:
        def _broken(url: str, headers: dict[str, str]) -> httpx.Response:
            raise RuntimeError("unexpected")

        _install(monkeypatch, _FakeProviders(_broken))
        cfg.gemini_api_key = _GOOD_KEY
        with caplog.at_level(logging.WARNING, logger=key_check.__name__):
            resp = await handlers.models_catalog(task="chat")
        assert _frontier_names(resp) == dict.fromkeys(_GEMINI_MODELS, KeyStatus.READY)
        assert "Could not verify the gemini API key: RuntimeError" in caplog.text


@pytest.fixture
def known_models(services: mock.MagicMock, monkeypatch: pytest.MonkeyPatch) -> KnownModelCache:
    """The chat-routing model cache with one local model and no local servers."""
    manifest = mock.MagicMock()
    manifest.ref = _LOCAL_REF
    services.registry.list_installed.return_value = [manifest]
    monkeypatch.setattr(discovery, "classify_all_remote_models", lambda: [])
    return KnownModelCache()


class TestChatRouting:
    def test_unsendable_key_keeps_local_models_routable(self, known_models, monkeypatch) -> None:
        _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        cfg.gemini_api_key = _PASTED_KEY
        assert known_models.resolve(_LOCAL_REF) == _LOCAL_REF
        assert known_models.resolve("gemini/gemini-2.0-flash") is None

    def test_unexpected_check_error_keeps_every_model_routable(
        self, known_models, monkeypatch
    ) -> None:
        def _broken(url: str, headers: dict[str, str]) -> httpx.Response:
            raise RuntimeError("unexpected")

        _install(monkeypatch, _FakeProviders(_broken))
        cfg.gemini_api_key = _GOOD_KEY
        assert known_models.resolve(_LOCAL_REF) == _LOCAL_REF
        assert known_models.resolve("gemini/gemini-2.0-flash") == "gemini/gemini-2.0-flash"


class TestSetModelRoute:
    async def test_rejected_key_ref_is_not_available(self, services, monkeypatch) -> None:
        _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        services.provider.list_models.return_value = []
        cfg.gemini_api_key = _BAD_KEY
        with pytest.raises(ValueError, match="not available"):
            await handlers.set_chat_model("gemini/gemini-2.0-flash")

    async def test_accepted_key_ref_is_selectable(self, services, monkeypatch) -> None:
        _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        services.provider.list_models.return_value = []
        cfg.gemini_api_key = _GOOD_KEY
        result = await handlers.set_chat_model("gemini/gemini-2.0-flash")
        assert result.model == "gemini/gemini-2.0-flash"


class TestPersistedRefValidation:
    def test_rejected_key_is_invalid_key(self, monkeypatch) -> None:
        _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg.gemini_api_key = _BAD_KEY
        assert validate_persisted_model("gemini/gemini-2.0-flash") == ValidationResult.INVALID_KEY

    def test_rejected_key_reason_names_the_provider(self, monkeypatch) -> None:
        from lilbee.modelhub.model_manager.validation import _classify_ref

        _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg.gemini_api_key = _BAD_KEY
        assert _classify_ref("gemini/gemini-2.0-flash") == (
            ValidationResult.INVALID_KEY,
            "gemini rejected the configured API key",
        )

    def test_accepted_key_is_ok(self, monkeypatch) -> None:
        _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg.gemini_api_key = _GOOD_KEY
        assert validate_persisted_model("gemini/gemini-2.0-flash") == ValidationResult.OK

    def test_missing_key_is_no_key(self, monkeypatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg.gemini_api_key = ""
        assert validate_persisted_model("gemini/gemini-2.0-flash") == ValidationResult.NO_KEY


class TestDiscovery:
    def test_groups_carry_the_rejected_status(self, services, monkeypatch) -> None:
        _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        cfg.gemini_api_key = _BAD_KEY
        cfg.openai_api_key = "sk-openai"
        groups = {g.provider: g.key_status for g in discover_api_model_groups()}
        assert groups == {"gemini": KeyStatus.INVALID_KEY, "openai": KeyStatus.READY}
        assert set(discover_api_models()) == {"OpenAI"}

    def test_no_keys_never_touches_the_backend(self, services) -> None:
        assert discover_api_model_groups() == []
        services.provider.list_chat_models.assert_not_called()

    def test_provider_without_models_is_not_checked(self, services, monkeypatch) -> None:
        fake = _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        cfg.mistral_api_key = "mistral-key"
        assert discover_api_model_groups() == []
        assert fake.calls == []


def _gemini_row(status: KeyStatus):
    from lilbee.cli.tui.screens.catalog_utils import frontier_row_from_remote

    rm = RemoteModel(
        name="gemini-2.0-flash", task="chat", family="", parameter_size="", provider="Gemini"
    )
    return frontier_row_from_remote(rm, provider_id="gemini", key_status=status)


class TestTuiCatalog:
    def test_worker_marks_rejected_rows(self, services, monkeypatch) -> None:
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        _install(monkeypatch, _FakeProviders(_gemini_rejects_bad_key))
        cfg.gemini_api_key = _BAD_KEY
        screen = CatalogScreen.__new__(CatalogScreen)
        rows = screen._fetch_frontier_models.__wrapped__(screen)
        assert {(r.name, r.provider_id, r.key_status) for r in rows} == {
            (name, "gemini", KeyStatus.INVALID_KEY) for name in _GEMINI_MODELS
        }

    def test_list_row_shows_the_rejected_label(self) -> None:
        from lilbee.cli.tui.widgets.model_list import _render_frontier

        rendered = _render_frontier(_gemini_row(KeyStatus.INVALID_KEY)).plain
        assert "key rejected" in rendered
        assert "gemini-2.0-flash" in rendered

    def test_card_pill_shows_the_rejected_label(self) -> None:
        from lilbee.cli.tui.widgets.catalog_card_shared import _key_status_pill

        assert _key_status_pill(KeyStatus.INVALID_KEY).plain.strip() == "key rejected"
        assert _key_status_pill(KeyStatus.READY).plain.strip() == "ready"

    def test_selecting_a_rejected_row_does_not_activate_it(self) -> None:
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.app import LilbeeApp
        from lilbee.cli.tui.screens.catalog import CatalogScreen

        screen = CatalogScreen.__new__(CatalogScreen)
        fake_app = mock.MagicMock(spec=LilbeeApp)
        with (
            mock.patch.object(
                CatalogScreen, "app", new_callable=mock.PropertyMock, return_value=fake_app
            ),
            mock.patch.object(CatalogScreen, "notify") as notify,
            mock.patch("lilbee.cli.tui.screens.catalog.apply_active_model") as apply,
        ):
            screen._select_frontier_row(_gemini_row(KeyStatus.INVALID_KEY))
        apply.assert_not_called()
        notify.assert_called_once()
        assert notify.call_args.args[0] == msg.CATALOG_KEY_REJECTED.format(
            provider="Gemini", key_field="gemini_api_key"
        )
        fake_app.switch_view.assert_called_once_with("Settings")


class TestKeyProbe:
    @pytest.mark.parametrize(
        ("provider", "url", "header", "value"),
        [
            ("openrouter", "https://openrouter.ai/api/v1/key", "Authorization", "Bearer k"),
            (
                "gemini",
                "https://generativelanguage.googleapis.com/v1beta/models",
                "x-goog-api-key",
                "k",
            ),
            ("anthropic", "https://api.anthropic.com/v1/models", "x-api-key", "k"),
            ("openai", "https://api.openai.com/v1/models", "Authorization", "Bearer k"),
            ("mistral", "https://api.mistral.ai/v1/models", "Authorization", "Bearer k"),
            ("deepseek", "https://api.deepseek.com/models", "Authorization", "Bearer k"),
        ],
    )
    def test_each_provider_sends_its_key(self, monkeypatch, provider, url, header, value) -> None:
        fake = _install(monkeypatch, _FakeProviders(lambda u, h: _response(200, u)))
        assert key_check._checked_key_status(provider, "k") is KeyStatus.READY
        assert fake.calls == [(url, mock.ANY)]
        assert fake.calls[0][1][header] == value

    def test_every_hosted_provider_has_a_probe(self) -> None:
        assert set(key_check._PROBES) == set(key_check.PROVIDER_API_KEY_ENV)

    def test_anthropic_sends_the_api_version(self, monkeypatch) -> None:
        fake = _install(monkeypatch, _FakeProviders(lambda u, h: _response(200, u)))
        key_check._checked_key_status("anthropic", "k")
        assert fake.calls[0][1]["anthropic-version"] == "2023-06-01"

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_rejection_is_invalid_key(self, monkeypatch, status) -> None:
        _install(monkeypatch, _FakeProviders(lambda u, h: _response(status, u)))
        assert key_check._checked_key_status("openai", "k") is KeyStatus.INVALID_KEY

    @pytest.mark.parametrize("status", [400, 404, 429, 500, 503])
    def test_other_errors_stay_ready_with_a_warning(self, monkeypatch, caplog, status) -> None:
        _install(monkeypatch, _FakeProviders(lambda u, h: _response(status, u)))
        with caplog.at_level(logging.WARNING, logger=key_check.__name__):
            assert key_check._checked_key_status("openai", "k") is KeyStatus.READY
        assert f"Could not verify the openai API key: HTTP {status}" in caplog.text

    def test_success_logs_nothing(self, monkeypatch, caplog) -> None:
        _install(monkeypatch, _FakeProviders(lambda u, h: _response(200, u)))
        with caplog.at_level(logging.WARNING, logger=key_check.__name__):
            assert key_check._checked_key_status("openai", "k") is KeyStatus.READY
        assert caplog.text == ""

    @pytest.mark.parametrize(
        "api_key", ["key\u200b", "caf\u00e9-key", "key\nX-Injected: 1", "key\t"]
    )
    def test_unsendable_key_is_invalid_without_a_call(self, monkeypatch, api_key) -> None:
        fake = _install(monkeypatch, _FakeProviders(lambda u, h: _response(200, u)))
        assert key_check._checked_key_status("openai", api_key) is KeyStatus.INVALID_KEY
        assert fake.calls == []

    def test_connection_error_stays_ready(self, monkeypatch) -> None:
        def _refused(url: str, headers: dict[str, str]) -> httpx.Response:
            raise httpx.ConnectError("refused")

        _install(monkeypatch, _FakeProviders(_refused))
        assert key_check._checked_key_status("openai", "k") is KeyStatus.READY

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ({"error": {"details": [{"reason": "API_KEY_INVALID"}]}}, KeyStatus.INVALID_KEY),
            ({"error": {"details": ["text", {"reason": "OTHER"}]}}, KeyStatus.READY),
            ({"error": {"message": "bad request"}}, KeyStatus.READY),
            ({"error": None}, KeyStatus.READY),
            ({"error": {"details": None}}, KeyStatus.READY),
            (None, KeyStatus.READY),
        ],
    )
    def test_gemini_bad_request_needs_the_invalid_key_reason(
        self, monkeypatch, body, expected
    ) -> None:
        def _answer(url: str, headers: dict[str, str]) -> httpx.Response:
            if body is None:
                return httpx.Response(400, content=b"not json", request=httpx.Request("GET", url))
            return _response(400, url, body)

        _install(monkeypatch, _FakeProviders(_answer))
        assert key_check._checked_key_status("gemini", "k") is expected

    def test_gemini_null_details_is_not_a_rejection(self) -> None:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        resp = _response(400, url, {"error": {"details": None}})
        assert key_check._gemini_rejected(resp) is False

    def test_gemini_forbidden_is_invalid_key(self, monkeypatch) -> None:
        _install(monkeypatch, _FakeProviders(lambda u, h: _response(403, u)))
        assert key_check._checked_key_status("gemini", "k") is KeyStatus.INVALID_KEY

    def test_verdict_is_cached_per_key(self, monkeypatch) -> None:
        fake = _install(monkeypatch, _FakeProviders(lambda u, h: _response(401, u)))
        for _ in range(3):
            key_check._checked_key_status("openai", "k")
        key_check._checked_key_status("openai", "k2")
        assert len(fake.calls) == 2


class TestKeyInUse:
    def test_env_var_wins_over_config(self, monkeypatch) -> None:
        fake = _install(monkeypatch, _FakeProviders(lambda u, h: _response(200, u)))
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        cfg.openai_api_key = "from-config"
        assert key_check.provider_key_status("openai") is KeyStatus.READY
        assert fake.calls[0][1]["Authorization"] == "Bearer from-env"

    def test_config_key_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg.openai_api_key = "from-config"
        assert key_check.provider_key_set("openai") is True

    def test_no_key_is_missing_without_a_call(self, monkeypatch) -> None:
        fake = _install(monkeypatch, _FakeProviders(lambda u, h: _response(200, u)))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg.openai_api_key = ""
        assert key_check.provider_key_set("openai") is False
        assert key_check.provider_key_status("openai") is KeyStatus.MISSING_KEY
        assert fake.calls == []

    def test_statuses_for_no_providers_is_empty(self) -> None:
        assert key_check.provider_key_statuses([]) == {}


def test_probe_get_sends_headers_with_the_check_timeout() -> None:
    url = "https://api.openai.com/v1/models"
    with mock.patch("lilbee.providers.key_check.httpx.get") as get:
        get.return_value = _response(200, url)
        resp = _REAL_HTTP_GET(url, headers={"Authorization": "Bearer k"})
    assert resp.status_code == 200
    get.assert_called_once_with(
        url, headers={"Authorization": "Bearer k"}, timeout=key_check.KEY_CHECK_TIMEOUT_S
    )
