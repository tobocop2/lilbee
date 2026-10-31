"""A model ref that names a stored server preset: parsing, role and address checks, the wire."""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import pytest

from lilbee.app import presets
from lilbee.core import blocked_networks, settings
from lilbee.core.config import Config, cfg
from lilbee.modelhub.role_validator import _is_registry_ref
from lilbee.providers.backends.liter_llm.backend import LiterLlmBackend
from lilbee.providers.backends.liter_llm.config import _route_model, rerank_route
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.model_ref import ProviderModelRef, parse_model_ref, translate_options
from lilbee.providers.roles import WorkerRole
from lilbee.providers.routing_provider import _is_native_rerank_ref
from lilbee.providers.sdk_backend import (
    CompletionRequest,
    CompletionResult,
    EmbeddingRequest,
    EmbeddingResult,
    RerankRequest,
    RerankResult,
)
from lilbee.providers.sdk_llm_provider import SdkLLMProvider
from lilbee.providers.server_registry import (
    BUILTIN_PREFIXES,
    RerankWire,
    check_role,
    checked_key,
    decode_entries,
    route_for,
)
from tests import _fake_sdk as fakes

PRESET_KEY = "SENTINEL-preset-key-77aa"
LLM_KEY = "SENTINEL-llm-api-key-31bc"
PUBLIC_HOST = "pub.example.test"
URL = f"http://{PUBLIC_HOST}:8080/v1"
ALL_ROLES = ["chat", "embed", "rerank", "vision"]


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Any) -> Iterator[None]:
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.server_presets = None
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture
def dns(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """The DNS the request check sees: a host maps to its answers, any other host has none."""
    table = {PUBLIC_HOST: ["93.184.216.34"]}

    def getaddrinfo(host: str, _port: Any) -> list[Any]:
        if host not in table:
            raise socket.gaierror("no answer")
        return [(2, 1, 6, "", (address, 0)) for address in table[host]]

    monkeypatch.setattr(blocked_networks.socket, "getaddrinfo", getaddrinfo)
    return table


def _store(*entries: dict[str, Any]) -> None:
    settings.set_value(cfg.data_root, "server_presets", json.dumps(list(entries)))


def _preset(
    name: str = "wired", url: str = URL, roles: list[str] | None = None, **extra: Any
) -> dict[str, Any]:
    return {"name": name, "url": url, "roles": roles or ALL_ROLES, **extra}


@dataclass
class _Backend:
    """An ``LlmSdkBackend`` that records what the provider hands it."""

    complete_calls: list[CompletionRequest] = field(default_factory=list)
    embed_calls: list[EmbeddingRequest] = field(default_factory=list)
    rerank_calls: list[RerankRequest] = field(default_factory=list)
    provider_name: str = "fake"

    def available(self) -> bool:
        return True

    def configure_logging(self, *, suppress_debug: bool) -> None:
        return None

    def complete(self, request: CompletionRequest) -> CompletionResult:
        self.complete_calls.append(request)
        return CompletionResult(content="hello")

    def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.embed_calls.append(request)
        return EmbeddingResult(vectors=[[0.1, 0.2]])

    def rerank(self, request: RerankRequest) -> RerankResult:
        self.rerank_calls.append(request)
        return RerankResult(scores=[0.5] * len(request.candidates))

    @property
    def requests(self) -> int:
        return len(self.complete_calls) + len(self.embed_calls) + len(self.rerank_calls)


@pytest.fixture
def backend() -> _Backend:
    return _Backend()


@pytest.fixture
def provider(backend: _Backend) -> SdkLLMProvider:
    return SdkLLMProvider(backend)  # type: ignore[arg-type]


class TestRouteFor:
    def test_a_stored_preset_gives_its_route(self) -> None:
        _store(_preset(api_key=PRESET_KEY, rerank_wire="tei", roles=["rerank", "chat"]))
        route = route_for("wired")
        assert route is not None
        assert (route.name, route.base_url, route.api_key) == ("wired", URL, PRESET_KEY)
        assert route.roles == (WorkerRole.RERANK, WorkerRole.CHAT)
        assert route.rerank_wire is RerankWire.TEI

    def test_a_keyless_preset_has_no_key_and_the_v1_wire(self) -> None:
        _store(_preset())
        route = route_for("wired")
        assert route is not None
        assert route.api_key is None
        assert route.rerank_wire is RerankWire.V1

    def test_the_route_picks_the_named_preset_among_several(self) -> None:
        _store(_preset("first", "http://one.test/v1"), _preset("second", "http://two.test/v1"))
        route = route_for("second")
        assert route is not None
        assert route.base_url == "http://two.test/v1"

    def test_no_preset_of_that_name_gives_none(self) -> None:
        _store(_preset("other"))
        assert route_for("wired") is None

    def test_a_built_in_prefix_pays_no_store_read(self) -> None:
        _store(_preset("openai"))
        with mock.patch("lilbee.providers.server_registry.stored_entries") as read:
            assert route_for("openai") is None
        read.assert_not_called()

    def test_an_unreadable_row_is_skipped_and_the_rest_routes(self) -> None:
        _store({"name": "wired", "url": URL}, _preset("good"))
        assert route_for("wired") is None
        assert route_for("good") is not None

    def test_a_hand_edited_preset_named_local_does_not_take_a_local_ref(self) -> None:
        _store(_preset("local"))
        assert route_for("local") is None
        assert parse_model_ref("local/x").name == "local/x"

    def test_a_store_that_does_not_parse_routes_nothing(self) -> None:
        settings.set_value(cfg.data_root, "server_presets", "not json")
        assert route_for("wired") is None

    def test_a_preset_another_process_writes_is_seen_at_once(self) -> None:
        assert route_for("wired") is None
        _store(_preset())
        assert route_for("wired") is not None
        _store()
        assert route_for("wired") is None


class TestStoredValue:
    def test_no_value_holds_no_entries(self) -> None:
        assert decode_entries(None) == []

    @pytest.mark.parametrize("raw", ["not json", '{"a": 1}', 7, "[" * 100000])
    def test_a_value_that_is_no_list_is_refused(self, raw: Any) -> None:
        with pytest.raises(ValueError, match="do not parse"):
            decode_entries(raw)

    def test_a_key_with_a_character_that_cannot_be_stored_is_refused_without_the_key(self) -> None:
        with pytest.raises(ValueError) as err:
            checked_key("bad\ud800key")
        assert "bad" not in str(err.value)


class TestCheckRole:
    def test_a_served_role_gives_the_route(self) -> None:
        _store(_preset(roles=["embed"]))
        assert check_role(parse_model_ref("wired/m"), WorkerRole.EMBED).name == "wired"

    @pytest.mark.parametrize("role", list(WorkerRole))
    def test_a_role_the_preset_does_not_serve_is_refused_and_names_both(
        self, role: WorkerRole
    ) -> None:
        served = next(r for r in WorkerRole if r is not role)
        _store(_preset(roles=[served.value]))
        refusal = f"'wired' does not serve the {role.value} role"
        with pytest.raises(ProviderError, match=refusal) as e:
            check_role(parse_model_ref("wired/m"), role)
        assert e.value.kind is ProviderErrorKind.BAD_REQUEST

    def test_a_ref_with_no_preset_is_refused(self) -> None:
        _store(_preset("other"))
        ref = ProviderModelRef(raw="wired/m", provider="wired", name="m")
        with pytest.raises(ProviderError, match="No server preset is named 'wired'") as e:
            check_role(ref, WorkerRole.CHAT)
        assert e.value.kind is ProviderErrorKind.NOT_FOUND


class TestParse:
    def test_a_preset_ref_parses_to_the_preset(self) -> None:
        _store(_preset())
        ref = parse_model_ref("wired/org/model:tag")
        assert (ref.provider, ref.name) == ("wired", "org/model:tag")
        assert ref.is_remote and ref.is_preset and not ref.is_api and not ref.is_local
        assert ref.for_openai_prefix() == "wired/org/model:tag"

    def test_the_same_ref_with_no_preset_is_local(self) -> None:
        ref = parse_model_ref("wired/m")
        assert (ref.provider, ref.name) == ("local", "wired/m")
        assert not ref.is_preset

    def test_a_removed_preset_leaves_the_next_parse_local(self) -> None:
        _store(_preset())
        assert parse_model_ref("wired/m").is_preset
        _store()
        assert not parse_model_ref("wired/m").is_preset

    def test_the_native_gguf_shape_wins_over_a_preset(self) -> None:
        _store(_preset("org"))
        ref = parse_model_ref("org/repo/file.gguf")
        assert ref.is_local
        assert parse_model_ref("org/repo").is_preset

    def test_a_built_in_prefix_wins_over_a_preset_of_that_name(self) -> None:
        _store(_preset("openai"))
        ref = parse_model_ref("openai/gpt")
        assert ref.is_api and not ref.is_preset

    def test_a_ref_with_no_slash_names_the_presets_in_its_error(self) -> None:
        with pytest.raises(ValueError, match="name of a server preset"):
            parse_model_ref("noslash")

    def test_a_preset_ref_keeps_its_raw_options(self) -> None:
        _store(_preset())
        options = {"top_k": 5, "temperature": 0.2}
        assert translate_options(options, parse_model_ref("wired/m")) == options

    def test_every_built_in_prefix_is_still_known(self) -> None:
        assert {"openai", "anthropic", "ollama", "lm_studio"} <= BUILTIN_PREFIXES


class TestRegistryRef:
    def test_a_preset_ref_is_no_registry_ref(self) -> None:
        _store(_preset())
        assert _is_registry_ref("wired/m") is False

    def test_the_same_ref_with_no_preset_is_a_registry_ref(self) -> None:
        assert _is_registry_ref("wired/m") is True

    def test_a_bare_name_that_equals_a_preset_is_a_registry_ref(self) -> None:
        _store(_preset())
        assert _is_registry_ref("wired") is True

    def test_a_bare_repo_under_a_preset_name_is_not_a_native_rerank_ref(self) -> None:
        services = mock.MagicMock()
        services.registry.installed_ref_for_repo.return_value = "org/repo/q.gguf"
        with mock.patch("lilbee.providers.routing_provider.get_services", return_value=services):
            assert _is_native_rerank_ref("wired/repo") is True
            _store(_preset())
            assert _is_native_rerank_ref("wired/repo") is False


class TestSdkProviderRouting:
    def test_chat_goes_to_the_preset_with_its_own_key(
        self, provider: SdkLLMProvider, backend: _Backend, dns: dict[str, list[str]]
    ) -> None:
        _store(_preset(api_key=PRESET_KEY))
        cfg.llm_api_key = LLM_KEY
        provider.chat([{"role": "user", "content": "hi"}], model="wired/m")
        request = backend.complete_calls[0]
        assert (request.api_base, request.api_key) == (URL, PRESET_KEY)
        assert request.ref.name == "m"

    def test_a_keyless_preset_never_sends_the_llm_api_key(
        self, provider: SdkLLMProvider, backend: _Backend, dns: dict[str, list[str]]
    ) -> None:
        _store(_preset())
        cfg.llm_api_key = LLM_KEY
        provider.chat([{"role": "user", "content": "hi"}], model="wired/m")
        assert backend.complete_calls[0].api_key is None

    def test_a_local_server_ref_still_sends_the_llm_api_key(
        self, provider: SdkLLMProvider, backend: _Backend
    ) -> None:
        cfg.llm_api_key = LLM_KEY
        provider.chat([{"role": "user", "content": "hi"}], model="lm_studio/m")
        assert backend.complete_calls[0].api_key == LLM_KEY

    def test_embed_goes_to_the_preset(
        self, provider: SdkLLMProvider, backend: _Backend, dns: dict[str, list[str]]
    ) -> None:
        _store(_preset(api_key=PRESET_KEY))
        cfg.embedding_model = "wired/e"
        provider.embed(["a"])
        request = backend.embed_calls[0]
        assert (request.api_base, request.api_key, request.ref.name) == (URL, PRESET_KEY, "e")

    @pytest.mark.parametrize("wire", list(RerankWire))
    def test_rerank_carries_the_wire_of_the_preset(
        self, provider: SdkLLMProvider, backend: _Backend, dns: dict[str, list[str]], wire: Any
    ) -> None:
        _store(_preset(rerank_wire=wire.value))
        cfg.reranker_model = "wired/r"
        provider.rerank("q", ["a", "b"])
        request = backend.rerank_calls[0]
        assert (request.api_base, request.rerank_wire) == (URL, wire)

    def test_rerank_of_a_hosted_ref_uses_the_v1_wire_and_no_base(
        self, provider: SdkLLMProvider, backend: _Backend
    ) -> None:
        cfg.reranker_model = "openai/rerank-x"
        provider.rerank("q", ["a"])
        request = backend.rerank_calls[0]
        assert (request.api_base, request.rerank_wire) == (None, RerankWire.V1)

    def test_vision_ocr_needs_the_vision_role_not_the_chat_role(
        self, provider: SdkLLMProvider, backend: _Backend, dns: dict[str, list[str]]
    ) -> None:
        _store(_preset(roles=["chat"]))
        with pytest.raises(ProviderError, match="does not serve the vision role"):
            provider.vision_ocr(b"png", "wired/v")
        assert backend.requests == 0

    def test_vision_ocr_reaches_a_preset_that_serves_vision(
        self, provider: SdkLLMProvider, backend: _Backend, dns: dict[str, list[str]]
    ) -> None:
        _store(_preset(roles=["vision"]))
        assert provider.vision_ocr(b"png", "wired/v") == "hello"
        assert backend.complete_calls[0].api_base == URL

    def test_vision_ocr_with_a_timeout_checks_the_vision_role_too(
        self, provider: SdkLLMProvider, backend: _Backend, dns: dict[str, list[str]]
    ) -> None:
        _store(_preset(roles=["chat"]))
        with pytest.raises(ProviderError, match="does not serve the vision role"):
            provider.vision_ocr(b"png", "wired/v", timeout=5)
        assert backend.requests == 0

    @pytest.mark.parametrize(
        "role, call",
        [
            (
                WorkerRole.CHAT,
                lambda p: p.chat([{"role": "user", "content": "x"}], model="wired/m"),
            ),
            (WorkerRole.EMBED, lambda p: p.embed(["x"])),
            (WorkerRole.RERANK, lambda p: p.rerank("q", ["x"])),
        ],
    )
    def test_a_role_the_preset_does_not_serve_sends_no_request(
        self,
        provider: SdkLLMProvider,
        backend: _Backend,
        dns: dict[str, list[str]],
        role: WorkerRole,
        call: Any,
    ) -> None:
        _store(_preset(roles=[r for r in ALL_ROLES if r != role.value]))
        cfg.embedding_model = "wired/e"
        cfg.reranker_model = "wired/r"
        with pytest.raises(ProviderError, match=f"does not serve the {role.value} role"):
            call(provider)
        assert backend.requests == 0

    @pytest.mark.parametrize("address", ["169.254.169.254", "0.0.0.0", "224.0.0.1", "240.0.0.1"])
    def test_a_host_that_turned_blocked_since_the_save_sends_no_request_and_no_key(
        self,
        provider: SdkLLMProvider,
        backend: _Backend,
        dns: dict[str, list[str]],
        address: str,
    ) -> None:
        _store(_preset(api_key=PRESET_KEY))
        dns[PUBLIC_HOST] = [address]
        with pytest.raises(ProviderError, match=f"blocked address {address}") as err:
            provider.chat([{"role": "user", "content": "x"}], model="wired/m")
        assert err.value.kind is ProviderErrorKind.BAD_REQUEST
        assert PRESET_KEY not in str(err.value)
        assert backend.requests == 0

    @pytest.mark.parametrize(
        "call",
        [
            lambda p: p.embed(["x"]),
            lambda p: p.rerank("q", ["x"]),
            lambda p: p.vision_ocr(b"png", "wired/v"),
        ],
        ids=["embed", "rerank", "vision"],
    )
    def test_a_blocked_host_sends_no_request_and_no_key_on_every_other_role(
        self,
        provider: SdkLLMProvider,
        backend: _Backend,
        dns: dict[str, list[str]],
        call: Any,
    ) -> None:
        _store(_preset(api_key=PRESET_KEY))
        cfg.embedding_model = "wired/e"
        cfg.reranker_model = "wired/r"
        dns[PUBLIC_HOST] = ["169.254.169.254"]
        with pytest.raises(ProviderError, match=r"blocked address 169\.254\.169\.254") as err:
            call(provider)
        assert PRESET_KEY not in str(err.value)
        assert backend.requests == 0

    def test_a_host_with_no_dns_answer_sends_no_request(
        self, provider: SdkLLMProvider, backend: _Backend, dns: dict[str, list[str]]
    ) -> None:
        _store(_preset())
        dns.clear()
        with pytest.raises(ProviderError, match="no DNS answer"):
            provider.chat([{"role": "user", "content": "x"}], model="wired/m")
        assert backend.requests == 0

    def test_a_ref_whose_preset_was_removed_before_the_call_is_refused(
        self, provider: SdkLLMProvider, backend: _Backend
    ) -> None:
        _store(_preset())
        ref = parse_model_ref("wired/m")
        _store()
        with (
            mock.patch("lilbee.providers.sdk_llm_provider.parse_model_ref", return_value=ref),
            pytest.raises(ProviderError, match="No server preset is named 'wired'"),
        ):
            provider.chat([{"role": "user", "content": "x"}], model="wired/m")
        assert backend.requests == 0

    def test_two_presets_with_one_model_name_each_reach_their_own_server(
        self, provider: SdkLLMProvider, backend: _Backend, dns: dict[str, list[str]]
    ) -> None:
        dns["two.test"] = ["93.184.216.35"]
        _store(
            _preset("one", URL, api_key="key-one"),  # pragma: allowlist secret
            _preset("two", "http://two.test/v1", api_key="key-two"),  # pragma: allowlist secret
        )
        for name in ("one", "two"):
            provider.chat([{"role": "user", "content": "x"}], model=f"{name}/same")
        assert [(c.api_base, c.api_key) for c in backend.complete_calls] == [
            (URL, "key-one"),
            ("http://two.test/v1", "key-two"),
        ]


class TestWireRoute:
    @pytest.mark.parametrize("base", [URL, "http://h:8080", "http://h:8080/v1/"])
    def test_a_preset_chat_ref_goes_out_as_an_openai_model(self, base: str) -> None:
        _store(_preset())
        assert _route_model(parse_model_ref("wired/org/m"), base) == "openai/org/m"

    @pytest.mark.parametrize(
        "base, wire, model, sent_base",
        [
            ("http://h:8080/v1", RerankWire.V1, "openai/r", "http://h:8080/v1"),
            ("http://h:8080", RerankWire.V1, "openai/r", "http://h:8080/v1"),
            ("http://h:8080/v1/", RerankWire.TEI, "huggingface/r", "http://h:8080"),
            ("http://h:8080", RerankWire.TEI, "huggingface/r", "http://h:8080"),
            (None, RerankWire.V1, "openai/r", None),
            (None, RerankWire.TEI, "huggingface/r", None),
        ],
    )
    def test_the_wire_picks_the_model_prefix_and_the_base(
        self, base: str | None, wire: RerankWire, model: str, sent_base: str | None
    ) -> None:
        _store(_preset())
        assert rerank_route(parse_model_ref("wired/r"), base, wire) == (model, sent_base)

    def test_a_ref_that_is_no_preset_keeps_its_own_route(self) -> None:
        ref = parse_model_ref("lm_studio/r")
        assert rerank_route(ref, "http://h:1234", RerankWire.TEI) == (
            "lm_studio/r",
            "http://h:1234/v1",
        )

    @pytest.fixture
    def sdk(self) -> Iterator[fakes.FakeSdk]:
        with fakes.installed(fakes.FakeSdk()) as fake:
            yield fake

    @pytest.mark.parametrize(
        "wire, model, base",
        [
            (RerankWire.V1, "openai/bge", "http://h:8080/v1"),
            (RerankWire.TEI, "huggingface/bge", "http://h:8080"),
        ],
    )
    def test_the_sdk_gets_one_model_for_the_client_and_the_request(
        self, sdk: fakes.FakeSdk, wire: RerankWire, model: str, base: str
    ) -> None:
        _store(_preset())
        sdk.rerank_result = fakes.rerank_response([(1, 0.9), (0, 0.1)])
        result = LiterLlmBackend().rerank(
            RerankRequest(
                ref=parse_model_ref("wired/bge"),
                query="q",
                candidates=["a", "b"],
                api_base="http://h:8080/v1",
                rerank_wire=wire,
            )
        )
        assert result.scores == [0.1, 0.9]
        assert sdk.last("rerank")["model"] == model
        assert sdk.clients[-1]["model_hint"] == model
        assert sdk.clients[-1]["base_url"] == base


class TestPresetWireField:
    def test_an_unknown_wire_is_refused_and_named_on_add(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown rerank wire 'grpc'\. Valid wires: v1, tei"):
            presets.add_preset("wired", URL, roles=["rerank"], rerank_wire="grpc")

    def test_an_unknown_wire_is_refused_on_edit(self, dns: dict[str, list[str]]) -> None:
        presets.add_preset("wired", URL, roles=["rerank"])
        with pytest.raises(ValueError, match="Unknown rerank wire 'grpc'"):
            presets.edit_preset("wired", rerank_wire="grpc")

    def test_the_tei_wire_is_stored_and_the_default_is_left_out(
        self, dns: dict[str, list[str]]
    ) -> None:
        presets.add_preset("tei", URL, roles=["rerank"], rerank_wire="tei")
        presets.add_preset("plain", URL, roles=["rerank"])
        stored = json.loads(settings.load(cfg.data_root)["server_presets"])
        rows = {row["name"]: row for row in stored}
        assert rows["tei"]["rerank_wire"] == "tei"
        assert "rerank_wire" not in rows["plain"]

    def test_an_edit_changes_the_wire_and_keeps_the_roles(self, dns: dict[str, list[str]]) -> None:
        presets.add_preset("wired", URL, roles=["rerank", "chat"])
        presets.edit_preset("wired", rerank_wire="tei")
        route = route_for("wired")
        assert route is not None
        assert route.rerank_wire is RerankWire.TEI
        assert route.roles == (WorkerRole.RERANK, WorkerRole.CHAT)
