"""The catalog lists the rows that an engine publishes, beside the picks and the Hub rows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from conftest import make_hf_client, stub_hf_listing
from lilbee.app.services import get_services
from lilbee.catalog import query as _query
from lilbee.catalog import runnable
from lilbee.catalog.hf_client import HfClient
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.query import get_catalog
from lilbee.catalog.runnable import EngineState, default_offers, hidden_summary
from lilbee.catalog.types import CatalogSize, CatalogSort, ModelFormat, ModelTask
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engines import lemonade
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.engines.lemonade_models import parse_models

PINNED_PATH = Path(__file__).parent / "fixtures" / "engine_model_lists" / "lemonade-pinned.json"
PINNED = json.loads(PINNED_PATH.read_text(encoding="utf-8"))
XDNA2 = host_caps.collect(
    system="Linux", machine="x86_64", cpu_info={}, nvidia=None, accel_devices=[("0x1022", "0x17f0")]
)
NO_NPU = host_caps.collect(
    system="Linux", machine="x86_64", cpu_info={}, nvidia=None, accel_devices=[]
)
ROWS = parse_models(PINNED)
"""The rows of the pinned list: 41 chat and 1 embedding."""


def _hub_row(repo: str, downloads: int = 10) -> CatalogModel:
    return CatalogModel(
        hf_repo=repo,
        gguf_filename="*.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=downloads,
        task=ModelTask.CHAT,
    )


def _pick(repo: str) -> CatalogModel:
    return CatalogModel(
        hf_repo=repo,
        gguf_filename="*.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=True,
        downloads=99,
        task=ModelTask.CHAT,
    )


@pytest.fixture
def listing(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """A browse on an XDNA2 host with the pinned list and two Hub rows; returns the Hub calls."""
    monkeypatch.setattr(_query, "get_picks", lambda **_: [])
    monkeypatch.setattr(runnable, "session_host", lambda: XDNA2)
    calls = stub_hf_listing(
        monkeypatch,
        {"text-generation": [_hub_row("hub/one", 5), _hub_row("hub/two", 4)]},
    )
    monkeypatch.setattr(get_services().hf_client, "fetch_engine_models", lambda engine: ROWS)
    return calls


def _refs(result: Any) -> list[str]:
    return [m.pull_ref for m in result.models]


def test_npu_rows_follow_the_picks_and_lead_the_hub_rows(
    monkeypatch: pytest.MonkeyPatch, listing: list[dict[str, object]]
) -> None:
    monkeypatch.setattr(_query, "get_picks", lambda **_: [_pick("pick/one")])
    chat = [m.pull_ref for m in ROWS if m.task is ModelTask.CHAT]
    result = get_catalog(task=ModelTask.CHAT, limit=100, hf_client=get_services().hf_client)
    assert _refs(result) == ["pick/one", *chat, "hub/one", "hub/two"]


def test_an_npu_row_lists_once_per_task_tab(listing: list[dict[str, object]]) -> None:
    embedding = get_catalog(task=ModelTask.EMBEDDING, limit=100, hf_client=get_services().hf_client)
    assert _refs(embedding) == ["embed-gemma:300m#xdna"]
    for task in (ModelTask.VISION, ModelTask.RERANK):
        assert _refs(get_catalog(task=task, limit=100, hf_client=get_services().hf_client)) == []


def test_paging_counts_the_npu_rows(listing: list[dict[str, object]]) -> None:
    chat = [m.pull_ref for m in ROWS if m.task is ModelTask.CHAT]
    first = get_catalog(task=ModelTask.CHAT, limit=5, hf_client=get_services().hf_client)
    assert (_refs(first), first.has_more) == (chat[:5], True)
    whole = get_catalog(task=ModelTask.CHAT, limit=len(chat), hf_client=get_services().hf_client)
    assert (_refs(whole), whole.has_more) == (chat, True)
    after = get_catalog(
        task=ModelTask.CHAT, limit=5, offset=len(chat), hf_client=get_services().hf_client
    )
    assert _refs(after) == ["hub/one", "hub/two"]


def test_a_featured_browse_reads_no_engine_list_and_lists_no_npu_row(
    monkeypatch: pytest.MonkeyPatch, listing: list[dict[str, object]]
) -> None:
    def _unread(engine: object) -> list[CatalogModel]:
        raise AssertionError("the featured browse read the engine list")

    monkeypatch.setattr(get_services().hf_client, "fetch_engine_models", _unread)
    assert (
        _refs(get_catalog(task=ModelTask.CHAT, featured=True, hf_client=get_services().hf_client))
        == []
    )


def test_the_hub_scan_never_asks_for_the_npu_format(listing: list[dict[str, object]]) -> None:
    get_catalog(task=ModelTask.CHAT, limit=100, hf_client=get_services().hf_client)
    assert {call["fmt"] for call in listing} == {ModelFormat.GGUF}
    assert ModelFormat.XDNA in LEMONADE.formats


@pytest.mark.parametrize(
    ("search", "refs"),
    [
        ("qwen3:4b", ["qwen3:4b#xdna"]),
        ("QWEN3:4B", ["qwen3:4b#xdna"]),
        ("Qwen3-4B", ["qwen3:4b#xdna", "qwen3-tk:4b#xdna", "qwen3-it:4b#xdna"]),
        ("qwen3 4b", ["qwen3:4b#xdna", "qwen3-tk:4b#xdna", "qwen3-it:4b#xdna"]),
        ("embed-gemma", ["embed-gemma:300m#xdna"]),
        ("zzz-no-such-model", []),
    ],
)
def test_search_finds_a_row_by_tag_or_published_name(
    listing: list[dict[str, object]], search: str, refs: list[str]
) -> None:
    found = [
        r
        for r in _refs(get_catalog(search=search, limit=100, hf_client=get_services().hf_client))
        if r.endswith("#xdna")
    ]
    assert found == refs


def test_search_npu_lists_every_row(listing: list[dict[str, object]]) -> None:
    found = [
        r
        for r in _refs(get_catalog(search="npu", limit=100, hf_client=get_services().hf_client))
        if r.endswith("#xdna")
    ]
    assert found == [m.pull_ref for m in ROWS]
    assert len(found) == 42


def test_search_and_task_filter_together(listing: list[dict[str, object]]) -> None:
    assert _refs(
        get_catalog(task=ModelTask.EMBEDDING, search="gemma", hf_client=get_services().hf_client)
    ) == ["embed-gemma:300m#xdna"]
    assert (
        _refs(get_catalog(task=ModelTask.CHAT, search="embed", hf_client=get_services().hf_client))
        == []
    )


def test_the_size_filter_reads_the_published_parameters(listing: list[dict[str, object]]) -> None:
    small = {
        m.hf_repo
        for m in get_catalog(
            size=CatalogSize.SMALL, limit=100, hf_client=get_services().hf_client
        ).models
    }
    assert ("qwen3:0.6b" in small, "qwen3:8b" in small) == (True, False)


def test_the_npu_rows_keep_the_file_order_under_a_sort(listing: list[dict[str, object]]) -> None:
    chat = [m.pull_ref for m in ROWS if m.task is ModelTask.CHAT]
    result = get_catalog(
        task=ModelTask.CHAT,
        sort=CatalogSort.SIZE_DESC,
        limit=100,
        hf_client=get_services().hf_client,
    )
    assert _refs(result) == [*chat, "hub/one", "hub/two"]


def test_the_installed_filter_selects_the_pulled_npu_row(listing: list[dict[str, object]]) -> None:
    class Manager:
        def list_installed(self) -> list[str]:
            return ["qwen3:4b#xdna"]

    on = get_catalog(
        task=ModelTask.CHAT,
        installed=True,
        model_manager=Manager(),
        limit=100,
        hf_client=get_services().hf_client,
    )
    assert _refs(on) == ["qwen3:4b#xdna"]
    off = _refs(
        get_catalog(
            task=ModelTask.CHAT,
            installed=False,
            model_manager=Manager(),
            limit=100,
            hf_client=get_services().hf_client,
        )
    )
    assert "qwen3:4b#xdna" not in off
    assert len([r for r in off if r.endswith("#xdna")]) == 40


def test_a_host_without_the_npu_hides_the_rows_with_the_gate_reason(
    monkeypatch: pytest.MonkeyPatch, listing: list[dict[str, object]]
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: NO_NPU)
    reason = LEMONADE.host_gate(NO_NPU).reason
    result = get_catalog(task=ModelTask.CHAT, search="qwen3:4b", hf_client=get_services().hf_client)
    assert _refs(result) == []
    assert [(h.hf_repo, h.reason, h.text) for h in result.hidden] == [
        ("qwen3:4b", "host_unsupported", f"This computer cannot run it. {reason}")
    ]
    assert hidden_summary([h.text for h in result.hidden]) == (
        f"1 hidden (This computer cannot run it. {reason})"
    )


def test_hidden_rows_follow_the_task_filter(
    monkeypatch: pytest.MonkeyPatch, listing: list[dict[str, object]]
) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: NO_NPU)
    chat = get_catalog(task=ModelTask.CHAT, limit=100, hf_client=get_services().hf_client)
    embedding = get_catalog(task=ModelTask.EMBEDDING, limit=100, hf_client=get_services().hf_client)
    assert (len(chat.hidden), len(embedding.hidden)) == (41, 1)
    assert [m.hf_repo for m in chat.models] == ["hub/one", "hub/two"]


def test_an_unknown_host_lists_the_rows(
    monkeypatch: pytest.MonkeyPatch, listing: list[dict[str, object]]
) -> None:
    unread = host_caps.collect(
        system="Linux", machine="x86_64", cpu_info={}, nvidia=None, accel_devices=None
    )
    monkeypatch.setattr(runnable, "session_host", lambda: unread)
    result = get_catalog(task=ModelTask.CHAT, limit=100, hf_client=get_services().hf_client)
    assert (len(result.models), result.hidden) == (43, ())


def test_the_offers_carry_lemonade_after_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lemonade.shutil, "which", lambda tool: None)
    offers = default_offers()
    assert [(str(o.adapter.name), o.state) for o in offers] == [
        ("llama.cpp", EngineState.INSTALLED),
        ("lemonade", EngineState.INSTALLABLE),
    ]


def test_an_installed_lemond_makes_the_offer_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lemonade.shutil, "which", lambda tool: "/bin/lemond")
    state = {str(o.adapter.name): o.state for o in default_offers()}
    assert state[EngineName.LEMONADE] is EngineState.INSTALLED


def test_an_unavailable_engine_lists_no_rows(
    monkeypatch: pytest.MonkeyPatch, listing: list[dict[str, object]]
) -> None:
    offers = [runnable.EngineOffer(LEMONADE, EngineState.UNAVAILABLE)]
    monkeypatch.setattr(_query, "default_offers", lambda: offers)
    assert _query._engine_rows(offers, hf_client=get_services().hf_client) == []


# ---- the read itself: one request, cached per client, failures leave the Hub rows ----


class _Router:
    """Serves the pinned list; ``"500json"`` is a server error whose body is a valid list."""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.urls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self.urls.append(url)
        request = httpx.Request("GET", url)
        if self.mode == "connect":
            raise httpx.ConnectError("offline", request=request)
        if self.mode == "500":
            return httpx.Response(500, text="oops", request=request)
        if self.mode == "500json":
            return httpx.Response(500, json=PINNED, request=request)
        if self.mode == "badjson":
            return httpx.Response(200, text="{not json", request=request)
        if self.mode == "list":
            return httpx.Response(200, json=[1, 2, 3], request=request)
        return httpx.Response(200, json=PINNED, request=request)


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> _Router:
    routed = _Router()
    monkeypatch.setattr(httpx, "get", routed.get)
    return routed


@pytest.mark.real_engine_model_lists
def test_the_list_is_read_once_from_the_pinned_tag_and_cached(router: _Router) -> None:
    client = get_services().hf_client
    first = client.fetch_engine_models(LEMONADE)
    second = client.fetch_engine_models(LEMONADE)
    assert router.urls == [LEMONADE.models_url]
    assert "/v1.0.7/" in router.urls[0]
    assert (len(first), second is first) == (42, True)


@pytest.mark.real_engine_model_lists
def test_a_new_client_reads_the_list_again(router: _Router) -> None:
    make_hf_client(get_services().hf_client._config).fetch_engine_models(LEMONADE)
    make_hf_client(get_services().hf_client._config).fetch_engine_models(LEMONADE)
    assert len(router.urls) == 2


@pytest.mark.real_engine_model_lists
def test_a_cached_list_expires(monkeypatch: pytest.MonkeyPatch, router: _Router) -> None:
    client = get_services().hf_client
    monkeypatch.setattr(HfClient, "CACHE_TTL", 0.0)
    client.fetch_engine_models(LEMONADE)
    client.fetch_engine_models(LEMONADE)
    assert len(router.urls) == 2


@pytest.mark.real_engine_model_lists
@pytest.mark.parametrize("mode", ["500", "500json", "connect", "badjson", "list"])
def test_a_failed_read_gives_no_rows_and_one_warning(
    router: _Router, mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    router.mode = mode
    client = get_services().hf_client
    with caplog.at_level(logging.DEBUG, logger="lilbee.catalog.hf_client"):
        assert client.fetch_engine_models(LEMONADE) == []
        assert client.fetch_engine_models(LEMONADE) == []
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warned) == 1
    assert "lemonade model list" in warned[0].getMessage()
    assert len(router.urls) == 1


@pytest.mark.real_hf_client
@pytest.mark.real_engine_model_lists
def test_an_offline_browse_warns_once_for_each_source(
    router: _Router, caplog: pytest.LogCaptureFixture
) -> None:
    router.mode = "connect"
    with caplog.at_level(logging.DEBUG, logger="lilbee.catalog.hf_client"):
        get_catalog(task=ModelTask.CHAT, hf_client=get_services().hf_client)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert [w.split(":")[0] for w in warned] == [
        "Failed to fetch models from the lemonade model list",
        "Failed to fetch models from HuggingFace",
    ]


@pytest.mark.real_engine_model_lists
def test_a_failed_read_is_not_repeated_until_the_cache_expires(
    monkeypatch: pytest.MonkeyPatch, router: _Router
) -> None:
    client = get_services().hf_client
    router.mode = "500"
    assert client.fetch_engine_models(LEMONADE) == []
    router.mode = "ok"
    assert client.fetch_engine_models(LEMONADE) == []
    assert len(router.urls) == 1
    monkeypatch.setattr(HfClient, "CACHE_TTL", 0.0)
    assert len(client.fetch_engine_models(LEMONADE)) == 42
    assert len(router.urls) == 2


@pytest.mark.real_engine_model_lists
def test_a_second_browse_with_the_publisher_down_makes_no_request(
    monkeypatch: pytest.MonkeyPatch, router: _Router
) -> None:
    router.mode = "connect"
    monkeypatch.setattr(runnable, "session_host", lambda: NO_NPU)
    stub_hf_listing(monkeypatch, {"text-generation": [_hub_row("hub/one")]})
    get_catalog(task=ModelTask.CHAT, hf_client=get_services().hf_client)
    get_catalog(task=ModelTask.CHAT, hf_client=get_services().hf_client)
    assert router.urls == [LEMONADE.models_url]


@pytest.mark.real_engine_model_lists
def test_a_failed_list_read_leaves_the_hub_rows(
    monkeypatch: pytest.MonkeyPatch, router: _Router
) -> None:
    router.mode = "connect"
    monkeypatch.setattr(_query, "get_picks", lambda **_: [])
    monkeypatch.setattr(runnable, "session_host", lambda: XDNA2)
    stub_hf_listing(monkeypatch, {"text-generation": [_hub_row("hub/one")]})
    result = get_catalog(task=ModelTask.CHAT, hf_client=get_services().hf_client)
    assert [m.hf_repo for m in result.models] == ["hub/one"]
    assert router.urls == [LEMONADE.models_url]
