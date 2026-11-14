"""The host gate and the catalog filter that hides rows this host cannot run."""

from __future__ import annotations

import pytest

from conftest import stub_hf_listing
from lilbee.catalog import query as _query
from lilbee.catalog.formatting import enrich_catalog
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.query import get_catalog
from lilbee.catalog.runnable import (
    EngineOffer,
    EngineState,
    HiddenReason,
    default_offers,
    explain,
    hidden_reason,
    hidden_summary,
    runnable_formats,
)
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.providers.fleet import host_caps
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.host_gate import HostGate, gate_on
from lilbee.providers.fleet.host_caps import Feature, HostCaps, NvidiaGpu, nvidia_feature
from tests.conformance.fake_adapter import FakeAdapter

PRESENT = Feature(True)
ABSENT = Feature(False, "lacks it")
UNKNOWN = Feature(False, "could not be read", known=False)


def _host(
    *,
    avx2: Feature = PRESENT,
    neon: Feature = PRESENT,
    gpus: tuple[NvidiaGpu, ...] = (),
    known: bool = True,
) -> HostCaps:
    return HostCaps(
        os=None,
        arch=None,
        macos_version=None,
        avx2=avx2,
        neon=neon,
        gpus=gpus,
        nvidia_driver_cuda=None,
        nvidia_reason="" if known else host_caps.NVIDIA_UNREADABLE,
        nvidia_known=known,
        npus=(),
    )


GPU = (NvidiaGpu("GTX", (6, 1)),)


@pytest.mark.parametrize(
    ("feature", "ok", "known", "auto"),
    [
        (PRESENT, True, True, True),
        (ABSENT, False, True, False),
        (UNKNOWN, True, False, False),
    ],
    ids=["present", "absent", "unknown"],
)
def test_gate_on_each_feature_state(feature: Feature, ok: bool, known: bool, auto: bool) -> None:
    gate = gate_on(feature)
    assert (gate.ok, gate.known, gate.auto_selectable) == (ok, known, auto)
    assert gate.reason == feature.reason


@pytest.mark.parametrize("field", ["avx2", "neon"])
@pytest.mark.parametrize(
    ("feature", "ok", "auto"),
    [(PRESENT, True, True), (ABSENT, False, False), (UNKNOWN, True, False)],
    ids=["known-present", "known-absent", "unknown"],
)
def test_host_field_gates_in_three_states(
    field: str, feature: Feature, ok: bool, auto: bool
) -> None:
    host = _host(**{field: feature})
    gate = gate_on(getattr(host, field))
    assert (gate.ok, gate.auto_selectable) == (ok, auto)


@pytest.mark.parametrize(
    ("gpus", "known", "ok", "auto"),
    [(GPU, True, True, True), ((), True, False, False), ((), False, True, False)],
    ids=["known-present", "known-absent", "unknown"],
)
def test_nvidia_field_gates_in_three_states(
    gpus: tuple[NvidiaGpu, ...], known: bool, ok: bool, auto: bool
) -> None:
    gate = gate_on(nvidia_feature(_host(gpus=gpus, known=known)))
    assert (gate.ok, gate.auto_selectable) == (ok, auto)
    if not ok or not known:
        assert gate.reason


def test_absent_wins_over_unknown_and_unknown_shows_its_reason() -> None:
    assert gate_on(UNKNOWN, ABSENT) == HostGate(False, "lacks it")
    assert gate_on(PRESENT, UNKNOWN) == HostGate(True, "could not be read", known=False)


def test_llama_runs_on_any_host() -> None:
    assert LLAMA.host_gate(_host(avx2=ABSENT, neon=ABSENT)).auto_selectable


def _fake_offer(state: EngineState = EngineState.INSTALLED) -> EngineOffer:
    return EngineOffer(FakeAdapter(), state)


def test_unknown_field_keeps_the_model_visible_and_says_why() -> None:
    host = _host(avx2=UNKNOWN, gpus=GPU)
    offers = [_fake_offer()]
    assert hidden_reason({ModelFormat.SAFETENSORS}, host, offers) is None
    assert offers[0].adapter.host_gate(host).reason == "could not be read"


def test_refused_host_hides_with_the_gate_reason() -> None:
    hidden = explain({ModelFormat.SAFETENSORS}, _host(avx2=ABSENT, gpus=GPU), [_fake_offer()])
    assert hidden is not None
    assert hidden.reason is HiddenReason.HOST_UNSUPPORTED
    assert "lacks it" in hidden.text


def test_no_engine_for_the_format_hides_it() -> None:
    hidden = explain({ModelFormat.MLX}, _host(), default_offers())
    assert hidden is not None
    assert hidden.reason is HiddenReason.NO_ENGINE
    assert "mlx" in hidden.text


def test_unavailable_engine_does_not_count() -> None:
    offers = [_fake_offer(EngineState.UNAVAILABLE)]
    assert (
        hidden_reason({ModelFormat.SAFETENSORS}, _host(gpus=GPU), offers) is HiddenReason.NO_ENGINE
    )


def test_runnable_formats_drops_what_no_engine_runs() -> None:
    both = {ModelFormat.GGUF, ModelFormat.MLX}
    assert runnable_formats(both, _host(), default_offers()) == {ModelFormat.GGUF}


def test_summary_counts_and_dedupes_reasons() -> None:
    assert hidden_summary([]) == ""
    assert hidden_summary(["no engine for mlx", "no engine for mlx"]) == (
        "2 hidden (no engine for mlx)"
    )


def _row(repo: str, formats: frozenset[ModelFormat]) -> CatalogModel:
    return CatalogModel(
        hf_repo=repo,
        gguf_filename="*.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=10,
        task=ModelTask.CHAT,
        formats=formats,
    )


def test_catalog_hides_a_row_and_lists_one_row_with_runnable_formats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_query, "get_picks", lambda: [])
    stub_hf_listing(
        monkeypatch,
        {
            "text-generation": [
                _row("a/both", frozenset({ModelFormat.GGUF, ModelFormat.MLX})),
                _row("b/mlx-only", frozenset({ModelFormat.MLX})),
            ]
        },
    )
    result = get_catalog(task=ModelTask.CHAT)
    assert [(m.hf_repo, m.formats) for m in result.models] == [("a/both", {ModelFormat.GGUF})]
    assert [(h.hf_repo, h.reason) for h in result.hidden] == [("b/mlx-only", "no_engine")]


def test_every_surface_shows_the_same_hidden_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_query, "get_picks", lambda: [])
    stub_hf_listing(
        monkeypatch, {"text-generation": [_row("b/mlx-only", frozenset({ModelFormat.MLX}))]}
    )
    result = get_catalog(task=ModelTask.CHAT)
    texts = [h.text for h in result.hidden]
    assert texts == ["no engine for mlx"]
    assert hidden_summary(texts) == "1 hidden (no engine for mlx)"
    assert enrich_catalog(result, set()) == []


@pytest.mark.asyncio
async def test_server_and_mcp_lists_carry_the_same_hidden_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee import mcp_server
    from lilbee.server import handlers

    monkeypatch.setattr(_query, "get_picks", lambda: [])
    stub_hf_listing(
        monkeypatch,
        {
            "text-generation": [
                _row("a/both", frozenset({ModelFormat.GGUF, ModelFormat.MLX})),
                _row("b/mlx-only", frozenset({ModelFormat.MLX})),
            ]
        },
    )
    resp = await handlers.models_catalog(task="chat", source="native")
    assert [h.text for h in resp.hidden] == ["no engine for mlx"]
    assert resp.hidden_summary == "1 hidden (no engine for mlx)"
    assert [(m.hf_repo, m.formats) for m in resp.models] == [("a/both", ["gguf"])]
    listed = mcp_server.catalog_browse(task="chat")
    assert listed["hidden_summary"] == resp.hidden_summary


def _two_format_offers() -> list[EngineOffer]:
    return [EngineOffer(LLAMA, EngineState.INSTALLED), _fake_offer()]


def _by_format_listing(
    monkeypatch: pytest.MonkeyPatch, rows: dict[ModelFormat, list[CatalogModel]]
) -> list[ModelFormat]:
    """Serve one stream per weight format, as the Hub filter does; record the formats asked."""
    from lilbee.app.services import get_services
    from lilbee.catalog.models import HfPage

    asked: list[ModelFormat] = []

    def _fetch(**kwargs: object) -> HfPage:
        fmt = ModelFormat(str(kwargs["fmt"]))
        asked.append(fmt)
        return HfPage(models=rows.get(fmt, []), next_cursor=None)

    monkeypatch.setattr(get_services().hf_client, "fetch_models", _fetch)
    return asked


def test_scan_reads_one_stream_per_engine_format_and_merges_by_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_query, "get_picks", lambda: [])
    monkeypatch.setattr(_query, "default_offers", _two_format_offers)
    monkeypatch.setattr(_query, "session_host", lambda: _host(gpus=GPU))
    gguf = _row("a/gguf", frozenset({ModelFormat.GGUF}))
    st = _row("b/st", frozenset({ModelFormat.SAFETENSORS}))
    both_a = _row("c/both", frozenset({ModelFormat.GGUF}))
    both_b = _row("c/both", frozenset({ModelFormat.SAFETENSORS, ModelFormat.GGUF}))
    asked = _by_format_listing(
        monkeypatch,
        {ModelFormat.GGUF: [gguf, both_a], ModelFormat.SAFETENSORS: [st, both_b]},
    )
    result = get_catalog(task=ModelTask.CHAT)
    assert sorted(asked) == [ModelFormat.GGUF, ModelFormat.SAFETENSORS]
    assert sorted(m.hf_repo for m in result.models) == ["a/gguf", "b/st", "c/both"]
    assert result.hidden == ()


def test_a_hidden_pick_the_scan_finds_again_counts_once(monkeypatch: pytest.MonkeyPatch) -> None:
    pick = _row("p/mlx-pick", frozenset({ModelFormat.MLX}))
    monkeypatch.setattr(_query, "get_picks", lambda: [pick])
    stub_hf_listing(
        monkeypatch, {"text-generation": [pick, _row("a/ok", frozenset({ModelFormat.GGUF}))]}
    )
    result = get_catalog(task=ModelTask.CHAT)
    assert [h.hf_repo for h in result.hidden] == ["p/mlx-pick"]


def test_hidden_rows_follow_the_search_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_query, "get_picks", lambda: [])
    stub_hf_listing(
        monkeypatch,
        {
            "text-generation": [
                _row("b/mlx-llama", frozenset({ModelFormat.MLX})),
                _row("c/mlx-qwen", frozenset({ModelFormat.MLX})),
            ]
        },
    )
    result = get_catalog(task=ModelTask.CHAT, search="qwen")
    assert [h.hf_repo for h in result.hidden] == ["c/mlx-qwen"]


def test_a_failing_host_read_leaves_the_catalog_open_and_says_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from lilbee.catalog import runnable

    def _refuse() -> HostCaps:
        raise OSError("uname refused")

    monkeypatch.setattr(runnable, "detect", _refuse)
    runnable.session_host.cache_clear()
    try:
        monkeypatch.setattr(_query, "get_picks", lambda: [])
        stub_hf_listing(
            monkeypatch, {"text-generation": [_row("a/ok", frozenset({ModelFormat.GGUF}))]}
        )
        with caplog.at_level("WARNING", logger="lilbee.catalog.runnable"):
            result = get_catalog(task=ModelTask.CHAT)
        host = runnable.session_host()
    finally:
        runnable.session_host.cache_clear()
    assert [m.hf_repo for m in result.models] == ["a/ok"]
    assert "uname refused" in caplog.text
    assert (host.avx2.known, host.nvidia_known) == (False, False)
    assert host.nvidia_reason == host_caps.NVIDIA_UNREADABLE


def test_a_host_read_that_raises_something_else_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.catalog import runnable

    def _bug() -> HostCaps:
        raise KeyError("a bug, not a host answer")

    monkeypatch.setattr(runnable, "detect", _bug)
    runnable.session_host.cache_clear()
    try:
        with pytest.raises(KeyError, match="a bug"):
            runnable.session_host()
    finally:
        runnable.session_host.cache_clear()


def test_ref_with_a_format_resolves_to_an_entry_of_that_format() -> None:
    from lilbee.catalog.query import resolve_pull_target

    entry = resolve_pull_target("acme/two#openvino")
    assert entry is not None
    assert (entry.hf_repo, entry.format, entry.pull_ref) == (
        "acme/two",
        ModelFormat.OPENVINO,
        "acme/two#openvino",
    )
    assert resolve_pull_target("acme/two#nonsense") is None
    assert resolve_pull_target("two#mlx") is None
    assert resolve_pull_target("ac me/two#mlx") is None
