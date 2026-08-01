"""Tests for catalog/picks.py: tier spread, role verification, and caching."""

from __future__ import annotations

import pytest

from lilbee.catalog.models import CatalogModel, HfPage
from lilbee.catalog.types import CatalogSize, ModelCompat, ModelTask

pytestmark = pytest.mark.live_picks  # these drive the real resolution path


def _model(hf_repo: str, task: str, params: int) -> CatalogModel:
    return CatalogModel(
        hf_repo=hf_repo,
        gguf_filename="*Q4_K_M.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=10,
        task=task,
        compat=ModelCompat.SUPPORTED,
        params=params,
    )


# Trending order within each tier is the order these appear, so the first two of
# each tier are the ones that must be picked.
_ALL_ROLES = {
    "chat": None,  # filled below
    "embedding": [_model("e/gte-base-GGUF", "embedding", 100_000_000)],
    "vision": [_model("v/Qwen-VL-GGUF", "vision", 8_000_000_000)],
    "rerank": [_model("r/bge-reranker-GGUF", "rerank", 2_000_000_000)],
}

_CHAT_CANDIDATES = [
    _model("a/Small-1B-GGUF", "chat", 1_000_000_000),
    _model("a/Small-2B-GGUF", "chat", 2_000_000_000),
    _model("a/Small-3B-GGUF", "chat", 3_000_000_000),
    _model("a/Mid-8B-GGUF", "chat", 8_000_000_000),
    _model("a/Mid-13B-GGUF", "chat", 13_000_000_000),
    _model("a/Large-27B-GGUF", "chat", 27_000_000_000),
    _model("a/Large-34B-GGUF", "chat", 34_000_000_000),
    _model("a/Huge-70B-GGUF", "chat", 70_000_000_000),
    _model("a/Huge-405B-GGUF", "chat", 405_000_000_000),
]


_ALL_ROLES["chat"] = _CHAT_CANDIDATES


@pytest.fixture(autouse=True)
def _clean_picks(monkeypatch):
    from lilbee.catalog import picks as picks_mod

    # Drive the retry deadline to zero so a re-resolve is observable without
    # waiting out the production backoff.
    monkeypatch.setattr(picks_mod, "_RETRY_BACKOFF_S", 0.0)
    picks_mod.reset_picks()
    yield
    picks_mod.reset_picks()


def _stub_fetch(monkeypatch, by_task: dict[str, list[CatalogModel]]) -> list[dict]:
    """Route hf_client.fetch_models to *by_task*; return the recorded call log."""
    from lilbee.app.services import get_services
    from lilbee.catalog.query import pipeline_to_task

    calls: list[dict] = []

    def fake(**kw):
        calls.append(kw)
        task = pipeline_to_task(kw.get("pipeline_tag", ""))
        return HfPage(models=list(by_task.get(task, [])), has_more=False)

    monkeypatch.setattr(get_services().hf_client, "fetch_models", fake)
    return calls


class TestChatTierSpread:
    def test_picks_two_from_every_parameter_tier(self, monkeypatch) -> None:
        from lilbee.catalog.picks import picks_for
        from lilbee.catalog.query import size_bucket

        _stub_fetch(monkeypatch, {"chat": _CHAT_CANDIDATES})
        chat = picks_for(ModelTask.CHAT)

        assert len(chat) == 8
        by_tier = [size_bucket(m.params) for m in chat]
        for tier in CatalogSize:
            assert by_tier.count(tier) == 2

    def test_takes_the_most_popular_of_each_tier(self, monkeypatch) -> None:
        """Candidates keep trending order, so each tier contributes its head."""
        from lilbee.catalog.picks import picks_for

        _stub_fetch(monkeypatch, {"chat": _CHAT_CANDIDATES})
        repos = [m.hf_repo for m in picks_for(ModelTask.CHAT)]

        assert repos[:2] == ["a/Small-1B-GGUF", "a/Small-2B-GGUF"]
        assert "a/Small-3B-GGUF" not in repos  # third in its tier, so it loses

    def test_sparse_tier_contributes_what_it_has(self, monkeypatch) -> None:
        """A tier short of its quota does not borrow from another tier."""
        from lilbee.catalog.picks import picks_for
        from lilbee.catalog.query import size_bucket

        thin = [m for m in _CHAT_CANDIDATES if size_bucket(m.params) != CatalogSize.HUGE]
        thin.append(_model("a/Only-70B-GGUF", "chat", 70_000_000_000))
        _stub_fetch(monkeypatch, {"chat": thin})
        chat = picks_for(ModelTask.CHAT)

        huge = [m for m in chat if size_bucket(m.params) == CatalogSize.HUGE]
        assert len(huge) == 1
        assert len(chat) == 7

    def test_models_without_a_parameter_count_are_skipped(self, monkeypatch) -> None:
        """A repo publishing no GGUF metadata has no tier to sit in."""
        from lilbee.catalog.picks import picks_for

        _stub_fetch(monkeypatch, {"chat": [_model("a/No-Meta-GGUF", "chat", 0)]})
        assert picks_for(ModelTask.CHAT) == ()


class TestRoleVerification:
    def test_drops_a_chat_model_mistagged_as_a_reranker(self, monkeypatch) -> None:
        """Regression: a live fetch returned a Llama-2 chat model under
        ``text-classification``, which would have been shown as a reranker."""
        from lilbee.catalog.picks import picks_for

        _stub_fetch(
            monkeypatch,
            {
                "rerank": [
                    _model("TheBloke/llama-2-13B-Guanaco-GGUF", "rerank", 13_000_000_000),
                    _model("pervll/bge-reranker-v2-GGUF", "rerank", 2_500_000_000),
                ]
            },
        )
        rerank = picks_for(ModelTask.RERANK)

        assert [m.hf_repo for m in rerank] == ["pervll/bge-reranker-v2-GGUF"]

    def test_drops_a_chat_model_mistagged_as_an_embedder(self, monkeypatch) -> None:
        from lilbee.catalog.picks import picks_for

        _stub_fetch(
            monkeypatch,
            {
                "embedding": [
                    _model("Anbeeld/Qwen3-35B-A3B-GGUF", "embedding", 35_000_000_000),
                    _model("zenyr/mxbai-embed-large-GGUF", "embedding", 300_000_000),
                    _model("x/Octen-Embedding-4B-GGUF", "embedding", 4_000_000_000),
                    _model("y/gte-base-GGUF", "embedding", 100_000_000),
                ]
            },
        )
        repos = [m.hf_repo for m in picks_for(ModelTask.EMBEDDING)]

        assert "Anbeeld/Qwen3-35B-A3B-GGUF" not in repos
        assert len(repos) == 3

    def test_vision_is_settled_by_the_projector_probe(self, monkeypatch) -> None:
        """An mmproj sibling is definitive, and catches VL repos no name matches.

        Neither candidate matches a vision name pattern, so only the projector
        probe can tell them apart.
        """
        from lilbee.catalog import picks as picks_mod
        from lilbee.catalog.picks import picks_for

        _stub_fetch(
            monkeypatch,
            {
                "vision": [
                    _model("a/Plain-Chat-GGUF", "vision", 7_000_000_000),
                    _model("a/Qwen3-VL-GGUF", "vision", 8_000_000_000),
                ]
            },
        )
        probed: list[str] = []

        def fake_probe(hf_repo: str) -> bool:
            probed.append(hf_repo)
            return hf_repo == "a/Qwen3-VL-GGUF"

        monkeypatch.setattr(picks_mod, "repo_has_mmproj", fake_probe)

        assert [m.hf_repo for m in picks_for(ModelTask.VISION)] == ["a/Qwen3-VL-GGUF"]
        assert probed == ["a/Plain-Chat-GGUF", "a/Qwen3-VL-GGUF"]

    def test_vision_probe_stops_once_the_quota_is_met(self, monkeypatch) -> None:
        """The probe costs one request per candidate, so it must not scan on."""
        from lilbee.catalog import picks as picks_mod
        from lilbee.catalog.picks import picks_for

        _stub_fetch(
            monkeypatch,
            {"vision": [_model(f"a/VL-{i}-GGUF", "vision", 8_000_000_000) for i in range(5)]},
        )
        probed: list[str] = []
        monkeypatch.setattr(picks_mod, "repo_has_mmproj", lambda r: (probed.append(r), True)[1])

        assert len(picks_for(ModelTask.VISION)) == 1
        assert probed == ["a/VL-0-GGUF"]


class TestPickFlagAndCaching:
    def test_picks_are_flagged_for_the_picks_section(self, monkeypatch) -> None:
        """fetch_models builds plain browse rows; the flag is what stars them."""
        from lilbee.catalog.picks import get_picks

        _stub_fetch(monkeypatch, {"chat": _CHAT_CANDIDATES})
        assert all(m.featured for m in get_picks())

    def test_resolved_once_per_process(self, monkeypatch) -> None:
        """Rows must not reshuffle while the user types in the catalog search."""
        from lilbee.catalog import picks as picks_mod
        from lilbee.catalog.picks import get_picks

        monkeypatch.setattr(picks_mod, "repo_has_mmproj", lambda r: True)
        calls = _stub_fetch(monkeypatch, _ALL_ROLES)
        first = get_picks()
        after_first = len(calls)
        second = get_picks()

        assert first == second
        assert len(calls) == after_first  # no refetch

    def test_an_empty_result_is_not_memoized(self, monkeypatch) -> None:
        """An offline launch that later regains network still fills in."""
        from lilbee.catalog.picks import get_picks

        _stub_fetch(monkeypatch, {})
        assert get_picks() == ()

        _stub_fetch(monkeypatch, {"chat": _CHAT_CANDIDATES})
        assert get_picks() != ()

    def test_a_role_missing_from_the_result_is_retried(self, monkeypatch) -> None:
        """One role's failed fetch must not leave it empty for the process."""
        from lilbee.catalog import picks as picks_mod
        from lilbee.catalog.picks import get_picks, picks_for

        monkeypatch.setattr(picks_mod, "repo_has_mmproj", lambda r: True)
        partial = dict(_ALL_ROLES, embedding=[])
        _stub_fetch(monkeypatch, partial)
        assert get_picks()  # served despite the gap
        assert picks_for(ModelTask.EMBEDDING) == ()

        _stub_fetch(monkeypatch, _ALL_ROLES)
        assert picks_for(ModelTask.EMBEDDING)  # refilled once HF recovered

    def test_a_complete_result_is_not_re_resolved(self, monkeypatch) -> None:
        """A full set is final; only a gap earns a retry."""
        from lilbee.catalog import picks as picks_mod
        from lilbee.catalog.picks import get_picks

        monkeypatch.setattr(picks_mod, "repo_has_mmproj", lambda r: True)
        calls = _stub_fetch(monkeypatch, _ALL_ROLES)
        get_picks()
        settled = len(calls)
        get_picks()
        get_picks()
        assert len(calls) == settled

    def test_a_fetch_failure_yields_no_picks(self, monkeypatch) -> None:
        from lilbee.app.services import get_services
        from lilbee.catalog.picks import get_picks

        def boom(**kw):
            raise RuntimeError("HuggingFace unreachable")

        monkeypatch.setattr(get_services().hf_client, "fetch_models", boom)
        assert get_picks() == ()

    def test_reset_drops_the_memo(self, monkeypatch) -> None:
        from lilbee.catalog.picks import get_picks, reset_picks

        calls = _stub_fetch(monkeypatch, {"chat": _CHAT_CANDIDATES})
        get_picks()
        before = len(calls)
        reset_picks()
        get_picks()

        assert len(calls) > before


class TestTrendingRequest:
    def test_requests_the_trending_ranking(self, monkeypatch) -> None:
        from lilbee.catalog.picks import TRENDING_SORT, get_picks

        calls = _stub_fetch(monkeypatch, {"chat": _CHAT_CANDIDATES})
        get_picks()

        assert calls
        assert all(c["sort"] == TRENDING_SORT for c in calls)


class TestBackoffAndConcurrency:
    def test_a_failed_resolution_is_not_retried_until_the_backoff_expires(
        self, monkeypatch
    ) -> None:
        """A degraded network must not turn every read into a fresh fan-out."""
        from lilbee.catalog import picks as picks_mod
        from lilbee.catalog.picks import get_picks

        monkeypatch.setattr(picks_mod, "_RETRY_BACKOFF_S", 3600.0)
        calls = _stub_fetch(monkeypatch, {})
        assert get_picks() == ()
        after_first = len(calls)

        assert get_picks() == ()
        assert len(calls) == after_first  # inside the backoff, no refetch

    def test_a_set_landed_by_another_thread_wins(self, monkeypatch) -> None:
        """A slow resolver must not clobber a complete set that arrived first."""
        from lilbee.catalog import picks as picks_mod

        monkeypatch.setattr(picks_mod, "repo_has_mmproj", lambda r: True)
        _stub_fetch(monkeypatch, _ALL_ROLES)
        picks = picks_mod.ModelPicks()
        landed = tuple(_CHAT_CANDIDATES[:1])

        real_resolve = picks_mod._resolve_picks

        def resolve_then_race() -> tuple:
            result = real_resolve()
            picks.seed(landed)  # another thread completes mid-fetch
            return result

        monkeypatch.setattr(picks_mod, "_resolve_picks", resolve_then_race)
        assert picks.all() == landed
