"""Tests for catalog.py: model catalog, HF API fetching, filtering, downloading."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub.hf_api import RepoSibling

from lilbee import catalog
from lilbee.app.services import get_services
from lilbee.catalog import (
    FEATURED_ALL,
    FEATURED_CHAT,
    FEATURED_EMBEDDING,
    FEATURED_RERANK,
    FEATURED_VISION,
    QUANT_TIERS,
    CatalogModel,
    CatalogResult,
    EnrichedModel,
    ModelFamily,
    ModelVariant,
    build_adhoc_entry,
    clean_display_name,
    download_model,
    enrich_catalog,
    find_catalog_entry,
    get_catalog,
    get_families,
    quant_tier,
)
from lilbee.catalog import (
    download as _download,
)
from lilbee.catalog import (
    families as _families,
)
from lilbee.catalog import (
    hf_client as _hf_client,
)
from lilbee.catalog import (
    models as _models,
)
from lilbee.catalog import (
    query as _query,
)
from lilbee.catalog.hf_client import hf_token
from lilbee.catalog.models import HfPage
from lilbee.catalog.refs import GGUF_GLOB, is_bare_hf_repo, pick_best_gguf
from lilbee.catalog.types import CatalogSize, CatalogSort, ModelTask
from lilbee.core.config import cfg

_EMPTY_HF_PAGE = HfPage(models=[], has_more=False)


@pytest.fixture(autouse=True)
def _clear_hf_cache():
    """Clear the HfClient TTL cache between tests."""
    from lilbee.app.services import get_services

    get_services().hf_client._cache.clear()
    yield
    get_services().hf_client._cache.clear()


def _fake_download(**kwargs: Any) -> str:
    """Fake hf_hub_download that writes to HF cache structure with hash-based filename."""
    import hashlib

    cache_dir = kwargs.get("cache_dir", "")
    repo_id = kwargs.get("repo_id", "")

    content = b"x" * 100
    digest = hashlib.sha256(content).hexdigest()

    if repo_id:
        safe_repo = repo_id.replace("/", "--")
        model_dir = Path(cache_dir) / f"models--{safe_repo}"
        blobs_dir = model_dir / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        dest = blobs_dir / digest
    else:
        dest = Path(cache_dir) / digest
        dest.parent.mkdir(parents=True, exist_ok=True)

    dest.write_bytes(content)
    return str(dest)


class TestCatalogModelDataclass:
    def test_frozen(self) -> None:
        m = FEATURED_CHAT[0]
        with pytest.raises(AttributeError):
            m.hf_repo = "nope"  # type: ignore[misc]

    def test_fields(self) -> None:
        m = FEATURED_CHAT[0]
        assert isinstance(m.hf_repo, str)
        assert isinstance(m.gguf_filename, str)
        assert isinstance(m.size_gb, (int, float))
        assert isinstance(m.min_ram_gb, (int, float))
        assert isinstance(m.description, str)
        assert isinstance(m.featured, bool)
        assert isinstance(m.downloads, int)
        assert isinstance(m.task, str)

    def test_ref_is_hf_repo(self) -> None:
        m = FEATURED_CHAT[0]
        assert m.ref == m.hf_repo

    def test_display_name_derived(self) -> None:
        m = FEATURED_CHAT[0]
        assert m.display_name == clean_display_name(m.hf_repo)


class TestCatalogResultDataclass:
    def test_frozen(self) -> None:
        r = CatalogResult(total=0, limit=20, offset=0, models=[])
        with pytest.raises(AttributeError):
            r.total = 1  # type: ignore[misc]


class TestHfToken:
    def test_env_var_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LILBEE_HF_TOKEN", "env-token")
        assert hf_token() == "env-token"

    def test_hf_token_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LILBEE_HF_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "hf-env-token")
        assert hf_token() == "hf-env-token"

    def test_falls_back_to_cfg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.core.config import cfg

        monkeypatch.delenv("LILBEE_HF_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(cfg, "hf_token", "cfg-token")
        assert hf_token() == "cfg-token"

    def test_env_var_overrides_cfg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.core.config import cfg

        monkeypatch.setenv("LILBEE_HF_TOKEN", "env-token")
        monkeypatch.setattr(cfg, "hf_token", "cfg-token")
        assert hf_token() == "env-token"

    def test_falls_back_to_huggingface_hub_get_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.core.config import cfg

        monkeypatch.delenv("LILBEE_HF_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(cfg, "hf_token", "")
        fake_hf_hub = MagicMock()
        fake_hf_hub.get_token.return_value = "cached-token"
        monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hf_hub)
        assert hf_token() == "cached-token"

    def test_returns_none_when_all_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from lilbee.core.config import cfg

        monkeypatch.delenv("LILBEE_HF_TOKEN", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(cfg, "hf_token", "")
        fake_hf_hub = MagicMock()
        fake_hf_hub.get_token.side_effect = Exception("no token")
        monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_hf_hub)
        assert hf_token() is None

    @patch("lilbee.catalog.hf_client.hf_token", return_value=None)
    def test_headers_empty_when_no_token(self, _mock_token: MagicMock) -> None:
        """hf_headers returns empty dict when no token available."""
        assert _hf_client.hf_headers() == {}

    @patch("lilbee.catalog.hf_client.hf_token", return_value="test-token-123")
    def test_headers_include_bearer_when_token_set(self, _mock_token: MagicMock) -> None:
        """hf_headers returns Authorization header when token is available."""
        assert _hf_client.hf_headers() == {"Authorization": "Bearer test-token-123"}


class TestFeaturedModels:
    def test_chat_not_empty(self) -> None:
        assert len(FEATURED_CHAT) > 0

    def test_embedding_not_empty(self) -> None:
        assert len(FEATURED_EMBEDDING) > 0

    def test_vision_not_empty(self) -> None:
        assert len(FEATURED_VISION) > 0

    def test_all_combined(self) -> None:
        expected = (
            len(FEATURED_CHAT)
            + len(FEATURED_EMBEDDING)
            + len(FEATURED_VISION)
            + len(FEATURED_RERANK)
        )
        assert len(FEATURED_ALL) == expected

    def test_featured_rerank_present(self) -> None:
        assert len(FEATURED_RERANK) > 0
        for m in FEATURED_RERANK:
            assert m.task == "rerank"

    def test_all_featured_flag_true(self) -> None:
        for m in FEATURED_ALL:
            assert m.featured is True

    def test_chat_task(self) -> None:
        for m in FEATURED_CHAT:
            assert m.task == "chat"

    def test_embedding_task(self) -> None:
        for m in FEATURED_EMBEDDING:
            assert m.task == "embedding"

    def test_vision_task(self) -> None:
        for m in FEATURED_VISION:
            assert m.task == "vision"

    def test_no_duplicate_repos(self) -> None:
        repos = [m.hf_repo for m in FEATURED_ALL]
        assert len(repos) == len(set(repos))

    def test_size_gb_positive(self) -> None:
        for m in FEATURED_ALL:
            assert m.size_gb > 0

    def test_min_ram_gb_positive(self) -> None:
        for m in FEATURED_ALL:
            assert m.min_ram_gb > 0


class TestIsRerankRef:
    """is_rerank_ref matches the canonical GGUF rerank catalog entries."""

    def test_empty_returns_false(self) -> None:
        from lilbee.catalog import is_rerank_ref

        assert is_rerank_ref("") is False

    def test_bare_hf_repo_matches(self) -> None:
        from lilbee.catalog import FEATURED_RERANK, is_rerank_ref

        assert FEATURED_RERANK, "rerank catalog must not be empty"
        assert is_rerank_ref(FEATURED_RERANK[0].hf_repo) is True

    def test_hf_full_ref_matches(self) -> None:
        """A full hf_repo/filename ref resolves through the featured catalog."""
        from lilbee.catalog import FEATURED_RERANK, is_rerank_ref

        entry = FEATURED_RERANK[0]
        # Featured filenames may be globs; fabricate a concrete filename
        # whose stem matches the glob to exercise by_full_ref.
        full_ref = f"{entry.hf_repo}/concrete-Q4_K_M.gguf"
        # Bare hf_repo always resolves; full ref does too via stripped
        # provider prefix or by_full_ref. is_rerank_ref accepts the bare
        # hf_repo path which is the canonical browse identity.
        assert is_rerank_ref(entry.hf_repo) is True
        # Provider-prefixed ref still resolves once the prefix is stripped.
        assert is_rerank_ref(f"openai/{entry.hf_repo}") is True
        # The full filename form must not crash even if it's not in by_full_ref.
        is_rerank_ref(full_ref)

    def test_substring_non_match(self) -> None:
        """``"base"`` must NOT match ``bge-reranker-base``."""
        from lilbee.catalog import is_rerank_ref

        assert is_rerank_ref("base") is False
        assert is_rerank_ref("reranker") is False

    def test_unknown_ref_returns_false(self) -> None:
        from lilbee.catalog import is_rerank_ref

        assert is_rerank_ref("bogus/not-real") is False


class TestResolveSiblingGguf:
    def test_picks_preferred_quant(self) -> None:
        siblings = [
            RepoSibling(rfilename="model-Q8_0.gguf"),
            RepoSibling(rfilename="model-Q4_K_M.gguf"),
            RepoSibling(rfilename="README.md"),
        ]
        assert _hf_client._resolve_sibling_gguf(siblings) == "model-Q4_K_M.gguf"

    def test_no_gguf_returns_glob(self) -> None:
        siblings = [RepoSibling(rfilename="model.bin"), RepoSibling(rfilename="config.json")]
        assert _hf_client._resolve_sibling_gguf(siblings) == GGUF_GLOB

    def test_empty_list_returns_glob(self) -> None:
        assert _hf_client._resolve_sibling_gguf([]) == GGUF_GLOB


class TestEstimateSizeFromSiblings:
    def test_sizes_the_picked_quant_not_the_largest(self) -> None:
        # The row names the picked quant (Q4_K_M); size must match that file, not
        # the larger Q8_0, or size-bucket filtering mis-buckets the model.
        siblings = [
            RepoSibling(rfilename="model-Q4_K_M.gguf", size=4_000_000_000),
            RepoSibling(rfilename="model-Q8_0.gguf", size=7_000_000_000),
        ]
        assert _hf_client._resolve_sibling_gguf(siblings) == "model-Q4_K_M.gguf"
        result = _hf_client._estimate_size_from_siblings(siblings)
        assert result == round(4_000_000_000 / (1024**3), 1)

    def test_returns_zero_when_no_size(self) -> None:
        siblings = [RepoSibling(rfilename="model.gguf", size=0)]
        assert _hf_client._estimate_size_from_siblings(siblings) == 0.0

    def test_returns_zero_for_empty_list(self) -> None:
        assert _hf_client._estimate_size_from_siblings([]) == 0.0


class TestFetchHfModels:
    def _mock_hf_response(self) -> list[dict]:
        return [
            {
                "id": "user/model-7b-gguf",
                "downloads": 5000,
                "cardData": {"description": "A test model"},
                "pipeline_tag": "text-generation",
                "siblings": [
                    {"rfilename": "model-7b-Q4_K_M.gguf"},
                    {"rfilename": "model-7b-Q8_0.gguf"},
                ],
                "gguf": {"total": 7_000_000_000, "architecture": "llama"},
            },
            {
                "id": "user/model-13b-gguf",
                "downloads": 1000,
                "pipeline_tag": "text-generation",
                "siblings": [
                    {"rfilename": "model-13b-Q4_K_M.gguf"},
                ],
                "gguf": {},
            },
        ]

    def test_parses_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = httpx.Response(200, json=self._mock_hf_response())

        def mock_get(*args: object, **kwargs: object) -> httpx.Response:
            return mock_resp

        monkeypatch.setattr(httpx, "get", mock_get)
        page = get_services().hf_client.fetch_models()
        models = page.models
        assert len(models) == 2
        assert models[0].hf_repo == "user/model-7b-gguf"
        assert models[0].display_name == "model 7b"
        assert models[0].downloads == 5000
        assert models[0].featured is False
        assert models[0].task == "chat"

    def test_resolves_concrete_gguf_filename_from_siblings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Search rows carry the quant the pull path would pick, not a glob."""
        mock_resp = httpx.Response(200, json=self._mock_hf_response())
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = get_services().hf_client.fetch_models().models
        assert models[0].gguf_filename == "model-7b-Q4_K_M.gguf"
        assert models[1].gguf_filename == "model-13b-Q4_K_M.gguf"

    def test_no_gguf_siblings_keeps_glob_filename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [{"id": "user/model", "downloads": 1, "siblings": [{"rfilename": "README.md"}]}]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = get_services().hf_client.fetch_models().models
        assert models[0].gguf_filename == GGUF_GLOB

    def test_estimates_size_from_gguf_total(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = httpx.Response(200, json=self._mock_hf_response())
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        page = get_services().hf_client.fetch_models()
        models = page.models
        # gguf.total = 7_000_000_000 bytes -> ~6.5 GB
        assert models[0].size_gb == round(7_000_000_000 / (1024**3), 1)
        assert models[0].min_ram_gb == round(max(2.0, models[0].size_gb * 1.5), 1)

    def test_empty_gguf_meta_falls_back_to_siblings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = httpx.Response(200, json=self._mock_hf_response())
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        page = get_services().hf_client.fetch_models()
        models = page.models
        # Second model has empty gguf metadata and siblings without size -> 0.0
        assert models[1].size_gb == 0.0

    def test_gguf_expand_param_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify expand=gguf is included in API request params."""
        mock_resp = httpx.Response(200, json=[])
        captured_params: dict = {}

        def capture_get(url: str, params: dict | None = None, **kwargs: Any) -> httpx.Response:
            captured_params.update(params or {})
            return mock_resp

        monkeypatch.setattr(httpx, "get", capture_get)
        get_services().hf_client.fetch_models()
        assert "gguf" in captured_params.get("expand", [])

    def test_library_param_passed_to_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify library parameter is included in API request."""
        mock_resp = httpx.Response(200, json=[])
        captured_params: dict = {}

        def capture_get(url: str, params: dict | None = None, **kwargs: Any) -> httpx.Response:
            captured_params.update(params or {})
            return mock_resp

        monkeypatch.setattr(httpx, "get", capture_get)
        get_services().hf_client.fetch_models(library="sentence-transformers")
        assert captured_params.get("library") == "sentence-transformers"

    def test_skips_entries_without_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [{"id": "", "downloads": 0}, {"downloads": 0}]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        page = get_services().hf_client.fetch_models()
        models = page.models
        assert len(models) == 0

    def test_http_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_get(*a: object, **kw: object) -> httpx.Response:
            raise httpx.ConnectError("fail")

        monkeypatch.setattr(httpx, "get", mock_get)
        page = get_services().hf_client.fetch_models()
        models = page.models
        assert models == []

    def test_invalid_json_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_get(*a: object, **kw: object) -> httpx.Response:
            raise ValueError("bad json")

        monkeypatch.setattr(httpx, "get", mock_get)
        page = get_services().hf_client.fetch_models()
        models = page.models
        assert models == []

    def test_http_status_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = httpx.Response(500)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        page = get_services().hf_client.fetch_models()
        models = page.models
        assert models == []

    def test_truncates_long_description(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [
            {
                "id": "user/test",
                "downloads": 0,
                "cardData": {"description": "A" * 200},
                "siblings": [{"rfilename": "model.gguf"}],
            }
        ]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        page = get_services().hf_client.fetch_models()
        models = page.models
        assert len(models[0].description) == 120

    def test_uses_pipeline_tag_for_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [
            {
                "id": "user/embed-model",
                "downloads": 100,
                "pipeline_tag": "feature-extraction",
                "siblings": [{"rfilename": "embed.gguf"}],
            },
            {
                "id": "user/vision-model",
                "downloads": 50,
                "pipeline_tag": "image-text-to-text",
                "siblings": [{"rfilename": "vision.gguf"}],
            },
        ]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        page = get_services().hf_client.fetch_models()
        models = page.models
        assert models[0].task == "embedding"
        assert models[1].task == "vision"

    def test_missing_pipeline_tag_defaults_to_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [{"id": "user/model", "downloads": 100, "siblings": [{"rfilename": "m.gguf"}]}]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        page = get_services().hf_client.fetch_models()
        models = page.models
        assert models[0].task == "chat"

    def test_has_more_true_when_link_header_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """has_more is True when the response contains a Link rel=next header."""
        data = [{"id": "user/model", "downloads": 100, "siblings": []}]
        mock_resp = httpx.Response(
            200,
            json=data,
            headers={"Link": '<https://huggingface.co/api/models?limit=50&skip=50>; rel="next"'},
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        page = get_services().hf_client.fetch_models()
        assert page.has_more is True
        assert len(page.models) == 1

    def test_has_more_false_when_no_link_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """has_more is False when no Link header in response (last page)."""
        data = [{"id": "user/model", "downloads": 100, "siblings": []}]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        page = get_services().hf_client.fetch_models()
        assert page.has_more is False

    def test_has_more_false_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Error responses return empty page with has_more=False."""
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(500))
        page = get_services().hf_client.fetch_models()
        assert page.has_more is False
        assert page.models == []

    def test_transport_error_warns_first_then_throttles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First HF transport failure warns; a repeat within the window is throttled to DEBUG."""

        def _raise(*_a: object, **_kw: object) -> httpx.Response:
            raise httpx.ConnectError("offline")

        monkeypatch.setattr(httpx, "get", _raise)
        client = get_services().hf_client
        # Fresh client: the -inf sentinel means "never warned", so the first
        # failure always warns regardless of the absolute monotonic clock value.
        assert client._last_fetch_failure_warn == float("-inf")
        assert client.fetch_models().models == []
        first_warn_at = client._last_fetch_failure_warn
        assert first_warn_at > float("-inf")  # the WARNING branch ran and stamped the clock
        assert client.fetch_models().models == []
        # Still inside FETCH_FAILURE_WARN_INTERVAL_S: the repeat is throttled to
        # DEBUG and does not re-stamp the clock.
        assert client._last_fetch_failure_warn == first_warn_at


class TestGetCatalog:
    def test_returns_featured_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_services().hf_client, "fetch_models", lambda **kw: _EMPTY_HF_PAGE)
        result = get_catalog()
        assert result.total == len(FEATURED_ALL)
        assert all(m.featured for m in result.models)

    def test_pagination(self) -> None:
        result = get_catalog(limit=2, offset=0)
        assert len(result.models) == 2
        assert result.limit == 2
        assert result.offset == 0

    def test_pagination_offset(self) -> None:
        r1 = get_catalog(limit=2, offset=0)
        r2 = get_catalog(limit=2, offset=2)
        repos1 = {m.hf_repo for m in r1.models}
        repos2 = {m.hf_repo for m in r2.models}
        assert repos1.isdisjoint(repos2)

    def test_filter_by_task_chat(self) -> None:
        result = get_catalog(task=ModelTask.CHAT)
        assert all(m.task == "chat" for m in result.models)

    def test_filter_by_task_embedding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_services().hf_client, "fetch_models", lambda **kw: _EMPTY_HF_PAGE)
        result = get_catalog(task=ModelTask.EMBEDDING)
        assert all(m.task == "embedding" for m in result.models)
        assert result.total == len(FEATURED_EMBEDDING)

    def test_filter_by_task_vision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_services().hf_client, "fetch_models", lambda **kw: _EMPTY_HF_PAGE)
        result = get_catalog(task=ModelTask.VISION)
        assert all(m.task == "vision" for m in result.models)
        assert result.total == len(FEATURED_VISION)

    def test_search_by_name(self) -> None:
        result = get_catalog(search="Qwen3")
        for m in result.models:
            assert "qwen3" in m.display_name.lower() or "qwen3" in m.hf_repo.lower()

    def test_search_by_description(self) -> None:
        result = get_catalog(search="default for lilbee")
        assert any("nomic" in m.hf_repo.lower() for m in result.models)

    def test_search_case_insensitive(self) -> None:
        result = get_catalog(search="QWEN3")
        assert result.total > 0

    def test_search_no_results(self) -> None:
        result = get_catalog(search="nonexistent_model_xyz")
        assert result.total == 0

    def test_filter_size_small(self) -> None:
        result = get_catalog(size=CatalogSize.SMALL)
        for m in result.models:
            assert m.size_gb < 3.0

    def test_filter_size_medium(self) -> None:
        result = get_catalog(size=CatalogSize.MEDIUM)
        for m in result.models:
            assert 3.0 <= m.size_gb < 10.0

    def test_filter_size_large(self) -> None:
        result = get_catalog(size=CatalogSize.LARGE)
        for m in result.models:
            assert m.size_gb >= 10.0

    def test_filter_size_invalid_rejected(self) -> None:
        with pytest.raises(KeyError):
            get_catalog(size="gigantic")  # type: ignore[arg-type]

    def test_filter_featured_true(self) -> None:
        result = get_catalog(featured=True)
        assert all(m.featured for m in result.models)

    def test_filter_featured_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_services().hf_client, "fetch_models", lambda **kw: _EMPTY_HF_PAGE)
        result = get_catalog(featured=False)
        assert result.total == 0

    def test_sort_featured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_services().hf_client, "fetch_models", lambda **kw: _EMPTY_HF_PAGE)
        result = get_catalog(sort=CatalogSort.FEATURED)
        downloads = [m.downloads for m in result.models]
        assert downloads == sorted(downloads, reverse=True)

    def test_sort_downloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_services().hf_client, "fetch_models", lambda **kw: _EMPTY_HF_PAGE)
        result = get_catalog(sort=CatalogSort.DOWNLOADS)
        downloads = [m.downloads for m in result.models]
        assert downloads == sorted(downloads, reverse=True)

    def test_sort_size_asc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_services().hf_client, "fetch_models", lambda **kw: _EMPTY_HF_PAGE)
        result = get_catalog(sort=CatalogSort.SIZE_ASC)
        sizes = [m.size_gb for m in result.models]
        assert sizes == sorted(sizes)

    def test_sort_size_desc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_services().hf_client, "fetch_models", lambda **kw: _EMPTY_HF_PAGE)
        result = get_catalog(sort=CatalogSort.SIZE_DESC)
        sizes = [m.size_gb for m in result.models]
        assert sizes == sorted(sizes, reverse=True)

    def test_sort_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_services().hf_client, "fetch_models", lambda **kw: _EMPTY_HF_PAGE)
        result = get_catalog(sort=CatalogSort.NAME)
        names = [m.display_name.lower() for m in result.models]
        assert names == sorted(names)

    def test_installed_filter_with_model_manager(self) -> None:
        class FakeManager:
            def list_installed(self) -> list[str]:
                return ["Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"]

        result = get_catalog(installed=True, model_manager=FakeManager())
        assert all(m.hf_repo == "Qwen/Qwen3-8B-GGUF" for m in result.models)

    def test_installed_filter_not_installed(self) -> None:
        class FakeManager:
            def list_installed(self) -> list[str]:
                return ["Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"]

        result = get_catalog(installed=False, model_manager=FakeManager())
        assert all(m.hf_repo != "Qwen/Qwen3-8B-GGUF" for m in result.models)

    def test_installed_filter_manager_error(self) -> None:
        class BadManager:
            def list_installed(self) -> list[str]:
                raise RuntimeError("no manager")

        result = get_catalog(installed=True, model_manager=BadManager())
        assert result.total == 0

    def test_combines_featured_and_hf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hf_models = [
            CatalogModel(
                hf_repo="user/hf-model",
                gguf_filename="*.gguf",
                size_gb=5.0,
                min_ram_gb=8,
                description="desc",
                featured=False,
                downloads=100,
                task="chat",
            )
        ]
        monkeypatch.setattr(
            get_services().hf_client,
            "fetch_models",
            lambda **kw: HfPage(models=hf_models, has_more=False),
        )
        result = get_catalog()
        repos = [m.hf_repo for m in result.models]
        assert "user/hf-model" in repos
        assert any("Qwen3" in r for r in repos)

    def test_deduplicates_hf_against_featured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hf_models = [
            CatalogModel(
                hf_repo="Qwen/Qwen3-8B-GGUF",
                gguf_filename="*.gguf",
                size_gb=5.0,
                min_ram_gb=8,
                description="duplicate",
                featured=False,
                downloads=100,
                task="chat",
            )
        ]
        monkeypatch.setattr(
            get_services().hf_client,
            "fetch_models",
            lambda **kw: HfPage(models=hf_models, has_more=False),
        )
        result = get_catalog()
        qwen3_models = [m for m in result.models if m.hf_repo == "Qwen/Qwen3-8B-GGUF"]
        assert len(qwen3_models) == 1
        assert qwen3_models[0].featured is True

    def test_has_more_propagated_from_hf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CatalogResult.has_more reflects the HF API Link header."""
        monkeypatch.setattr(
            get_services().hf_client,
            "fetch_models",
            lambda **kw: HfPage(models=[], has_more=True),
        )
        result = get_catalog()
        assert result.has_more is True

    def test_has_more_false_when_featured_only(self) -> None:
        """Featured-only requests have has_more=False (no HF fetch)."""
        result = get_catalog(featured=True)
        assert result.has_more is False

    def test_has_more_false_when_no_more_pages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_services().hf_client, "fetch_models", lambda **kw: _EMPTY_HF_PAGE)
        result = get_catalog()
        assert result.has_more is False


class TestFindCatalogEntry:
    def test_match_by_hf_repo(self) -> None:
        result = find_catalog_entry("Qwen/Qwen3-8B-GGUF")
        assert result is not None
        assert result.hf_repo == "Qwen/Qwen3-8B-GGUF"

    def test_match_by_hf_repo_case_insensitive(self) -> None:
        result = find_catalog_entry("qwen/qwen3-8b-gguf")
        assert result is not None
        assert result.hf_repo == "Qwen/Qwen3-8B-GGUF"

    def test_not_found(self) -> None:
        result = find_catalog_entry("Nonexistent Model")
        assert result is None

    def test_empty_string(self) -> None:
        result = find_catalog_entry("")
        assert result is None

    def test_provider_prefix_stripped(self) -> None:
        """``ollama/<repo>`` resolves once the provider prefix is stripped."""
        result = find_catalog_entry("ollama/Qwen/Qwen3-8B-GGUF")
        assert result is not None
        assert result.hf_repo == "Qwen/Qwen3-8B-GGUF"

    def test_full_ref_with_concrete_filename(self) -> None:
        """A featured entry with a concrete (non-glob) filename is reachable
        via the full ``hf_repo/filename`` ref.
        """
        from lilbee.catalog.refs import format_native_gguf_ref

        # Pick a featured entry whose gguf_filename is NOT a glob.
        concrete = next(m for m in FEATURED_ALL if "*" not in m.gguf_filename)
        full_ref = format_native_gguf_ref(concrete.hf_repo, concrete.gguf_filename)
        result = find_catalog_entry(full_ref)
        assert result is not None
        assert result.hf_repo == concrete.hf_repo

    def test_non_hf_keys_return_none(self) -> None:
        """Bare names and display labels are not lookup keys."""
        assert find_catalog_entry("qwen3:0.6b") is None
        assert find_catalog_entry("qwen3") is None
        assert find_catalog_entry("Qwen3 8B") is None


class TestHfRepoFromRef:
    def test_flat_ref_yields_repo(self) -> None:
        from lilbee.catalog.refs import hf_repo_from_ref

        assert hf_repo_from_ref("Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf") == "Qwen/Qwen3-8B-GGUF"

    def test_subdir_ref_yields_first_two_segments(self) -> None:
        from lilbee.catalog.refs import hf_repo_from_ref

        ref = "unsloth/MiniMax-M2-GGUF/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
        assert hf_repo_from_ref(ref) == "unsloth/MiniMax-M2-GGUF"

    def test_bare_repo_returned_unchanged(self) -> None:
        from lilbee.catalog.refs import hf_repo_from_ref

        assert hf_repo_from_ref("Qwen/Qwen3-8B-GGUF") == "Qwen/Qwen3-8B-GGUF"

    def test_provider_prefixed_ref_returned_unchanged(self) -> None:
        from lilbee.catalog.refs import hf_repo_from_ref

        assert hf_repo_from_ref("ollama/llama3:8b") == "ollama/llama3:8b"


class TestGgufFilenameFromRef:
    def test_flat_ref_yields_filename(self) -> None:
        from lilbee.catalog.refs import gguf_filename_from_ref

        ref = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
        assert gguf_filename_from_ref(ref) == "Qwen3-8B-Q4_K_M.gguf"

    def test_subdir_ref_keeps_quant_subdir(self) -> None:
        from lilbee.catalog.refs import gguf_filename_from_ref

        ref = "unsloth/MiniMax-M2-GGUF/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
        assert gguf_filename_from_ref(ref) == "Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"

    def test_bare_repo_yields_empty(self) -> None:
        from lilbee.catalog.refs import gguf_filename_from_ref

        assert gguf_filename_from_ref("Qwen/Qwen3-8B-GGUF") == ""

    def test_provider_prefixed_ref_yields_empty(self) -> None:
        from lilbee.catalog.refs import gguf_filename_from_ref

        assert gguf_filename_from_ref("ollama/llama3:8b") == ""


class TestBuildAdhocEntry:
    def test_valid_repo_derives_defaults(self) -> None:
        entry = build_adhoc_entry("bartowski/gemma-2-2b-it-GGUF")
        assert entry.hf_repo == "bartowski/gemma-2-2b-it-GGUF"
        assert entry.gguf_filename == "*.gguf"
        assert entry.display_name == "gemma 2 2b it"
        assert entry.featured is False
        assert entry.task == ModelTask.CHAT

    def test_respects_task_override(self) -> None:
        entry = build_adhoc_entry("foo/bar-GGUF", task=ModelTask.EMBEDDING)
        assert entry.task == ModelTask.EMBEDDING

    def test_rerank_task_accepted(self) -> None:
        """Ad-hoc reranker entries preserve the RERANK task tag."""
        entry = build_adhoc_entry("foo/bar-reranker", task=ModelTask.RERANK)
        assert entry.task == ModelTask.RERANK
        assert entry.gguf_filename == "*.gguf"

    def test_explicit_gguf_filename_is_pinned(self) -> None:
        """A concrete filename (incl. a subdir) overrides the default glob."""
        entry = build_adhoc_entry(
            "unsloth/MiniMax-M2-GGUF",
            gguf_filename="Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf",
        )
        assert entry.gguf_filename == "Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"


class TestIsHfRepoId:
    @pytest.mark.parametrize(
        "value",
        ["bartowski/gemma-2-2b-it-GGUF", "Qwen/Qwen3-8B-GGUF", "foo/bar", "Foo-BAR_foo.bar123/x"],
    )
    def test_accepts_valid_repo_ids(self, value: str) -> None:
        assert _query._is_hf_repo_id(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "qwen3",
            "qwen3:0.6b",
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF",
            "datasets/foo/bar",
            "foo--bar/baz",
            "foo/bar..baz",
            "/bar",
            "foo/",
            "",
        ],
    )
    def test_rejects_non_repo_ids(self, value: str) -> None:
        assert _query._is_hf_repo_id(value) is False


class TestResolvePullTarget:
    def test_featured_hf_repo_returns_featured_entry(self) -> None:
        entry = catalog.resolve_pull_target("Qwen/Qwen3-8B-GGUF")
        assert entry is not None
        assert entry.featured is True
        assert entry.hf_repo == "Qwen/Qwen3-8B-GGUF"
        assert entry.gguf_filename != "*.gguf"

    def test_featured_hf_repo_case_insensitive(self) -> None:
        entry = catalog.resolve_pull_target("qwen/qwen3-8b-gguf")
        assert entry is not None
        assert entry.featured is True

    def test_unknown_hf_repo_builds_adhoc(self) -> None:
        entry = catalog.resolve_pull_target("bartowski/gemma-2-2b-it-GGUF")
        assert entry is not None
        assert entry.featured is False
        assert entry.hf_repo == "bartowski/gemma-2-2b-it-GGUF"
        assert entry.gguf_filename == "*.gguf"

    def test_unknown_short_name_returns_none(self) -> None:
        assert catalog.resolve_pull_target("not-a-real-model") is None

    def test_resolve_pull_target_classifies_embedding_repo(self):
        """Non-featured embedding repos must not register as chat models (bb-euk)."""
        from lilbee.catalog.types import ModelTask

        bare = catalog.resolve_pull_target("Qwen/Qwen3-Embedding-8B-GGUF")
        assert bare is not None and bare.task is ModelTask.EMBEDDING
        exact = catalog.resolve_pull_target(
            "Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf"
        )
        assert exact is not None and exact.task is ModelTask.EMBEDDING

    def test_resolve_pull_target_classifies_reranker_repo(self):
        from lilbee.catalog.types import ModelTask

        entry = catalog.resolve_pull_target("BAAI/bge-reranker-v2-m3-GGUF")
        assert entry is not None and entry.task is ModelTask.RERANK

    @pytest.mark.parametrize(
        "value",
        ["https://huggingface.co/Qwen/Qwen3-8B-GGUF", "datasets/foo/bar", "foo--bar/baz"],
    )
    def test_malformed_hf_inputs_return_none(self, value: str) -> None:
        assert catalog.resolve_pull_target(value) is None

    def test_subdir_gguf_ref_builds_adhoc_with_exact_filename(self) -> None:
        """A full subdir ref (F2) resolves to an ad-hoc entry pinning that file."""
        ref = "unsloth/MiniMax-M2-GGUF/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
        entry = catalog.resolve_pull_target(ref)
        assert entry is not None
        assert entry.featured is False
        assert entry.hf_repo == "unsloth/MiniMax-M2-GGUF"
        assert entry.gguf_filename == "Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"

    def test_flat_gguf_ref_builds_adhoc_with_exact_filename(self) -> None:
        ref = "bartowski/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q5_K_M.gguf"
        entry = catalog.resolve_pull_target(ref)
        assert entry is not None
        assert entry.hf_repo == "bartowski/gemma-2-2b-it-GGUF"
        assert entry.gguf_filename == "gemma-2-2b-it-Q5_K_M.gguf"

    def test_traversal_gguf_ref_returns_none(self) -> None:
        """A ``.gguf`` ref whose subdir path tries to escape the repo is rejected."""
        assert catalog.resolve_pull_target("owner/repo/../escape/m.gguf") is None

    def test_explicit_quant_overrides_featured_default(self) -> None:
        """F5: naming a specific .gguf on a featured repo pins that exact quant.

        The featured entry for the repo pins a default quant; an explicit
        filename must win (HF-first) instead of being overridden.
        """
        featured = catalog.resolve_pull_target("Qwen/Qwen3-8B-GGUF")
        assert featured is not None
        explicit_ref = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
        entry = catalog.resolve_pull_target(explicit_ref)
        assert entry is not None
        assert entry.gguf_filename == "Qwen3-8B-Q4_K_M.gguf"
        assert entry.gguf_filename != featured.gguf_filename


class TestSplitShardFilenames:
    def test_single_file_returns_itself(self) -> None:
        assert catalog.download.split_shard_filenames("model-Q4_K_M.gguf") == ["model-Q4_K_M.gguf"]

    def test_split_returns_every_part_in_order(self) -> None:
        parts = catalog.download.split_shard_filenames("Q4_K_M/M-Q4_K_M-00001-of-00003.gguf")
        assert parts == [
            "Q4_K_M/M-Q4_K_M-00001-of-00003.gguf",
            "Q4_K_M/M-Q4_K_M-00002-of-00003.gguf",
            "Q4_K_M/M-Q4_K_M-00003-of-00003.gguf",
        ]


class TestSplitShardDownload:
    def test_fetches_all_shards_and_finalizes_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A split GGUF pulls every shard, and on_complete fires once after the set."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: "m-00001-of-00002.gguf")
        requested: list[str] = []

        def fake(**kwargs: Any) -> str:
            requested.append(kwargs["filename"])
            return _fake_download(**kwargs)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
        completed: list[Path] = []
        download_model(entry, on_complete=lambda _e, p: completed.append(p))

        assert requested == ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
        assert len(completed) == 1  # manifest write only after the full set is on disk

    def test_xet_fallback_when_file_too_large_for_http(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 'too large for HTTP' error retries with xet enabled, then restores the flag."""
        import huggingface_hub.constants as hc

        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        monkeypatch.setattr(hc, "HF_HUB_DISABLE_XET", True)  # lilbee's default
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: e.gguf_filename)

        calls = {"n": 0}
        disable_during_retry: list[bool] = []

        def fake(**kwargs: Any) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError(
                    "The file is too large to be downloaded using the regular download method."
                )
            disable_during_retry.append(hc.HF_HUB_DISABLE_XET)
            return _fake_download(**kwargs)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
        download_model(entry)

        assert calls["n"] == 2  # original attempt + xet retry
        assert disable_during_retry == [False]  # xet was on for the retry
        assert hc.HF_HUB_DISABLE_XET is True  # flag restored afterwards

    def test_xet_flip_holds_lock_during_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The global XET flip stays under _xet_flip_lock for the whole download.

        Two overlapping xet downloads would otherwise nest their save/restore and
        leave xet permanently toggled; the lock makes the flip window exclusive.
        """
        import huggingface_hub.constants as hc

        from lilbee.catalog.download import DownloadConfig, _download_with_xet, _xet_flip_lock

        monkeypatch.setattr(hc, "HF_HUB_DISABLE_XET", True)
        locked_during: list[bool] = []

        def fake(**kwargs: Any) -> str:
            locked_during.append(_xet_flip_lock.locked())
            return str(tmp_path / "x.gguf")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
        config = DownloadConfig(repo_id="r/r", filename="x.gguf", token=None)
        _download_with_xet(config)

        assert locked_during == [True]  # lock held across the flipped window
        assert hc.HF_HUB_DISABLE_XET is True  # restored, lock released

    def test_xet_flip_restores_and_releases_on_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed xet download still restores the flag and releases the lock."""
        import huggingface_hub.constants as hc

        from lilbee.catalog.download import DownloadConfig, _download_with_xet, _xet_flip_lock

        monkeypatch.setattr(hc, "HF_HUB_DISABLE_XET", True)

        def fake(**kwargs: Any) -> str:
            raise ValueError("boom")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake)
        config = DownloadConfig(repo_id="r/r", filename="x.gguf", token=None)
        with pytest.raises(ValueError, match="boom"):
            _download_with_xet(config)

        assert hc.HF_HUB_DISABLE_XET is True  # restored even on failure
        assert not _xet_flip_lock.locked()  # lock released


class TestDownloadModel:
    def test_returns_existing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        existing = tmp_path / entry.gguf_filename
        content = b"fake model"
        existing.write_bytes(content)
        # Cached file matches HF-reported size: accepted as complete, no re-download.
        monkeypatch.setattr(
            catalog.download, "fetch_expected_file_size", lambda repo, name: len(content)
        )
        result = download_model(entry)
        assert result == existing

    def test_existing_file_calls_progress_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When model already exists, on_progress is called with 100%."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        existing = tmp_path / entry.gguf_filename
        content = b"fake model"
        existing.write_bytes(content)
        monkeypatch.setattr(
            catalog.download, "fetch_expected_file_size", lambda repo, name: len(content)
        )

        progress_calls: list[tuple[int, int]] = []

        def on_progress(downloaded: int, total: int) -> None:
            progress_calls.append((downloaded, total))

        download_model(entry, on_progress=on_progress)
        assert len(progress_calls) == 1
        assert progress_calls[0][0] == progress_calls[0][1]

    def test_tqdm_class_callback_invoked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """tqdm_class-based callback is invoked during download."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: e.gguf_filename)

        def fake_download(**kwargs: Any) -> str:
            result = _fake_download(**kwargs)
            tqdm_class = kwargs.get("tqdm_class")
            if tqdm_class:
                bar = tqdm_class(total=1000)
                bar.update(100)
                bar.update(100)
                bar.close()
            return result

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

        progress_calls: list[tuple[int, int | None]] = []

        def on_progress(downloaded: int, total: int | None) -> None:
            progress_calls.append((downloaded, total))

        download_model(entry, on_progress=on_progress)
        assert len(progress_calls) >= 1

    def test_creates_models_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        models_dir = tmp_path / "models"
        monkeypatch.setattr(cfg, "models_dir", models_dir)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: e.gguf_filename)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_download)
        result = download_model(entry)
        assert result.exists()

    def test_calls_progress_callback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: e.gguf_filename)

        progress_calls: list[tuple[int, int | None]] = []

        def fake_with_progress(**kwargs: Any) -> str:
            tqdm_class = kwargs.get("tqdm_class")
            if tqdm_class:
                bar = tqdm_class(total=1000)
                bar.update(500)
                bar.update(500)
                bar.close()
            return _fake_download(**kwargs)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_with_progress)

        def on_progress(downloaded: int, total: int | None) -> None:
            progress_calls.append((downloaded, total))

        download_model(entry, on_progress=on_progress)
        # 2 tqdm updates + 1 final 100% call from download_model
        assert len(progress_calls) == 3
        assert progress_calls[-1] == (100, 100)

    def test_gated_repo_raises_permission_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: e.gguf_filename)

        from huggingface_hub.utils import GatedRepoError

        def fake_download(**kwargs: Any) -> str:
            raise GatedRepoError("Gated repo", response=MagicMock())

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        with pytest.raises(PermissionError, match="requires HuggingFace authentication"):
            download_model(entry)

    def test_task_cancelled_propagates_unwrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TaskCancelledError from the progress callback must bubble up as-is.

        Regression test for bb-nis1: catalog.py used to wrap TaskCancelledError in
        a generic RuntimeError via its 'except Exception' block, causing
        cancelled downloads to land as FAILED instead of CANCELLED in the
        Task Center.
        """
        from lilbee.runtime.cancellation import TaskCancelledError

        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: e.gguf_filename)

        def fake_download(**kwargs: Any) -> str:
            raise TaskCancelledError

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        with pytest.raises(TaskCancelledError):
            download_model(entry)

    def test_repo_not_found_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: e.gguf_filename)

        from huggingface_hub.utils import RepositoryNotFoundError

        def fake_download(**kwargs: Any) -> str:
            raise RepositoryNotFoundError("Not found", response=MagicMock())

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        with pytest.raises(RuntimeError, match="not found on HuggingFace"):
            download_model(entry)

    def test_unexpected_exception_is_wrapped_with_type_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An error class the translator doesn't special-case is wrapped in a
        RuntimeError that names the original exception type, not leaked raw."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: e.gguf_filename)

        def fake_download(**kwargs: Any) -> str:
            raise KeyError("missing sibling")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        with pytest.raises(RuntimeError, match=r"Failed to download.*KeyError"):
            download_model(entry)


class TestResolveFilename:
    def test_exact_filename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = FEATURED_EMBEDDING[0]
        result = catalog.resolve_filename(entry)
        assert result == entry.gguf_filename

    def test_wildcard_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = FEATURED_CHAT[0]
        data = {
            "siblings": [
                {"rfilename": "Qwen3-0.6B-Q4_K_M.gguf"},
                {"rfilename": "Qwen3-0.6B-Q8_0.gguf"},
            ]
        }
        mock_resp = httpx.Response(200, json=data, request=httpx.Request("GET", "https://x"))
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        result = catalog.resolve_filename(entry)
        assert result == "Qwen3-0.6B-Q4_K_M.gguf"

    def test_wildcard_no_match_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = FEATURED_CHAT[0]
        data = {"siblings": [{"rfilename": "something-else.bin"}]}
        mock_resp = httpx.Response(200, json=data, request=httpx.Request("GET", "https://x"))
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        with pytest.raises(RuntimeError, match="No GGUF files found"):
            catalog.resolve_filename(entry)

    def test_wildcard_api_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = FEATURED_CHAT[0]

        def raise_connect(*a: object, **kw: object) -> httpx.Response:
            raise httpx.ConnectError("x")

        monkeypatch.setattr(httpx, "get", raise_connect)
        with pytest.raises(RuntimeError, match="Cannot query files"):
            catalog.resolve_filename(entry)

    def test_wildcard_http_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = FEATURED_CHAT[0]
        mock_resp = httpx.Response(500)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        with pytest.raises(RuntimeError):
            catalog.resolve_filename(entry)

    def test_wildcard_401_raises_permission_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP 401 response raises PermissionError with auth message."""
        entry = FEATURED_CHAT[0]
        mock_resp = httpx.Response(401)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        with pytest.raises(PermissionError, match="requires HuggingFace authentication"):
            catalog.resolve_filename(entry)

    def test_pick_best_gguf_prefers_q4_k_m(self) -> None:
        files = ["model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-Q5_K_M.gguf"]
        assert pick_best_gguf(files) == "model-Q4_K_M.gguf"

    def test_pick_best_gguf_fallback_first(self) -> None:
        files = ["model-weird.gguf"]
        assert pick_best_gguf(files) == "model-weird.gguf"


class TestIsBareHfRepo:
    @pytest.mark.parametrize("value", ["Qwen/Qwen3-8B-GGUF", "bartowski/SmolLM2-360M-GGUF"])
    def test_accepts_bare_repos(self, value: str) -> None:
        assert is_bare_hf_repo(value) is True

    @pytest.mark.parametrize(
        "value",
        ["Qwen/Qwen3-8B-GGUF/q4.gguf", "Qwen/file.gguf", "qwen3:0.6b", "a/b/c", ""],
    )
    def test_rejects_other_shapes(self, value: str) -> None:
        assert is_bare_hf_repo(value) is False


class TestTaskToPipeline:
    def test_chat(self) -> None:
        assert _query._task_to_pipeline("chat") == ("text-generation", None)

    def test_embedding(self) -> None:
        expected = ("feature-extraction", "sentence-transformers")
        assert _query._task_to_pipeline("embedding") == expected

    def test_vision(self) -> None:
        assert _query._task_to_pipeline("vision") == ("image-text-to-text", None)

    def test_unknown(self) -> None:
        assert _query._task_to_pipeline("unknown") == ("text-generation", None)

    def test_none(self) -> None:
        assert _query._task_to_pipeline(None) == ("text-generation", None)


class TestPipelineToTask:
    def test_text_generation(self) -> None:
        assert _query.pipeline_to_task("text-generation") == "chat"

    def test_feature_extraction(self) -> None:
        assert _query.pipeline_to_task("feature-extraction") == "embedding"

    def test_image_text_to_text(self) -> None:
        assert _query.pipeline_to_task("image-text-to-text") == "vision"

    def test_image_to_text(self) -> None:
        assert _query.pipeline_to_task("image-to-text") == "vision"

    def test_unknown_defaults_to_chat(self) -> None:
        assert _query.pipeline_to_task("unknown-tag") == "chat"

    def test_empty_defaults_to_chat(self) -> None:
        assert _query.pipeline_to_task("") == "chat"

    def test_text_ranking_maps_to_rerank(self) -> None:
        """HF's canonical cross-encoder pipeline tag is ``text-ranking``."""
        assert _query.pipeline_to_task("text-ranking") == "rerank"

    def test_text_classification_maps_to_rerank(self) -> None:
        """``text-classification`` is the HF tag used by GGUF rerankers."""
        assert _query.pipeline_to_task("text-classification") == "rerank"

    def test_sentence_similarity_maps_to_embedding(self) -> None:
        assert _query.pipeline_to_task("sentence-similarity") == "embedding"


class TestGetInstalledModels:
    def test_returns_installed_names(self) -> None:
        manager = MagicMock()
        manager.list_installed.return_value = ["a/b", "c/d"]
        assert _query._get_installed_models(manager) == {"a/b", "c/d"}

    def test_manager_failure_returns_empty_and_logs(self, caplog) -> None:
        manager = MagicMock()
        manager.list_installed.side_effect = RuntimeError("registry broken")
        with caplog.at_level("WARNING"):
            assert _query._get_installed_models(manager) == set()
        assert any("treating as none installed" in r.getMessage() for r in caplog.records)


class TestFeaturedVisionModel:
    def test_featured_vision_is_lightonocr(self) -> None:
        assert len(FEATURED_VISION) == 1
        assert "LightOnOCR" in FEATURED_VISION[0].display_name

    def test_featured_vision_is_small(self) -> None:
        assert FEATURED_VISION[0].size_gb <= 2.0


class TestSortModels:
    def test_size_asc(self) -> None:
        models = list(FEATURED_ALL)
        sorted_m = _query._sort_models(models, "size_asc")
        sizes = [m.size_gb for m in sorted_m]
        assert sizes == sorted(sizes)

    def test_size_desc(self) -> None:
        models = list(FEATURED_ALL)
        sorted_m = _query._sort_models(models, "size_desc")
        sizes = [m.size_gb for m in sorted_m]
        assert sizes == sorted(sizes, reverse=True)

    def test_downloads(self) -> None:
        models = list(FEATURED_ALL)
        sorted_m = _query._sort_models(models, "downloads")
        downloads = [m.downloads for m in sorted_m]
        assert downloads == sorted(downloads, reverse=True)

    def test_name_sort(self) -> None:
        models = list(FEATURED_ALL)
        sorted_m = _query._sort_models(models, "name")
        names = [m.display_name.lower() for m in sorted_m]
        assert names == sorted(names)

    def test_featured_default(self) -> None:
        models = list(FEATURED_ALL)
        sorted_m = _query._sort_models(models, "featured")
        assert len(sorted_m) == len(models)


class TestHfCacheEviction:
    """Tests for HfClient.fetch_models cache eviction and size cap."""

    def test_expired_entries_evicted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Expired cache entries are removed on the next fetch."""
        import time as _time

        cache = get_services().hf_client._cache

        # Seed an expired entry (timestamp 0, way older than TTL)
        cache["old:key:sort:50"] = (0.0, _EMPTY_HF_PAGE)
        # Ensure monotonic returns a time that makes the entry expired
        monkeypatch.setattr(_time, "monotonic", lambda: 1000.0)

        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.links = {}
        monkeypatch.setattr("lilbee.catalog.hf_client.httpx.get", lambda *a, **kw: mock_resp)

        get_services().hf_client.fetch_models(pipeline_tag="text-generation")
        assert "old:key:sort:50" not in cache

    def test_cache_size_capped_at_50(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When cache exceeds 50 entries, the oldest is evicted."""
        import time as _time

        cache = get_services().hf_client._cache

        base_time = 1000.0
        # Fill cache with 50 entries (timestamps 1000..1049)
        for i in range(50):
            cache[f"key:{i}"] = (base_time + i, _EMPTY_HF_PAGE)

        # Next fetch will add entry #51, triggering eviction of oldest (key:0)
        monkeypatch.setattr(_time, "monotonic", lambda: base_time + 50)

        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.links = {}
        monkeypatch.setattr("lilbee.catalog.hf_client.httpx.get", lambda *a, **kw: mock_resp)

        get_services().hf_client.fetch_models(pipeline_tag="unique")
        assert len(cache) == 50
        assert "key:0" not in cache


class TestModelVariantDataclass:
    def test_frozen(self) -> None:
        v = ModelVariant("repo", "file.gguf", "8B", "Q4_K_M", 5000, True)
        with pytest.raises(AttributeError):
            v.hf_repo = "nope"  # type: ignore[misc]

    def test_default_mmproj(self) -> None:
        v = ModelVariant("repo", "file.gguf", "8B", "Q4_K_M", 5000, False)
        assert v.mmproj_filename == ""


class TestModelFamilyDataclass:
    def test_frozen(self) -> None:
        f = ModelFamily(slug="qwen3", name="Qwen3", task="chat", description="desc", variants=())
        with pytest.raises(AttributeError):
            f.name = "nope"  # type: ignore[misc]

    def test_fields(self) -> None:
        v = ModelVariant("repo", "file.gguf", "8B", "Q4_K_M", 5000, True)
        f = ModelFamily(slug="qwen3", name="Qwen3", task="chat", description="Fast", variants=(v,))
        assert f.name == "Qwen3"
        assert f.slug == "qwen3"
        assert f.task == "chat"
        assert len(f.variants) == 1


class TestExtractFamilyName:
    def test_qwen3_8b(self) -> None:
        assert _families._extract_family_name("Qwen3 8B") == "Qwen3"

    def test_qwen3_06b(self) -> None:
        assert _families._extract_family_name("Qwen3 0.6B") == "Qwen3"

    def test_qwen3_coder(self) -> None:
        assert _families._extract_family_name("Qwen3-Coder 30B A3B") == "Qwen3 Coder"

    def test_mistral(self) -> None:
        assert _families._extract_family_name("Mistral 7B Instruct") == "Mistral"

    def test_no_space_before_version(self) -> None:
        """Names without 'space + digit' pattern return the full name."""
        assert _families._extract_family_name("Nomic Embed Text v1.5") == "Nomic Embed Text v1.5"

    def test_hyphenated_version(self) -> None:
        """Names with hyphenated versions get hyphens replaced by spaces."""
        assert _families._extract_family_name("LightOnOCR-2") == "LightOnOCR"

    def test_gguf_suffix_stripped(self) -> None:
        """GGUF suffix is stripped via clean_display_name before extraction."""
        assert _families._extract_family_name("Qwen3-8B-GGUF") == "Qwen3"

    def test_instruct_gguf_suffix_stripped(self) -> None:
        """Instruct and GGUF suffixes are stripped before extraction."""
        assert _families._extract_family_name("Mistral-7B-Instruct-GGUF") == "Mistral"

    def test_clean_display_name_applied_to_hf_names(self) -> None:
        """HF model names with repo-style suffixes produce clean family names."""
        # Simulates what _build_families sees for HF models
        assert _families._extract_family_name("Meta-Llama-3-8B-Instruct-GGUF") == "Llama"


class TestExtractQuant:
    def test_wildcard_pattern(self) -> None:
        assert catalog.extract_quant("*Q4_K_M.gguf") == "Q4_K_M"

    def test_full_filename(self) -> None:
        assert catalog.extract_quant("nomic-embed-text-v1.5.Q4_K_M.gguf") == "Q4_K_M"

    def test_q8_0(self) -> None:
        assert catalog.extract_quant("model-Q8_0.gguf") == "Q8_0"

    def test_no_quant(self) -> None:
        assert catalog.extract_quant("model.gguf") == ""


class TestGetFamilies:
    def test_returns_list(self) -> None:
        families = get_families()
        assert isinstance(families, list)
        assert all(isinstance(f, ModelFamily) for f in families)

    def test_has_chat_families(self) -> None:
        families = get_families()
        chat_families = [f for f in families if f.task == "chat"]
        assert len(chat_families) > 0

    def test_has_embedding_families(self) -> None:
        families = get_families()
        embed_families = [f for f in families if f.task == "embedding"]
        assert len(embed_families) > 0

    def test_has_vision_families(self) -> None:
        families = get_families()
        vision_families = [f for f in families if f.task == "vision"]
        assert len(vision_families) > 0

    def test_qwen3_grouped(self) -> None:
        families = get_families()
        qwen3 = [f for f in families if f.name.startswith("Qwen3") and "Coder" not in f.name]
        assert len(qwen3) == 1
        assert len(qwen3[0].variants) == 3  # 0.6B, 4B, 8B

    def test_qwen3_recommended_from_toml(self) -> None:
        families = get_families()
        qwen3 = next(f for f in families if f.name.startswith("Qwen3") and "Coder" not in f.name)
        # TOML marks qwen3:0.6b (first variant) as recommended
        assert qwen3.variants[0].recommended is True
        assert qwen3.variants[-1].recommended is False

    def test_single_variant_recommended_from_toml(self) -> None:
        """Single-variant families use recommended flag from TOML."""
        families = get_families()
        singles = [f for f in families if len(f.variants) == 1]
        # At least some single-variant families are explicitly marked recommended
        assert any(fam.variants[0].recommended for fam in singles)

    def test_total_variants_matches_featured(self) -> None:
        families = get_families()
        total_variants = sum(len(f.variants) for f in families)
        assert total_variants == len(FEATURED_ALL)

    def test_variant_has_correct_fields(self) -> None:
        families = get_families()
        qwen3 = next(f for f in families if f.name.startswith("Qwen3") and "Coder" not in f.name)
        v = qwen3.variants[0]  # 0.6B
        assert v.param_count == "0.6B"
        assert v.quant == "Q8_0"
        assert v.size_mb > 0
        assert v.hf_repo == "Qwen/Qwen3-0.6B-GGUF"

    def test_order_chat_then_embedding_then_vision(self) -> None:
        families = get_families()
        tasks = [f.task for f in families]
        # All chat tasks should come before embedding, embedding before vision
        chat_last = max(i for i, t in enumerate(tasks) if t == "chat")
        embed_first = min(i for i, t in enumerate(tasks) if t == "embedding")
        vision_first = min(i for i, t in enumerate(tasks) if t == "vision")
        assert chat_last < embed_first
        assert embed_first < vision_first


class TestVisionMmprojFiles:
    def test_all_vision_entries_have_mmproj(self) -> None:
        """Every featured vision model has an mmproj entry in VISION_MMPROJ_FILES."""
        from lilbee.catalog import VISION_MMPROJ_FILES

        for entry in FEATURED_VISION:
            assert entry.hf_repo in VISION_MMPROJ_FILES, (
                f"Vision model {entry.name} ({entry.hf_repo}) missing from VISION_MMPROJ_FILES"
            )
            assert VISION_MMPROJ_FILES[entry.hf_repo], (
                f"Vision model {entry.name} has empty mmproj pattern"
            )

    def test_download_model_calls_mmproj_for_vision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """download_model downloads mmproj file for vision entries."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_VISION[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: "model-Q4_K_M.gguf")

        download_calls: list[dict] = []

        def fake_download(**kwargs: Any) -> str:
            download_calls.append(kwargs)
            return _fake_download(**kwargs)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        monkeypatch.setattr(
            catalog.download,
            "_resolve_mmproj_filename",
            lambda repo, pat: "model-mmproj-f16.gguf",
        )

        download_model(entry)

        assert len(download_calls) == 2
        filenames = [c["filename"] for c in download_calls]
        assert "model-Q4_K_M.gguf" in filenames
        assert "model-mmproj-f16.gguf" in filenames

    def test_download_model_skips_mmproj_for_chat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """download_model does NOT download mmproj for chat entries."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: e.gguf_filename)

        download_calls: list[dict] = []

        def fake_download(**kwargs: Any) -> str:
            download_calls.append(kwargs)
            return _fake_download(**kwargs)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        download_model(entry)

        assert len(download_calls) == 1

    def test_download_model_vision_mmproj_resolution_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When mmproj resolution fails, model download still succeeds."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_VISION[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: "model-Q4_K_M.gguf")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_download)
        monkeypatch.setattr(catalog.download, "_resolve_mmproj_filename", lambda repo, pat: None)

        result = download_model(entry)
        assert result.exists()

    def test_download_mmproj_uses_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """mmproj downloads go through the HF cache tree, not a flat local_dir.
        Regression guard: previously _download_mmproj used ``local_dir=`` while
        the main download used ``cache_dir=``, producing two incompatible
        storage layouts under ``cfg.models_dir``.
        """
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_VISION[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: "model-Q4_K_M.gguf")

        download_calls: list[dict] = []

        def fake_download(**kwargs: Any) -> str:
            download_calls.append(kwargs)
            return _fake_download(**kwargs)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        monkeypatch.setattr(
            catalog.download,
            "_resolve_mmproj_filename",
            lambda repo, pat: "model-mmproj-f16.gguf",
        )

        download_model(entry)

        mmproj_calls = [c for c in download_calls if "mmproj" in c.get("filename", "")]
        assert len(mmproj_calls) == 1
        assert "cache_dir" in mmproj_calls[0]
        assert "local_dir" not in mmproj_calls[0]
        assert mmproj_calls[0]["cache_dir"] == str(tmp_path)


class TestVisionMmprojFallback:
    def test_unmapped_vision_model_uses_default_pattern(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A vision model not in VISION_MMPROJ_FILES still gets mmproj via default pattern."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        custom_entry = CatalogModel(
            hf_repo="user/CustomVision-1B-GGUF",
            gguf_filename="*Q4_K_M.gguf",
            size_gb=1.0,
            min_ram_gb=4,
            description="Custom vision model",
            featured=True,
            downloads=0,
            task="vision",
        )
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: "custom-Q4_K_M.gguf")

        download_calls: list[dict] = []

        def fake_download(**kwargs: Any) -> str:
            download_calls.append(kwargs)
            return _fake_download(**kwargs)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        monkeypatch.setattr(
            catalog.download,
            "_resolve_mmproj_filename",
            lambda repo, pat: "custom-mmproj-f16.gguf",
        )

        download_model(custom_entry)

        assert len(download_calls) == 2
        filenames = [c["filename"] for c in download_calls]
        assert "custom-Q4_K_M.gguf" in filenames
        assert "custom-mmproj-f16.gguf" in filenames

    def test_mmproj_cache_hit_fires_progress_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When HF returns a cached mmproj (no tqdm invocation) the callback still fires.
        Regression guard for the ``not tracker.was_used`` cache-hit branch in
        ``_download_mmproj``: without it, callers see 0% progress for the
        mmproj leg even though the file is fully present on disk.
        """
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_VISION[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: "model-Q4_K_M.gguf")
        monkeypatch.setattr(
            catalog.download,
            "_resolve_mmproj_filename",
            lambda repo, pat: "model-mmproj-f16.gguf",
        )

        # Pre-stage a cached blob so hf_hub_download resolves to it without
        # invoking the tqdm callback (simulating HF's cache-hit path).
        cached_blob = tmp_path / "cached-blob.bin"
        cached_blob.write_bytes(b"mmproj-payload-42")

        def fake_download(**kwargs: Any) -> str:
            if "mmproj" in kwargs.get("filename", ""):
                return str(cached_blob)
            return _fake_download(**kwargs)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

        calls: list[tuple[int, int]] = []
        download_model(entry, on_progress=lambda d, t: calls.append((d, t)))

        # The mmproj leg should have fired at least one callback at 100%.
        cached_size = cached_blob.stat().st_size
        assert (cached_size, cached_size) in calls


class TestFindMmprojFile:
    def test_returns_none_for_unknown_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-vision chat model names must not inherit another model's mmproj.
        Regression: a generic fallback used to return any ``*mmproj*.gguf`` under
        ``models_dir``, causing ``get_capabilities('qwen3:8b')`` to report 'vision'
        whenever any vision model was installed.
        """
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        # Simulate a LightOnOCR mmproj present in the cache from a prior install.
        (tmp_path / "model-mmproj-f16.gguf").write_bytes(b"fake")

        from lilbee.catalog import find_mmproj_file

        assert find_mmproj_file("qwen3:8b") is None
        assert find_mmproj_file("anything") is None

    def test_returns_none_when_no_mmproj(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cfg, "models_dir", tmp_path)

        from lilbee.catalog import find_mmproj_file

        result = find_mmproj_file("LightOnOCR-2")
        assert result is None

    def test_returns_none_when_dir_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cfg, "models_dir", tmp_path / "nonexistent")

        from lilbee.catalog import find_mmproj_file

        result = find_mmproj_file("LightOnOCR-2")
        assert result is None

    @staticmethod
    def _write_repo_mmproj(models_dir: Path, hf_repo: str, filename: str) -> Path:
        """Write *filename* into *hf_repo*'s HF cache subtree, as hf_hub_download would."""
        snapshot = models_dir / f"models--{hf_repo.replace('/', '--')}" / "snapshots" / "rev0"
        snapshot.mkdir(parents=True, exist_ok=True)
        mmproj = snapshot / filename
        mmproj.write_bytes(b"fake")
        return mmproj

    def test_finds_mmproj_with_fnmatch_pattern(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """find_mmproj_file matches using VISION_MMPROJ_FILES patterns."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)

        # Test with LightOnOCR-2 (featured vision model), in its own cache subtree.
        mmproj = self._write_repo_mmproj(
            tmp_path, "noctrex/LightOnOCR-2-1B-GGUF", "model-mmproj-f16.gguf"
        )

        from lilbee.catalog import find_mmproj_file

        result = find_mmproj_file("LightOnOCR-2")
        assert result == mmproj

    def test_finds_mmproj_via_hf_repo_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """find_mmproj_file also matches against hf_repo."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)

        mmproj = self._write_repo_mmproj(
            tmp_path, "noctrex/LightOnOCR-2-1B-GGUF", "model-mmproj-f16.gguf"
        )

        from lilbee.catalog import find_mmproj_file

        # Match against hf_repo instead of display name
        result = find_mmproj_file("noctrex/LightOnOCR-2-1B-GGUF")
        assert result == mmproj

    def test_does_not_return_other_repos_mmproj(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An mmproj that belongs to a different repo must not be returned for a
        repo whose own mmproj isn't present (cross-contamination guard)."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        # A different vision repo's mmproj sits in the cache, but the featured
        # LightOnOCR repo has none of its own.
        self._write_repo_mmproj(tmp_path, "someone/other-vision-GGUF", "mmproj-f16.gguf")

        from lilbee.catalog import find_mmproj_file

        assert find_mmproj_file("noctrex/LightOnOCR-2-1B-GGUF") is None

    def test_returns_none_when_repo_cache_has_no_mmproj(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The matched repo's cache exists but holds only a non-mmproj GGUF, so
        the scoped walk finds nothing and returns None."""
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        self._write_repo_mmproj(tmp_path, "noctrex/LightOnOCR-2-1B-GGUF", "model-Q4_K_M.gguf")

        from lilbee.catalog import find_mmproj_file

        assert find_mmproj_file("noctrex/LightOnOCR-2-1B-GGUF") is None


class TestResolveMmprojFilename:
    def test_exact_filename_passthrough(self) -> None:
        result = _download._resolve_mmproj_filename("repo", "exact-mmproj.gguf")
        assert result == "exact-mmproj.gguf"

    def test_wildcard_resolves_via_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {
            "siblings": [
                {"rfilename": "model-Q4_K_M.gguf"},
                {"rfilename": "model-mmproj-f16.gguf"},
                {"rfilename": "model-mmproj-f32.gguf"},
            ]
        }
        mock_resp = httpx.Response(
            200, json=data, request=httpx.Request("GET", "https://example.com")
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)

        result = _download._resolve_mmproj_filename("repo", "*mmproj*.gguf")
        # Prefers f16 over f32
        assert result == "model-mmproj-f16.gguf"

    def test_returns_none_on_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_error(*a, **kw):
            raise RuntimeError("network error")

        monkeypatch.setattr(httpx, "get", raise_error)
        result = _download._resolve_mmproj_filename("repo", "*mmproj*.gguf")
        assert result is None

    def test_returns_none_when_no_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {"siblings": [{"rfilename": "model-Q4_K_M.gguf"}]}
        mock_resp = httpx.Response(
            200, json=data, request=httpx.Request("GET", "https://example.com")
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)

        result = _download._resolve_mmproj_filename("repo", "*mmproj*.gguf")
        assert result is None

    def test_returns_first_when_no_f16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no f16 file exists, returns first match."""
        data = {
            "siblings": [
                {"rfilename": "model-mmproj-f32.gguf"},
                {"rfilename": "model-mmproj.bin"},
            ]
        }
        mock_resp = httpx.Response(
            200, json=data, request=httpx.Request("GET", "https://example.com")
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)

        result = _download._resolve_mmproj_filename("repo", "*mmproj*.gguf")
        assert result == "model-mmproj-f32.gguf"


class TestHfModelsSearchFilter:
    """HF API uses search=GGUF to find GGUF repos."""

    def test_returns_all_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [
            {"id": "user/model-GGUF", "downloads": 1000, "siblings": []},
            {"id": "user/another-GGUF", "downloads": 500, "siblings": []},
        ]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        page = get_services().hf_client.fetch_models()
        models = page.models
        assert len(models) == 2


class TestHfSearchValue:
    """Helper that joins the user's query onto the GGUF filter for HF ``search=``."""

    def test_empty_query_returns_only_gguf_filter(self) -> None:
        assert _hf_client._hf_search_value("") == "GGUF"

    def test_single_term_space_joined_after_gguf(self) -> None:
        assert _hf_client._hf_search_value("qwen3") == "GGUF qwen3"

    def test_whitespace_split_collapses_into_single_string(self) -> None:
        assert _hf_client._hf_search_value("qwen3 8b  instruct") == "GGUF qwen3 8b instruct"


class TestFetchHfModelsSearchForwarding:
    """User search text reaches HF as one space-joined ``search=`` value."""

    def test_search_value_sent_as_single_query_param(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_params: list[httpx.QueryParams] = []
        mock_resp = httpx.Response(200, json=[])

        def capture_get(url: str, **kwargs: Any) -> httpx.Response:
            captured_params.append(kwargs["params"])
            return mock_resp

        monkeypatch.setattr(httpx, "get", capture_get)
        get_services().hf_client.fetch_models(search="qwen3 8b")
        assert captured_params[0].get_list("search") == ["GGUF qwen3 8b"]

    def test_empty_search_still_sends_gguf_term(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_params: list[httpx.QueryParams] = []
        mock_resp = httpx.Response(200, json=[])

        def capture_get(url: str, **kwargs: Any) -> httpx.Response:
            captured_params.append(kwargs["params"])
            return mock_resp

        monkeypatch.setattr(httpx, "get", capture_get)
        get_services().hf_client.fetch_models()
        assert captured_params[0].get_list("search") == ["GGUF"]

    def test_different_search_terms_do_not_collide_in_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cache key must include the search terms so distinct queries aren't aliased."""
        calls = 0

        def capture_get(*a: object, **kw: object) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=[])

        monkeypatch.setattr(httpx, "get", capture_get)
        get_services().hf_client.fetch_models(search="qwen")
        get_services().hf_client.fetch_models(search="llama")
        # A third call with the first term should be served from cache.
        get_services().hf_client.fetch_models(search="qwen")
        assert calls == 2

    def test_get_catalog_forwards_search_to_hf_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The top-level catalog API must pass search through to the HF fetcher."""
        captured_kwargs: dict[str, Any] = {}

        def fake_fetch(**kwargs: Any) -> _models.HfPage:
            captured_kwargs.update(kwargs)
            return _EMPTY_HF_PAGE

        monkeypatch.setattr(get_services().hf_client, "fetch_models", fake_fetch)
        get_catalog(search="llama3", featured=False)
        assert captured_kwargs.get("search") == "llama3"


class TestGatedRepoShowsLoginMessage:
    def test_permission_error_mentions_login(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = FEATURED_VISION[0]
        monkeypatch.setattr(catalog.download, "resolve_filename", lambda e: e.gguf_filename)

        from huggingface_hub.utils import GatedRepoError

        def fake_download(**kwargs: Any) -> str:
            raise GatedRepoError("Gated repo", response=MagicMock())

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        with pytest.raises(PermissionError, match="requires HuggingFace authentication"):
            download_model(entry)


class TestCleanDisplayName:
    def test_strips_org_and_gguf(self) -> None:
        assert clean_display_name("Qwen/Qwen2.5-7B-Instruct-GGUF") == "Qwen2.5 7B"

    def test_strips_meta_prefix(self) -> None:
        assert clean_display_name("meta-llama/Meta-Llama-3-8B") == "Llama 3 8B"

    def test_strips_chat_suffix(self) -> None:
        assert clean_display_name("org/Model-7B-Chat-GGUF") == "Model 7B"

    def test_strips_date_suffix(self) -> None:
        assert clean_display_name("org/Model-7B-2507") == "Model 7B"

    def test_no_org_prefix(self) -> None:
        assert clean_display_name("Model-7B-GGUF") == "Model 7B"

    def test_plain_name(self) -> None:
        assert clean_display_name("org/SimpleModel") == "SimpleModel"

    def test_multiple_suffixes(self) -> None:
        result = clean_display_name("org/Model-7B-Instruct-GGUF")
        assert result == "Model 7B"

    def test_mistral_instruct(self) -> None:
        result = clean_display_name("mistralai/Mistral-7B-Instruct-v0.3-GGUF")
        assert result == "Mistral 7B v0.3"

    def test_strips_qat_marker(self) -> None:
        assert clean_display_name("unsloth/embeddinggemma-300M-qat-GGUF") == "embeddinggemma 300M"

    def test_strips_embedding_suffix(self) -> None:
        assert clean_display_name("ggml-org/all-MiniLM-L6-v2-Embedding-GGUF") == "all MiniLM L6 v2"

    def test_strips_trailing_quant(self) -> None:
        assert clean_display_name("ggml-org/all-MiniLM-L6-v2-Q8_0") == "all MiniLM L6 v2"

    def test_strips_combined_quant_and_qat(self) -> None:
        assert (
            clean_display_name("unsloth/embeddinggemma-300M-qat-Q8_0-GGUF") == "embeddinggemma 300M"
        )


class TestDownloadTaskName:
    """download_task_name yields the same string as ``CatalogModel.display_name``."""

    def test_repo_ref_matches_catalog_display_name(self) -> None:
        from lilbee.catalog import CatalogModel, download_task_name

        model = CatalogModel(
            hf_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            gguf_filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
            size_gb=0.4,
            min_ram_gb=2.0,
            description="",
            featured=True,
            downloads=0,
            task=ModelTask.CHAT,
        )
        assert download_task_name(model.hf_repo) == model.display_name

    def test_native_gguf_ref_strips_filename(self) -> None:
        from lilbee.catalog import download_task_name

        ref = "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        assert download_task_name(ref) == "Qwen2.5 0.5B"

    def test_empty_and_slashless_strings_return_empty(self) -> None:
        from lilbee.catalog import download_task_name

        assert download_task_name("") == ""
        assert download_task_name("ollama") == ""
        assert download_task_name("openai") == ""

    def test_single_slash_gguf_ref_strips_to_empty(self) -> None:
        """A malformed ``<file>.gguf`` ref with only one slash cleans to empty."""
        from lilbee.catalog import download_task_name

        assert download_task_name("foo/file.gguf") == ""

    def test_subdir_quant_ref_uses_repo_not_subdir(self) -> None:
        """A subdir-quant giant ref labels by its repo, not ``repo/Q4_K_M``. (F2)"""
        from lilbee.catalog import download_task_name

        ref = "unsloth/MiniMax-M2-GGUF/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
        assert download_task_name(ref) == "MiniMax M2"


class TestDisplayLabelForRef:
    def test_native_hf_ref_uses_clean_repo_name(self) -> None:
        from lilbee.catalog import display_label_for_ref

        ref = "Qwen/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        assert display_label_for_ref(ref) == "Qwen2.5 7B"

    def test_subdir_quant_ref_uses_repo_not_subdir(self) -> None:
        """A subdir-quant giant ref labels by its repo, matching download_task_name. (F2)"""
        from lilbee.catalog import display_label_for_ref

        ref = "unsloth/MiniMax-M2-GGUF/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
        assert display_label_for_ref(ref) == "MiniMax M2"

    def test_ollama_prefix_drops_only_the_prefix(self) -> None:
        from lilbee.catalog import display_label_for_ref

        assert display_label_for_ref("ollama/qwen3:0.6b") == "qwen3:0.6b"

    def test_openai_prefix_drops_only_the_prefix(self) -> None:
        from lilbee.catalog import display_label_for_ref

        assert display_label_for_ref("openai/gpt-4o") == "gpt-4o"

    def test_empty_string(self) -> None:
        from lilbee.catalog import display_label_for_ref

        assert display_label_for_ref("") == ""

    def test_unrecognized_shape_passes_through(self) -> None:
        """Bare names with no '/' are returned unchanged so the picker
        still has something to show even for stale config values."""
        from lilbee.catalog import display_label_for_ref

        assert display_label_for_ref("qwen3:0.6b") == "qwen3:0.6b"


class TestQuantTier:
    def test_all_quant_types_mapped(self) -> None:
        for quant_name, expected_tier in QUANT_TIERS.items():
            assert quant_tier(quant_name) == expected_tier

    def test_empty_returns_dash(self) -> None:
        assert quant_tier("") == "--"

    def test_unknown_returns_dash(self) -> None:
        assert quant_tier("WEIRD_QUANT") == "--"

    def test_compact_tiers(self) -> None:
        for q in ("Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L"):
            assert quant_tier(q) == "compact"

    def test_balanced_tiers(self) -> None:
        for q in ("Q4_K_S", "Q4_K_M", "Q4_0"):
            assert quant_tier(q) == "balanced"

    def test_high_quality_tiers(self) -> None:
        for q in ("Q5_K_S", "Q5_K_M", "Q6_K"):
            assert quant_tier(q) == "high quality"

    def test_full_precision(self) -> None:
        assert quant_tier("Q8_0") == "full precision"

    def test_unquantized(self) -> None:
        assert quant_tier("F16") == "unquantized"
        assert quant_tier("F32") == "unquantized"


class TestEnrichCatalog:
    def _make_result(self) -> CatalogResult:
        models = [
            CatalogModel(
                hf_repo="user/Model-7B-Instruct-GGUF",
                gguf_filename="*Q4_K_M.gguf",
                size_gb=4.0,
                min_ram_gb=8.0,
                description="A test model",
                featured=False,
                downloads=1000,
                task="chat",
            ),
            CatalogModel(
                hf_repo="Qwen/Qwen3-8B-GGUF",
                gguf_filename="*Q4_K_M.gguf",
                size_gb=5.0,
                min_ram_gb=8.0,
                description="Strong general purpose",
                featured=True,
                downloads=0,
                task="chat",
            ),
        ]
        return CatalogResult(total=2, limit=20, offset=0, models=models)

    def test_returns_enriched_models(self) -> None:
        result = self._make_result()
        enriched = enrich_catalog(result, set())
        assert len(enriched) == 2
        assert all(isinstance(e, EnrichedModel) for e in enriched)

    def test_display_name_populated(self) -> None:
        result = self._make_result()
        enriched = enrich_catalog(result, set())
        # Display name is derived from clean_display_name(hf_repo), which
        # strips -Instruct/-GGUF and replaces dashes with spaces.
        assert enriched[0].display_name == "Model 7B"
        assert enriched[1].display_name == "Qwen3 8B"

    def test_quality_tier_populated(self) -> None:
        result = self._make_result()
        enriched = enrich_catalog(result, set())
        assert enriched[0].quality_tier == "balanced"

    def test_installed_status(self) -> None:
        result = self._make_result()
        # installed_refs are full hf_repo/filename refs from the registry.
        enriched = enrich_catalog(result, {"Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"})
        assert enriched[0].installed is False
        assert enriched[0].source == "native"
        assert enriched[1].installed is True
        assert enriched[1].source == "native"

    def test_source_is_native_regardless_of_installed_names(self) -> None:
        result = self._make_result()
        enriched = enrich_catalog(
            result,
            {
                "user/Model-7B-Instruct-GGUF/model-Q4_K_M.gguf",
                "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
            },
        )
        assert all(e.source == "native" for e in enriched)
        assert enriched[0].installed is True
        assert enriched[1].installed is True

    def test_preserves_original_fields(self) -> None:
        result = self._make_result()
        enriched = enrich_catalog(result, set())
        original = result.models[0]
        e = enriched[0]
        assert e.hf_repo == original.hf_repo
        assert e.gguf_filename == original.gguf_filename
        assert e.size_gb == original.size_gb
        assert e.description == original.description
        assert e.featured == original.featured
        assert e.downloads == original.downloads
        assert e.task == original.task

    def test_param_count_extracted_from_display_name(self) -> None:
        result = self._make_result()
        enriched = enrich_catalog(result, set())
        # "Model 7B" (cleaned from user/Model-7B-Instruct-GGUF) -> "7B";
        # "Qwen3 8B" -> "8B".
        assert enriched[0].param_count == "7B"
        assert enriched[1].param_count == "8B"

    def test_param_count_empty_when_no_numeric_suffix(self) -> None:
        """Embedding models like Nomic Embed Text v1.5 have no NB suffix."""
        result = CatalogResult(
            total=1,
            limit=20,
            offset=0,
            models=[
                CatalogModel(
                    hf_repo="nomic-ai/nomic-embed-text-v1.5-GGUF",
                    gguf_filename="*Q4_K_M.gguf",
                    size_gb=0.3,
                    min_ram_gb=1.0,
                    description="Embedding model",
                    featured=True,
                    downloads=42,
                    task="embedding",
                )
            ],
        )
        enriched = enrich_catalog(result, set())
        assert enriched[0].param_count == ""


class TestFormatSizeMb:
    def test_zero_returns_dash(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import _format_size_mb

        assert _format_size_mb(0) == "--"

    def test_mb_value(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import _format_size_mb

        assert _format_size_mb(512) == "512 MB"

    def test_gb_value(self) -> None:
        from lilbee.cli.tui.screens.catalog_utils import _format_size_mb

        assert _format_size_mb(2048) == "2.0 GB"


class TestDownloadProgressCallback:
    """Tests for tqdm_class-based progress callback infrastructure."""

    def test_tqdm_class_param_exists_in_hf_hub_download(self) -> None:
        """Verify huggingface_hub accepts tqdm_class parameter."""
        import inspect

        from huggingface_hub import hf_hub_download

        sig = inspect.signature(hf_hub_download)
        assert "tqdm_class" in sig.parameters

    def test_download_config_has_tqdm_class_field(self) -> None:
        """Verify DownloadConfig accepts tqdm_class."""
        from lilbee.catalog import DownloadConfig
        from lilbee.catalog.download_progress import _ProgressTracker

        tracker = _ProgressTracker(lambda x, y: None)
        config = DownloadConfig(
            repo_id="test/test",
            filename="test.gguf",
            token="test",
            tqdm_class=tracker.make_tqdm_class(),
        )
        assert config.tqdm_class is not None

    def test_callback_tqdm_class_forwards_updates(self) -> None:
        """Verify _ProgressTracker tqdm class forwards updates to callback."""
        from lilbee.catalog.download_progress import _ProgressTracker

        calls: list[tuple[int, int]] = []

        def user_callback(downloaded: int, total: int) -> None:
            calls.append((downloaded, total))

        tracker = _ProgressTracker(user_callback)
        cls = tracker.make_tqdm_class()
        bar = cls(total=200)
        bar.update(100)
        bar.update(100)
        bar.close()

        assert calls == [(100, 200), (200, 200)]
        assert tracker.was_used is True


class TestRegisterDownloadedModel:
    def _entry(self) -> object:
        from lilbee.catalog import CatalogModel

        return CatalogModel(
            hf_repo="user/test",
            gguf_filename="test.gguf",
            size_gb=1.0,
            min_ram_gb=2,
            description="test",
            featured=False,
            downloads=0,
            task="chat",
        )

    def test_writes_manifest_on_success(self, tmp_path: Path) -> None:
        """register_downloaded_model writes a manifest readable via the registry."""
        from lilbee.core.config import cfg
        from lilbee.modelhub.registry import ModelRegistry, register_downloaded_model

        file_path = tmp_path / "test.gguf"
        file_path.write_bytes(b"fake model bytes")

        old = cfg.models_dir
        cfg.models_dir = tmp_path
        try:
            register_downloaded_model(self._entry(), file_path)
            installed = ModelRegistry(tmp_path).list_installed()
        finally:
            cfg.models_dir = old

        refs = [m.ref for m in installed]
        assert "user/test/test.gguf" in refs

    def _seed_hf_cache(self, models_dir: Path, content: bytes = b"fake") -> Path:
        """Lay down a faithful HuggingFace cache entry; return the snapshot file path."""
        import hashlib

        from lilbee.modelhub.registry import repo_to_dir

        rev = "0" * 40
        cache = models_dir / f"models--{repo_to_dir('user/test')}"
        (cache / "blobs").mkdir(parents=True)
        blob = cache / "blobs" / hashlib.sha256(content).hexdigest()
        blob.write_bytes(content)
        snap = cache / "snapshots" / rev
        snap.mkdir(parents=True)
        (snap / "test.gguf").symlink_to(blob)
        (cache / "refs").mkdir(parents=True)
        (cache / "refs" / "main").write_text(rev)
        return snap / "test.gguf"

    def test_manifest_write_failure_recovers_via_cache(self, tmp_path: Path) -> None:
        """A manifest-write hiccup is logged, not raised: the bytes are in the HF cache."""
        from lilbee.core.config import cfg
        from lilbee.modelhub.registry import ModelRegistry, register_downloaded_model

        file_path = self._seed_hf_cache(tmp_path)
        old_models_dir = cfg.models_dir
        cfg.models_dir = tmp_path
        try:
            with patch(
                "lilbee.modelhub.registry.ModelRegistry._write_manifest",
                side_effect=OSError("disk full"),
            ):
                register_downloaded_model(self._entry(), file_path)  # must not raise
            assert ModelRegistry(tmp_path).is_installed("user/test/test.gguf")
        finally:
            cfg.models_dir = old_models_dir

    def test_broken_download_re_raises(self, tmp_path: Path) -> None:
        """If the GGUF isn't even in the cache the download is broken; the failure propagates."""
        from lilbee.core.config import cfg
        from lilbee.modelhub.registry import register_downloaded_model

        file_path = tmp_path / "test.gguf"
        file_path.write_bytes(b"fake")  # not in the HF cache layout
        old_models_dir = cfg.models_dir
        cfg.models_dir = tmp_path
        try:
            with (
                patch(
                    "lilbee.modelhub.registry.ModelRegistry.install",
                    side_effect=RuntimeError("disk full"),
                ),
                pytest.raises(RuntimeError, match="disk full"),
            ):
                register_downloaded_model(self._entry(), file_path)
        finally:
            cfg.models_dir = old_models_dir

    def test_download_model_invokes_on_complete(self, tmp_path: Path) -> None:
        """download_model calls on_complete(entry, file_path) after the bytes land."""
        from lilbee.catalog.download import download_model
        from lilbee.core.config import cfg

        entry = self._entry()
        existing = tmp_path / entry.gguf_filename
        existing.write_bytes(b"fake")

        old = cfg.models_dir
        cfg.models_dir = tmp_path
        captured: list[tuple[object, Path]] = []
        try:
            download_model(
                entry,
                on_complete=lambda e, p: captured.append((e, p)),
            )
        finally:
            cfg.models_dir = old

        assert captured == [(entry, existing)]
