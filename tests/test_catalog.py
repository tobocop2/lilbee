"""Tests for catalog.py — model catalog, HF API fetching, filtering, downloading."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from lilbee import catalog
from lilbee.catalog import (
    FEATURED_ALL,
    FEATURED_CHAT,
    FEATURED_EMBEDDING,
    FEATURED_VISION,
    QUANT_TIERS,
    CatalogModel,
    CatalogResult,
    EnrichedModel,
    ModelFamily,
    ModelVariant,
    clean_display_name,
    download_model,
    enrich_catalog,
    find_catalog_entry,
    get_catalog,
    get_families,
    quant_tier,
)


@pytest.fixture(autouse=True)
def _clear_hf_cache():
    """Clear the HuggingFace API cache between tests."""
    catalog._hf_cache.clear()
    yield
    catalog._hf_cache.clear()


class TestCatalogModelDataclass:
    def test_frozen(self) -> None:
        m = FEATURED_CHAT[0]
        with pytest.raises(AttributeError):
            m.name = "nope"  # type: ignore[misc]

    def test_fields(self) -> None:
        m = FEATURED_CHAT[0]
        assert isinstance(m.name, str)
        assert isinstance(m.hf_repo, str)
        assert isinstance(m.gguf_filename, str)
        assert isinstance(m.size_gb, (int, float))
        assert isinstance(m.min_ram_gb, (int, float))
        assert isinstance(m.description, str)
        assert isinstance(m.featured, bool)
        assert isinstance(m.downloads, int)
        assert isinstance(m.task, str)


class TestCatalogResultDataclass:
    def test_frozen(self) -> None:
        r = CatalogResult(total=0, limit=20, offset=0, models=[])
        with pytest.raises(AttributeError):
            r.total = 1  # type: ignore[misc]


class TestFeaturedModels:
    def test_chat_not_empty(self) -> None:
        assert len(FEATURED_CHAT) > 0

    def test_embedding_not_empty(self) -> None:
        assert len(FEATURED_EMBEDDING) > 0

    def test_vision_not_empty(self) -> None:
        assert len(FEATURED_VISION) > 0

    def test_all_combined(self) -> None:
        expected = len(FEATURED_CHAT) + len(FEATURED_EMBEDDING) + len(FEATURED_VISION)
        assert len(FEATURED_ALL) == expected

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


class TestFetchHfModels:
    def _mock_hf_response(self) -> list[dict]:
        return [
            {
                "id": "user/model-7b-gguf",
                "downloads": 5000,
                "description": "A test model",
                "siblings": [
                    {"rfilename": "model-7b-Q4_K_M.gguf", "size": 4_000_000_000},
                    {"rfilename": "model-7b-Q8_0.gguf", "size": 7_000_000_000},
                ],
            },
            {
                "id": "user/model-13b-gguf",
                "downloads": 1000,
                "description": "",
                "siblings": [
                    {"rfilename": "model-13b-Q4_K_M.gguf", "size": 0},
                ],
            },
        ]

    def test_parses_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = httpx.Response(200, json=self._mock_hf_response())

        def mock_get(*args: object, **kwargs: object) -> httpx.Response:
            return mock_resp

        monkeypatch.setattr(httpx, "get", mock_get)
        models = catalog._fetch_hf_models()
        assert len(models) == 2
        assert models[0].name == "model-7b-gguf"
        assert models[0].hf_repo == "user/model-7b-gguf"
        assert models[0].downloads == 5000
        assert models[0].featured is False
        assert models[0].task == "chat"

    def test_estimates_size_from_largest_gguf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = httpx.Response(200, json=self._mock_hf_response())
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = catalog._fetch_hf_models()
        # Largest GGUF file is 7GB -> ~6.5 GB estimate
        assert 6.0 < models[0].size_gb < 7.5

    def test_no_gguf_size_info_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = httpx.Response(200, json=self._mock_hf_response())
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = catalog._fetch_hf_models()
        # Second model has gguf sibling with size=0 -> fallback 0.0 (unknown)
        assert models[1].size_gb == 0.0

    def test_skips_entries_without_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [{"id": "", "downloads": 0}, {"downloads": 0}]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = catalog._fetch_hf_models()
        assert len(models) == 0

    def test_http_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_get(*a: object, **kw: object) -> httpx.Response:
            raise httpx.ConnectError("fail")

        monkeypatch.setattr(httpx, "get", mock_get)
        models = catalog._fetch_hf_models()
        assert models == []

    def test_invalid_json_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_get(*a: object, **kw: object) -> httpx.Response:
            raise ValueError("bad json")

        monkeypatch.setattr(httpx, "get", mock_get)
        models = catalog._fetch_hf_models()
        assert models == []

    def test_http_status_error_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = httpx.Response(500)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = catalog._fetch_hf_models()
        assert models == []

    def test_truncates_long_description(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [
            {
                "id": "user/test",
                "downloads": 0,
                "description": "A" * 200,
                "siblings": [{"rfilename": "model.gguf"}],
            }
        ]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = catalog._fetch_hf_models()
        assert len(models[0].description) <= 120

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
        models = catalog._fetch_hf_models()
        assert models[0].task == "embedding"
        assert models[1].task == "vision"

    def test_missing_pipeline_tag_defaults_to_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [{"id": "user/model", "downloads": 100, "siblings": [{"rfilename": "m.gguf"}]}]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = catalog._fetch_hf_models()
        assert models[0].task == "chat"


class TestGetCatalog:
    def test_returns_featured_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: [])
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
        names1 = {m.name for m in r1.models}
        names2 = {m.name for m in r2.models}
        assert names1.isdisjoint(names2)

    def test_filter_by_task_chat(self) -> None:
        result = get_catalog(task="chat")
        assert all(m.task == "chat" for m in result.models)

    def test_filter_by_task_embedding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: [])
        result = get_catalog(task="embedding")
        assert all(m.task == "embedding" for m in result.models)
        assert result.total == len(FEATURED_EMBEDDING)

    def test_filter_by_task_vision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: [])
        result = get_catalog(task="vision")
        assert all(m.task == "vision" for m in result.models)
        assert result.total == len(FEATURED_VISION)

    def test_search_by_name(self) -> None:
        result = get_catalog(search="Qwen3")
        for m in result.models:
            assert "qwen3" in m.name.lower() or "qwen3" in m.hf_repo.lower()

    def test_search_by_description(self) -> None:
        result = get_catalog(search="default for lilbee")
        assert any("nomic" in m.name.lower() for m in result.models)

    def test_search_case_insensitive(self) -> None:
        result = get_catalog(search="QWEN3")
        assert result.total > 0

    def test_search_no_results(self) -> None:
        result = get_catalog(search="nonexistent_model_xyz")
        assert result.total == 0

    def test_filter_size_small(self) -> None:
        result = get_catalog(size="small")
        for m in result.models:
            assert m.size_gb < 3.0

    def test_filter_size_medium(self) -> None:
        result = get_catalog(size="medium")
        for m in result.models:
            assert 3.0 <= m.size_gb < 10.0

    def test_filter_size_large(self) -> None:
        result = get_catalog(size="large")
        for m in result.models:
            assert m.size_gb >= 10.0

    def test_filter_size_invalid_ignored(self) -> None:
        result_all = get_catalog()
        result_bad = get_catalog(size="gigantic")
        assert result_all.total == result_bad.total

    def test_filter_featured_true(self) -> None:
        result = get_catalog(featured=True)
        assert all(m.featured for m in result.models)

    def test_filter_featured_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: [])
        result = get_catalog(featured=False)
        assert result.total == 0

    def test_sort_featured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: [])
        result = get_catalog(sort="featured")
        downloads = [m.downloads for m in result.models]
        assert downloads == sorted(downloads, reverse=True)

    def test_sort_downloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: [])
        result = get_catalog(sort="downloads")
        downloads = [m.downloads for m in result.models]
        assert downloads == sorted(downloads, reverse=True)

    def test_sort_size_asc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: [])
        result = get_catalog(sort="size_asc")
        sizes = [m.size_gb for m in result.models]
        assert sizes == sorted(sizes)

    def test_sort_size_desc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: [])
        result = get_catalog(sort="size_desc")
        sizes = [m.size_gb for m in result.models]
        assert sizes == sorted(sizes, reverse=True)

    def test_sort_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: [])
        result = get_catalog(sort="name")
        names = [m.name.lower() for m in result.models]
        assert names == sorted(names)

    def test_installed_filter_with_model_manager(self) -> None:
        class FakeManager:
            def list_installed(self) -> list[str]:
                return ["Qwen3 8B"]

        result = get_catalog(installed=True, model_manager=FakeManager())
        assert all(m.name == "Qwen3 8B" for m in result.models)

    def test_installed_filter_not_installed(self) -> None:
        class FakeManager:
            def list_installed(self) -> list[str]:
                return ["Qwen3 8B"]

        result = get_catalog(installed=False, model_manager=FakeManager())
        assert all(m.name != "Qwen3 8B" for m in result.models)

    def test_installed_filter_manager_error(self) -> None:
        class BadManager:
            def list_installed(self) -> list[str]:
                raise RuntimeError("no manager")

        result = get_catalog(installed=True, model_manager=BadManager())
        assert result.total == 0

    def test_combines_featured_and_hf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hf_models = [
            CatalogModel("HF Model", "user/hf-model", "*.gguf", 5.0, 8, "desc", False, 100, "chat")
        ]
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: hf_models)
        result = get_catalog()
        names = [m.name for m in result.models]
        assert "HF Model" in names
        assert "Qwen3 8B" in names

    def test_deduplicates_hf_against_featured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hf_models = [
            CatalogModel(
                "Qwen3 8B",
                "Qwen/Qwen3-8B-GGUF",
                "*.gguf",
                5.0,
                8,
                "duplicate",
                False,
                100,
                "chat",
            )
        ]
        monkeypatch.setattr(catalog, "_fetch_hf_models", lambda **kw: hf_models)
        result = get_catalog()
        qwen3_models = [m for m in result.models if m.hf_repo == "Qwen/Qwen3-8B-GGUF"]
        assert len(qwen3_models) == 1
        assert qwen3_models[0].featured is True


class TestFindCatalogEntry:
    def test_exact_match(self) -> None:
        result = find_catalog_entry("Qwen3 8B")
        assert result is not None
        assert result.name == "Qwen3 8B"

    def test_case_insensitive(self) -> None:
        result = find_catalog_entry("qwen3 8b")
        assert result is not None
        assert result.name == "Qwen3 8B"

    def test_not_found(self) -> None:
        result = find_catalog_entry("Nonexistent Model")
        assert result is None

    def test_empty_string(self) -> None:
        result = find_catalog_entry("")
        assert result is None


class TestDownloadModel:
    def test_returns_existing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        existing = tmp_path / entry.gguf_filename
        existing.write_bytes(b"fake model")
        result = download_model(entry)
        assert result == existing

    def test_creates_models_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        models_dir = tmp_path / "models"
        monkeypatch.setattr(catalog.cfg, "models_dir", models_dir)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog, "_resolve_filename", lambda e: e.gguf_filename)

        def fake_download(**kwargs: Any) -> str:
            dest = Path(kwargs["local_dir"]) / kwargs["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * 100)
            return str(dest)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        result = download_model(entry)
        assert result.exists()
        assert result.parent == models_dir

    def test_calls_progress_callback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog, "_resolve_filename", lambda e: e.gguf_filename)

        progress_calls: list[tuple[int, int]] = []

        def fake_download(**kwargs: Any) -> str:
            tqdm_cls = kwargs.get("tqdm_class")
            if tqdm_cls:
                bar = tqdm_cls(total=100)
                bar.update(50)
                bar.update(50)
                bar.close()
            dest = Path(kwargs["local_dir"]) / kwargs["filename"]
            dest.write_bytes(b"x" * 100)
            return str(dest)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

        def on_progress(downloaded: int, total: int) -> None:
            progress_calls.append((downloaded, total))

        download_model(entry, on_progress=on_progress)
        assert len(progress_calls) == 2
        assert progress_calls[-1] == (100, 100)

    def test_gated_repo_raises_permission_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog, "_resolve_filename", lambda e: e.gguf_filename)

        from huggingface_hub.utils import GatedRepoError

        def fake_download(**kwargs: Any) -> str:
            raise GatedRepoError("Gated repo", response=MagicMock())

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        with pytest.raises(PermissionError, match="requires HuggingFace authentication"):
            download_model(entry)

    def test_repo_not_found_raises_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog, "_resolve_filename", lambda e: e.gguf_filename)

        from huggingface_hub.utils import RepositoryNotFoundError

        def fake_download(**kwargs: Any) -> str:
            raise RepositoryNotFoundError("Not found", response=MagicMock())

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        with pytest.raises(RuntimeError, match="not found on HuggingFace"):
            download_model(entry)


class TestResolveFilename:
    def test_exact_filename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = FEATURED_EMBEDDING[0]
        result = catalog._resolve_filename(entry)
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
        result = catalog._resolve_filename(entry)
        assert result == "Qwen3-0.6B-Q4_K_M.gguf"

    def test_wildcard_no_match_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = FEATURED_CHAT[0]
        data = {"siblings": [{"rfilename": "something-else.bin"}]}
        mock_resp = httpx.Response(200, json=data, request=httpx.Request("GET", "https://x"))
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        with pytest.raises(RuntimeError, match="No GGUF files found"):
            catalog._resolve_filename(entry)

    def test_wildcard_api_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = FEATURED_CHAT[0]

        def raise_connect(*a: object, **kw: object) -> httpx.Response:
            raise httpx.ConnectError("x")

        monkeypatch.setattr(httpx, "get", raise_connect)
        with pytest.raises(RuntimeError, match="Cannot query files"):
            catalog._resolve_filename(entry)

    def test_wildcard_http_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entry = FEATURED_CHAT[0]
        mock_resp = httpx.Response(500)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        with pytest.raises(RuntimeError):
            catalog._resolve_filename(entry)

    def test_pick_best_gguf_prefers_q4_k_m(self) -> None:
        files = ["model-Q8_0.gguf", "model-Q4_K_M.gguf", "model-Q5_K_M.gguf"]
        assert catalog._pick_best_gguf(files) == "model-Q4_K_M.gguf"

    def test_pick_best_gguf_fallback_first(self) -> None:
        files = ["model-weird.gguf"]
        assert catalog._pick_best_gguf(files) == "model-weird.gguf"


class TestTaskToPipeline:
    def test_chat(self) -> None:
        assert catalog._task_to_pipeline("chat") == "text-generation"

    def test_embedding(self) -> None:
        assert catalog._task_to_pipeline("embedding") == "feature-extraction"

    def test_vision(self) -> None:
        assert catalog._task_to_pipeline("vision") == "image-text-to-text"

    def test_unknown(self) -> None:
        assert catalog._task_to_pipeline("unknown") == "text-generation"

    def test_none(self) -> None:
        assert catalog._task_to_pipeline(None) == "text-generation"


class TestPipelineToTask:
    def test_text_generation(self) -> None:
        assert catalog._pipeline_to_task("text-generation") == "chat"

    def test_feature_extraction(self) -> None:
        assert catalog._pipeline_to_task("feature-extraction") == "embedding"

    def test_image_text_to_text(self) -> None:
        assert catalog._pipeline_to_task("image-text-to-text") == "vision"

    def test_image_to_text(self) -> None:
        assert catalog._pipeline_to_task("image-to-text") == "vision"

    def test_unknown_defaults_to_chat(self) -> None:
        assert catalog._pipeline_to_task("unknown-tag") == "chat"

    def test_empty_defaults_to_chat(self) -> None:
        assert catalog._pipeline_to_task("") == "chat"


class TestFeaturedVisionModel:
    def test_featured_vision_is_lightonocr(self) -> None:
        assert len(FEATURED_VISION) == 1
        assert "LightOnOCR" in FEATURED_VISION[0].name

    def test_featured_vision_is_small(self) -> None:
        assert FEATURED_VISION[0].size_gb <= 2.0


class TestSortModels:
    def test_size_asc(self) -> None:
        models = list(FEATURED_ALL)
        sorted_m = catalog._sort_models(models, "size_asc")
        sizes = [m.size_gb for m in sorted_m]
        assert sizes == sorted(sizes)

    def test_size_desc(self) -> None:
        models = list(FEATURED_ALL)
        sorted_m = catalog._sort_models(models, "size_desc")
        sizes = [m.size_gb for m in sorted_m]
        assert sizes == sorted(sizes, reverse=True)

    def test_downloads(self) -> None:
        models = list(FEATURED_ALL)
        sorted_m = catalog._sort_models(models, "downloads")
        downloads = [m.downloads for m in sorted_m]
        assert downloads == sorted(downloads, reverse=True)

    def test_name_sort(self) -> None:
        models = list(FEATURED_ALL)
        sorted_m = catalog._sort_models(models, "name")
        names = [m.name.lower() for m in sorted_m]
        assert names == sorted(names)

    def test_featured_default(self) -> None:
        models = list(FEATURED_ALL)
        sorted_m = catalog._sort_models(models, "featured")
        assert len(sorted_m) == len(models)


class TestFetchModelFileSize:
    def test_returns_size_from_tree_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"path": "model-Q4_K_M.gguf", "size": 5_000_000_000},
            {"path": "model-Q8_0.gguf", "size": 9_000_000_000},
            {"path": "README.md", "size": 100},
        ]
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr("lilbee.catalog.httpx.get", lambda *a, **kw: mock_resp)

        result = catalog.fetch_model_file_size("user/repo")
        assert result == round(5_000_000_000 / (1024**3), 1)

    def test_returns_zero_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "lilbee.catalog.httpx.get", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail"))
        )
        assert catalog.fetch_model_file_size("user/repo") == 0.0

    def test_returns_zero_no_gguf_files(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"path": "README.md", "size": 100}]
        mock_resp.raise_for_status = MagicMock()
        monkeypatch.setattr("lilbee.catalog.httpx.get", lambda *a, **kw: mock_resp)

        assert catalog.fetch_model_file_size("user/repo") == 0.0


class TestHfCacheEviction:
    """Tests for _fetch_hf_models cache eviction and size cap."""

    def test_expired_entries_evicted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Expired cache entries are removed on the next fetch."""
        import time as _time

        from lilbee.catalog import _hf_cache

        # Seed an expired entry (timestamp 0, way older than TTL)
        _hf_cache["old:key:sort:50"] = (0.0, [])
        # Ensure monotonic returns a time that makes the entry expired
        monkeypatch.setattr(_time, "monotonic", lambda: 1000.0)

        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        monkeypatch.setattr("lilbee.catalog.httpx.get", lambda *a, **kw: mock_resp)

        catalog._fetch_hf_models(pipeline_tag="text-generation", tags="gguf")
        assert "old:key:sort:50" not in _hf_cache

    def test_cache_size_capped_at_50(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When cache exceeds 50 entries, the oldest is evicted."""
        import time as _time

        from lilbee.catalog import _hf_cache

        base_time = 1000.0
        # Fill cache with 50 entries (timestamps 1000..1049)
        for i in range(50):
            _hf_cache[f"key:{i}"] = (base_time + i, [])

        # Next fetch will add entry #51, triggering eviction of oldest (key:0)
        monkeypatch.setattr(_time, "monotonic", lambda: base_time + 50)

        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        monkeypatch.setattr("lilbee.catalog.httpx.get", lambda *a, **kw: mock_resp)

        catalog._fetch_hf_models(pipeline_tag="unique", tags="gguf")
        assert len(_hf_cache) == 50
        assert "key:0" not in _hf_cache


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
        f = ModelFamily("Qwen3", "chat", "desc", ())
        with pytest.raises(AttributeError):
            f.name = "nope"  # type: ignore[misc]

    def test_fields(self) -> None:
        v = ModelVariant("repo", "file.gguf", "8B", "Q4_K_M", 5000, True)
        f = ModelFamily("Qwen3", "chat", "Fast", (v,))
        assert f.name == "Qwen3"
        assert f.task == "chat"
        assert len(f.variants) == 1


class TestExtractFamilyName:
    def test_qwen3_8b(self) -> None:
        assert catalog._extract_family_name("Qwen3 8B") == "Qwen3"

    def test_qwen3_06b(self) -> None:
        assert catalog._extract_family_name("Qwen3 0.6B") == "Qwen3"

    def test_qwen3_coder(self) -> None:
        assert catalog._extract_family_name("Qwen3-Coder 30B A3B") == "Qwen3-Coder"

    def test_mistral(self) -> None:
        assert catalog._extract_family_name("Mistral 7B Instruct") == "Mistral"

    def test_no_space_before_version(self) -> None:
        """Names without 'space + digit' pattern return the full name."""
        assert catalog._extract_family_name("Nomic Embed Text v1.5") == "Nomic Embed Text v1.5"

    def test_hyphenated_version(self) -> None:
        """Names with hyphenated versions return the full name."""
        assert catalog._extract_family_name("LightOnOCR-2") == "LightOnOCR-2"


class TestExtractQuant:
    def test_wildcard_pattern(self) -> None:
        assert catalog._extract_quant("*Q4_K_M.gguf") == "Q4_K_M"

    def test_full_filename(self) -> None:
        assert catalog._extract_quant("nomic-embed-text-v1.5.Q4_K_M.gguf") == "Q4_K_M"

    def test_q8_0(self) -> None:
        assert catalog._extract_quant("model-Q8_0.gguf") == "Q8_0"

    def test_no_quant(self) -> None:
        assert catalog._extract_quant("model.gguf") == ""


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
        qwen3 = [f for f in families if f.name == "Qwen3"]
        assert len(qwen3) == 1
        assert len(qwen3[0].variants) == 3  # 0.6B, 4B, 8B

    def test_qwen3_largest_recommended(self) -> None:
        families = get_families()
        qwen3 = next(f for f in families if f.name == "Qwen3")
        assert qwen3.variants[-1].recommended is True
        assert qwen3.variants[0].recommended is False

    def test_single_variant_not_recommended(self) -> None:
        """A family with only one variant should not mark it as recommended."""
        families = get_families()
        singles = [f for f in families if len(f.variants) == 1]
        for fam in singles:
            assert fam.variants[0].recommended is False

    def test_total_variants_matches_featured(self) -> None:
        families = get_families()
        total_variants = sum(len(f.variants) for f in families)
        assert total_variants == len(FEATURED_ALL)

    def test_variant_has_correct_fields(self) -> None:
        families = get_families()
        qwen3 = next(f for f in families if f.name == "Qwen3")
        v = qwen3.variants[0]  # 0.6B
        assert v.param_count == "0.6B"
        assert v.quant == "Q4_K_M"
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
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        entry = FEATURED_VISION[0]
        monkeypatch.setattr(catalog, "_resolve_filename", lambda e: "model-Q4_K_M.gguf")

        download_calls: list[dict] = []

        def fake_download(**kwargs: Any) -> str:
            download_calls.append(kwargs)
            dest = Path(kwargs["local_dir"]) / kwargs["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * 100)
            return str(dest)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        monkeypatch.setattr(
            catalog, "_resolve_mmproj_filename", lambda repo, pat: "model-mmproj-f16.gguf"
        )

        download_model(entry)

        # Should have two downloads: main model + mmproj
        assert len(download_calls) == 2
        filenames = [c["filename"] for c in download_calls]
        assert "model-Q4_K_M.gguf" in filenames
        assert "model-mmproj-f16.gguf" in filenames

    def test_download_model_skips_mmproj_for_chat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """download_model does NOT download mmproj for chat entries."""
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog, "_resolve_filename", lambda e: e.gguf_filename)

        download_calls: list[dict] = []

        def fake_download(**kwargs: Any) -> str:
            download_calls.append(kwargs)
            dest = Path(kwargs["local_dir"]) / kwargs["filename"]
            dest.write_bytes(b"x")
            return str(dest)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        download_model(entry)

        assert len(download_calls) == 1


class TestFindMmprojFile:
    def test_finds_mmproj_in_models_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        mmproj = tmp_path / "model-mmproj-f16.gguf"
        mmproj.write_bytes(b"fake")

        from lilbee.catalog import find_mmproj_file

        result = find_mmproj_file("anything")
        assert result == mmproj

    def test_returns_none_when_no_mmproj(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)

        from lilbee.catalog import find_mmproj_file

        result = find_mmproj_file("anything")
        assert result is None

    def test_returns_none_when_dir_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path / "nonexistent")

        from lilbee.catalog import find_mmproj_file

        result = find_mmproj_file("anything")
        assert result is None


class TestResolveMmprojFilename:
    def test_exact_filename_passthrough(self) -> None:
        result = catalog._resolve_mmproj_filename("repo", "exact-mmproj.gguf")
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

        result = catalog._resolve_mmproj_filename("repo", "*mmproj*.gguf")
        # Prefers f16 over f32
        assert result == "model-mmproj-f16.gguf"

    def test_returns_none_on_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_error(*a, **kw):
            raise RuntimeError("network error")

        monkeypatch.setattr(httpx, "get", raise_error)
        result = catalog._resolve_mmproj_filename("repo", "*mmproj*.gguf")
        assert result is None

    def test_returns_none_when_no_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = {"siblings": [{"rfilename": "model-Q4_K_M.gguf"}]}
        mock_resp = httpx.Response(
            200, json=data, request=httpx.Request("GET", "https://example.com")
        )
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)

        result = catalog._resolve_mmproj_filename("repo", "*mmproj*.gguf")
        assert result is None


class TestHfModelsWithoutGgufExcluded:
    """Models tagged 'gguf' but without actual .gguf files are filtered out."""

    def test_no_gguf_siblings_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [
            {
                "id": "user/gpt2",
                "downloads": 50000,
                "siblings": [
                    {"rfilename": "pytorch_model.bin"},
                    {"rfilename": "config.json"},
                ],
            },
            {
                "id": "user/real-gguf-model",
                "downloads": 1000,
                "siblings": [
                    {"rfilename": "model-Q4_K_M.gguf", "size": 4_000_000_000},
                ],
            },
        ]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = catalog._fetch_hf_models()
        assert len(models) == 1
        assert models[0].hf_repo == "user/real-gguf-model"

    def test_empty_siblings_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [
            {"id": "user/no-files", "downloads": 100, "siblings": []},
        ]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = catalog._fetch_hf_models()
        assert len(models) == 0


class TestGatedRepoShowsLoginMessage:
    def test_permission_error_mentions_login(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        entry = FEATURED_VISION[0]
        monkeypatch.setattr(catalog, "_resolve_filename", lambda e: e.gguf_filename)

        from huggingface_hub.utils import GatedRepoError

        def fake_download(**kwargs: Any) -> str:
            raise GatedRepoError("Gated repo", response=MagicMock())

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        with pytest.raises(PermissionError, match="requires HuggingFace authentication"):
            download_model(entry)

    def test_featured_vision_description_mentions_login(self) -> None:
        for entry in FEATURED_VISION:
            if "LightOnOCR" in entry.name:
                assert "requires login" in entry.description


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


class TestQuantTier:
    def test_all_quant_types_mapped(self) -> None:
        for quant_name, expected_tier in QUANT_TIERS.items():
            assert quant_tier(quant_name) == expected_tier

    def test_unknown_returns_unknown(self) -> None:
        assert quant_tier("") == "unknown"
        assert quant_tier("WEIRD_QUANT") == "unknown"

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
                "Model-7B-GGUF",
                "user/Model-7B-Instruct-GGUF",
                "*Q4_K_M.gguf",
                4.0,
                8.0,
                "A test model",
                False,
                1000,
                "chat",
            ),
            CatalogModel(
                "Qwen3 8B",
                "Qwen/Qwen3-8B-GGUF",
                "*Q4_K_M.gguf",
                5.0,
                8.0,
                "Strong general purpose",
                True,
                0,
                "chat",
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
        assert enriched[0].display_name == "Model 7B"
        assert enriched[1].display_name == "Qwen3 8B"

    def test_quality_tier_populated(self) -> None:
        result = self._make_result()
        enriched = enrich_catalog(result, set())
        assert enriched[0].quality_tier == "balanced"

    def test_installed_status(self) -> None:
        result = self._make_result()
        enriched = enrich_catalog(result, {"Qwen3 8B"})
        assert enriched[0].installed is False
        assert enriched[0].source == "native"
        assert enriched[1].installed is True
        assert enriched[1].source == "litellm"

    def test_preserves_original_fields(self) -> None:
        result = self._make_result()
        enriched = enrich_catalog(result, set())
        original = result.models[0]
        e = enriched[0]
        assert e.name == original.name
        assert e.hf_repo == original.hf_repo
        assert e.size_gb == original.size_gb
        assert e.description == original.description
        assert e.featured == original.featured
        assert e.downloads == original.downloads
        assert e.task == original.task
