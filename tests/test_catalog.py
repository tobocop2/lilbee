"""Tests for catalog.py — model catalog, HF API fetching, filtering, downloading."""

from pathlib import Path
from typing import ClassVar

import httpx
import pytest

from lilbee import catalog
from lilbee.catalog import (
    FEATURED_ALL,
    FEATURED_CHAT,
    FEATURED_EMBEDDING,
    FEATURED_VISION,
    CatalogModel,
    CatalogResult,
    download_model,
    find_catalog_entry,
    get_catalog,
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
                "siblings": [],
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

    def test_no_gguf_files_fallback_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_resp = httpx.Response(200, json=self._mock_hf_response())
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = catalog._fetch_hf_models()
        # Second model has no siblings -> fallback 0.0 (unknown)
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
                "siblings": [],
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
                "siblings": [],
            },
            {
                "id": "user/vision-model",
                "downloads": 50,
                "pipeline_tag": "image-text-to-text",
                "siblings": [],
            },
        ]
        mock_resp = httpx.Response(200, json=data)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock_resp)
        models = catalog._fetch_hf_models()
        assert models[0].task == "embedding"
        assert models[1].task == "vision"

    def test_missing_pipeline_tag_defaults_to_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [{"id": "user/model", "downloads": 100, "siblings": []}]
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
        from lilbee.registry import ModelManifest, ModelRef, ModelRegistry

        registry = ModelRegistry(tmp_path)
        source = tmp_path / entry.gguf_filename
        source.write_bytes(b"fake model")
        ref = ModelRef.parse(entry.name)
        manifest = ModelManifest(
            name=ref.name,
            tag=ref.tag,
            blob="",
            size_bytes=10,
            task=entry.task,
            source_repo=entry.hf_repo,
            source_filename=entry.gguf_filename,
            downloaded_at="2026-01-01T00:00:00+00:00",
        )
        blob_path = registry.install(ref, source, manifest)
        result = download_model(entry)
        assert result == blob_path

    def test_creates_models_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        models_dir = tmp_path / "models"
        monkeypatch.setattr(catalog.cfg, "models_dir", models_dir)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog, "_resolve_filename", lambda e: e.gguf_filename)

        class FakeStream:
            headers: ClassVar[dict[str, str]] = {"content-length": "100"}

            def __init__(self, *a: object, **kw: object) -> None:
                pass

            def __enter__(self) -> "FakeStream":
                return self

            def __exit__(self, *a: object) -> None:
                pass

            def raise_for_status(self) -> None:
                pass

            def iter_bytes(self, chunk_size: int = 8192) -> list[bytes]:
                return [b"x" * 100]

        monkeypatch.setattr(httpx, "stream", lambda *a, **kw: FakeStream())
        result = download_model(entry)
        assert result.exists()
        assert result.parent == models_dir / "blobs"

    def test_calls_progress_callback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog, "_resolve_filename", lambda e: e.gguf_filename)

        progress_calls: list[tuple[int, int]] = []

        class FakeStream:
            headers: ClassVar[dict[str, str]] = {"content-length": "100"}

            def __init__(self, *a: object, **kw: object) -> None:
                pass

            def __enter__(self) -> "FakeStream":
                return self

            def __exit__(self, *a: object) -> None:
                pass

            def raise_for_status(self) -> None:
                pass

            def iter_bytes(self, chunk_size: int = 8192) -> list[bytes]:
                return [b"x" * 50, b"x" * 50]

        monkeypatch.setattr(httpx, "stream", lambda *a, **kw: FakeStream())

        def on_progress(downloaded: int, total: int) -> None:
            progress_calls.append((downloaded, total))

        download_model(entry, on_progress=on_progress)
        assert len(progress_calls) == 2
        assert progress_calls[-1] == (100, 100)

    def test_http_error_cleans_up_partial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(catalog.cfg, "models_dir", tmp_path)
        entry = FEATURED_EMBEDDING[0]
        monkeypatch.setattr(catalog, "_resolve_filename", lambda e: e.gguf_filename)

        class FakeStream:
            headers: ClassVar[dict[str, str]] = {"content-length": "100"}

            def __init__(self, *a: object, **kw: object) -> None:
                pass

            def __enter__(self) -> "FakeStream":
                return self

            def __exit__(self, *a: object) -> None:
                pass

            def raise_for_status(self) -> None:
                pass

            def iter_bytes(self, chunk_size: int = 8192) -> list[bytes]:
                raise httpx.HTTPStatusError(
                    "500",
                    request=httpx.Request("GET", "http://x"),
                    response=httpx.Response(500),
                )

        monkeypatch.setattr(httpx, "stream", lambda *a, **kw: FakeStream())
        with pytest.raises(httpx.HTTPStatusError):
            download_model(entry)
        assert not (tmp_path / entry.gguf_filename).exists()


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
