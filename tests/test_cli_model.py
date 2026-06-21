"""Tests for the `lilbee model` CLI sub-app and its typed data helpers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.app import models as model_mod
from lilbee.app.models import (
    AdoptStatus,
    CatalogEntryData,
    ListModelsResult,
    ManifestData,
    ModelEntry,
    PullEvent,
    PullResult,
    PullStatus,
    RemoveResult,
    ShowModelResult,
)
from lilbee.app.settings import SettingsUpdateResult
from lilbee.catalog.types import ModelSource
from lilbee.cli import app
from lilbee.core.config import cfg
from lilbee.modelhub.model_manager import ModelNotFoundError
from lilbee.modelhub.registry import ModelManifest

runner = CliRunner()


def _manifest(hf_repo: str, gguf_filename: str, *, size: int, task: str) -> ModelManifest:
    return ModelManifest(
        hf_repo=hf_repo,
        gguf_filename=gguf_filename,
        size_bytes=size,
        task=task,
        downloaded_at="2026-04-11T00:00:00+00:00",
    )


def _remote(
    name: str, task: str, parameter_size: str = "8B", provider: str = "Ollama"
) -> MagicMock:
    rm = MagicMock()
    rm.name = name
    rm.task = task
    rm.parameter_size = parameter_size
    # Source is derived from the provider label, so it must be a real string.
    rm.provider = provider
    return rm


def _catalog_model(*, hf_repo: str = "Qwen/Qwen3-0.6B-GGUF", task: str = "chat") -> MagicMock:
    from lilbee.catalog.types import ModelCompat

    entry = MagicMock()
    entry.ref = hf_repo
    entry.display_name = "Qwen3 0.6B"
    entry.hf_repo = hf_repo
    entry.gguf_filename = "*Q4_K_M.gguf"
    entry.size_gb = 0.5
    entry.min_ram_gb = 2.0
    entry.description = "Tiny chat model"
    entry.task = task
    entry.featured = True
    entry.recommended = True
    entry.architecture = ""
    entry.compat = ModelCompat.UNKNOWN
    return entry


# Canonical refs reused across the suite.
_CHAT_REPO = "Qwen/Qwen3-0.6B-GGUF"
_CHAT_FILE = "Qwen3-0.6B-Q4_K_M.gguf"
_CHAT_REF = f"{_CHAT_REPO}/{_CHAT_FILE}"
_OLLAMA_REF = "ollama/llama3:latest"


class _FakeManager:
    """Minimal ModelManager test double with recorded call sites."""

    def __init__(
        self,
        *,
        native: list[str] | None = None,
        litellm: list[str] | None = None,
    ) -> None:
        self._native = list(native or [])
        self._litellm = list(litellm or [])
        self.pull_calls: list[tuple[str, ModelSource]] = []
        self.remove_calls: list[tuple[str, ModelSource | None]] = []

    def list_installed(self, source: ModelSource | None = None) -> list[str]:
        if source is None:
            return sorted({*self._native, *self._litellm})
        if source is ModelSource.NATIVE:
            return list(self._native)
        return list(self._litellm)

    def is_installed(self, model: str, source: ModelSource | None = None) -> bool:
        if source is None:
            return model in self._native or model in self._litellm
        if source is ModelSource.NATIVE:
            return model in self._native
        return model in self._litellm

    def get_source(self, model: str) -> ModelSource | None:
        if model in self._native:
            return ModelSource.NATIVE
        if model in self._litellm:
            return ModelSource.REMOTE
        return None

    def pull(self, model, source, *, on_bytes=None, allow_unsupported=False):
        self.pull_calls.append((model, source))
        if on_bytes is not None:
            on_bytes(50, 100)
        return f"/fake/{model}.gguf"

    def remove(self, model, source=None) -> bool:
        # Mirror ModelManager: only native models are removable; local servers
        # are read-only, so a non-native source is refused.
        self.remove_calls.append((model, source))
        if source is not None and source is not ModelSource.NATIVE:
            raise ValueError("lilbee runs Ollama models but doesn't remove them.")
        if model in self._native:
            self._native.remove(model)
            return True
        return False


@pytest.fixture
def fake_manager():
    manager = _FakeManager(native=[_CHAT_REF], litellm=[_OLLAMA_REF])
    with patch("lilbee.app.models.get_services", return_value=MagicMock(model_manager=manager)):
        yield manager


@pytest.fixture
def empty_manager():
    manager = _FakeManager()
    with patch("lilbee.app.models.get_services", return_value=MagicMock(model_manager=manager)):
        yield manager


@pytest.fixture
def native_manifests():
    manifests = {
        _CHAT_REF: _manifest(_CHAT_REPO, _CHAT_FILE, size=5 * 1024**3, task="chat"),
    }
    with patch("lilbee.app.models._native_manifest_index", return_value=manifests):
        yield manifests


@pytest.fixture
def no_remote_classify():
    with patch("lilbee.modelhub.model_manager.classify_all_remote_models", return_value=[]):
        yield


@pytest.fixture
def with_remote_classify():
    remote = [_remote(_OLLAMA_REF, task="chat", parameter_size="8B")]
    with patch("lilbee.modelhub.model_manager.classify_all_remote_models", return_value=remote):
        yield remote


class TestModelEntryFactories:
    def test_from_native_populates_size_and_task(self):
        manifest = _manifest(_CHAT_REPO, _CHAT_FILE, size=2 * 1024**3, task="chat")
        entry = ModelEntry.from_native(_CHAT_REF, manifest)
        assert entry.source == "native"
        assert entry.size_gb == 2.0
        assert entry.task == "chat"
        # Display derives from clean_display_name(hf_repo).
        assert entry.display_name == "Qwen3 0.6B"

    def test_from_native_missing_manifest(self):
        entry = ModelEntry.from_native("Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", None)
        assert entry.source == "native"
        assert entry.task is None
        assert entry.size_gb is None
        assert entry.display_name == ""

    def test_from_native_reports_full_split_total(self):
        # A split GGUF lists its total on-disk size, not the
        # first-shard size, so list/show agree with the freed-on-remove total.
        manifest = _manifest(_CHAT_REPO, _CHAT_FILE, size=1 * 1024**3, task="chat")
        manifest.total_size_bytes = 6 * 1024**3  # six-shard total
        entry = ModelEntry.from_native(_CHAT_REF, manifest)
        assert entry.size_gb == 6.0

    def test_manifest_data_reports_full_split_total(self):
        manifest = _manifest(_CHAT_REPO, _CHAT_FILE, size=1 * 1024**3, task="chat")
        manifest.total_size_bytes = 6 * 1024**3
        data = ManifestData.from_manifest(manifest)
        assert data.size_gb == 6.0
        assert data.size_bytes == 6 * 1024**3

    def test_from_backend_with_remote(self):
        remote = _remote(_OLLAMA_REF, task="chat", parameter_size="8B")
        entry = ModelEntry.from_backend(_OLLAMA_REF, remote, ModelSource.OLLAMA)
        assert entry.source == "ollama"
        assert entry.task == "chat"
        assert entry.display_name == "8B"

    def test_from_backend_prefixes_bare_name(self):
        remote = _remote("qwen2.5-coder", task="chat", parameter_size="7B")
        entry = ModelEntry.from_backend("qwen2.5-coder", remote, ModelSource.LM_STUDIO)
        assert entry.source == "lm_studio"
        assert entry.name == "lm_studio/qwen2.5-coder"

    def test_from_backend_missing_remote(self):
        entry = ModelEntry.from_backend(_OLLAMA_REF, None, ModelSource.OLLAMA)
        assert entry.source == "ollama"
        assert entry.task is None
        assert entry.display_name == ""


class TestRemoveModelDataFreedSize:
    def test_legacy_split_manifest_reports_full_shard_total(self, tmp_path, monkeypatch):
        # A pre-accounting split manifest
        # (total_size_bytes None) still frees every shard, so the reported freed
        # size must reflect the on-disk total, not just the first shard.
        from unittest.mock import MagicMock

        from lilbee.app.models import _bytes_to_gb, remove_model_data

        manifest = _manifest(_CHAT_REPO, _CHAT_FILE, size=10, task="chat")  # first shard only
        shards = []
        for n in (1, 2, 3):
            path = tmp_path / f"shard{n}.gguf"
            path.write_bytes(b"x" * 100)  # 100 bytes each -> 300 total
            shards.append(path)
        registry = MagicMock()
        registry.shard_paths.return_value = shards
        manager = MagicMock()
        manager.remove.return_value = True
        services = MagicMock(model_manager=manager, registry=registry)
        monkeypatch.setattr(model_mod, "get_services", lambda: services)
        monkeypatch.setattr(model_mod, "_native_manifest_index", lambda: {_CHAT_REF: manifest})

        result = remove_model_data(_CHAT_REF)
        assert result.freed_gb == _bytes_to_gb(300)  # all 3 shards, not the 10-byte first

    def test_modern_manifest_uses_recorded_total(self, monkeypatch):
        from unittest.mock import MagicMock

        from lilbee.app.models import _bytes_to_gb, remove_model_data

        manifest = _manifest(_CHAT_REPO, _CHAT_FILE, size=10, task="chat")
        manifest.total_size_bytes = 600  # accounted at install; no shard re-stat needed
        registry = MagicMock()
        manager = MagicMock()
        manager.remove.return_value = True
        services = MagicMock(model_manager=manager, registry=registry)
        monkeypatch.setattr(model_mod, "get_services", lambda: services)
        monkeypatch.setattr(model_mod, "_native_manifest_index", lambda: {_CHAT_REF: manifest})

        result = remove_model_data(_CHAT_REF)
        assert result.freed_gb == _bytes_to_gb(600)
        registry.shard_paths.assert_not_called()  # recorded total used directly


class TestListModelsData:
    def test_default_lists_both_sources(self, fake_manager, native_manifests, with_remote_classify):
        data = model_mod.list_models_data()
        assert isinstance(data, ListModelsResult)
        assert data.total == 2
        sources = {e.source for e in data.models}
        assert sources == {"native", "ollama"}

    def test_filter_source_native_skips_litellm_http(self, fake_manager, native_manifests):
        with patch("lilbee.modelhub.model_manager.classify_all_remote_models") as classify:
            data = model_mod.list_models_data(source=ModelSource.NATIVE)
        classify.assert_not_called()
        assert data.total == 1
        assert data.models[0].name == _CHAT_REF

    def test_filter_source_remote_keeps_all_backend(
        self, fake_manager, native_manifests, with_remote_classify
    ):
        # REMOTE is the generic backend bucket: every backend entry, granular source.
        data = model_mod.list_models_data(source=ModelSource.REMOTE)
        assert data.total == 1
        assert data.models[0].source == "ollama"

    def test_filter_source_ollama_matches_backend(
        self, fake_manager, native_manifests, with_remote_classify
    ):
        data = model_mod.list_models_data(source=ModelSource.OLLAMA)
        assert {e.source for e in data.models} == {"ollama"}

    def test_filter_source_lm_studio_empty_against_ollama_backend(
        self, fake_manager, native_manifests, with_remote_classify
    ):
        # Backend is Ollama (default base url), so an LM Studio filter yields nothing.
        data = model_mod.list_models_data(source=ModelSource.LM_STUDIO)
        assert data.models == []

    def test_task_filter_drops_entries_without_matching_task(
        self, fake_manager, native_manifests, with_remote_classify
    ):
        data = model_mod.list_models_data(task="chat")
        assert {e.name for e in data.models} == {_CHAT_REF, _OLLAMA_REF}
        empty = model_mod.list_models_data(task="embedding")
        assert empty.total == 0

    def test_empty_when_no_models_installed(self, empty_manager, no_remote_classify):
        with patch("lilbee.app.models._native_manifest_index", return_value={}):
            data = model_mod.list_models_data()
        assert data.total == 0


class TestListCmd:
    def test_human_output(self, fake_manager, native_manifests, with_remote_classify):
        result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0, result.output
        assert _CHAT_REF in result.output
        assert _OLLAMA_REF in result.output

    def test_json_output_roundtrips(self, fake_manager, native_manifests, with_remote_classify):
        result = runner.invoke(app, ["--json", "model", "list"])
        assert result.exit_code == 0, result.output
        parsed = ListModelsResult.model_validate_json(result.output)
        assert parsed.total == 2
        assert {e.name for e in parsed.models} == {_CHAT_REF, _OLLAMA_REF}

    def test_empty_human_message(self, empty_manager, no_remote_classify):
        with patch("lilbee.app.models._native_manifest_index", return_value={}):
            result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0
        assert "No models installed" in result.output

    def test_invalid_source_raises_bad_param(self, fake_manager):
        result = runner.invoke(app, ["model", "list", "--source", "bogus"])
        assert result.exit_code != 0
        assert "bogus" in result.output

    def test_invalid_source_json_returns_error(self, fake_manager):
        result = runner.invoke(app, ["--json", "model", "list", "--source", "bogus"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "error" in data
        assert "bogus" in data["error"]

    def test_invalid_task_raises_bad_param(self, fake_manager):
        result = runner.invoke(app, ["model", "list", "--task", "bogus"])
        assert result.exit_code != 0
        # Friendly message listing valid tasks; never leaks the internal enum name.
        assert "ModelTask" not in result.output
        assert "chat" in result.output

    def test_invalid_task_json_mode_emits_error_envelope(self, fake_manager):
        result = runner.invoke(app, ["--json", "model", "list", "--task", "bogus"])
        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "error" in data
        assert "bogus" in data["error"]
        assert "ModelTask" not in data["error"]


class TestShowModelData:
    def test_catalog_and_installed_merged(self, fake_manager, native_manifests):
        entry = _catalog_model()
        with (
            patch("lilbee.catalog.find_catalog_entry", return_value=entry),
            patch(
                "lilbee.app.models._resolve_native_path",
                return_value="/fake/path.gguf",
            ),
        ):
            data = model_mod.show_model_data(_CHAT_REF)
        assert isinstance(data, ShowModelResult)
        assert data.installed is True
        assert data.source == "native"
        assert data.path == "/fake/path.gguf"
        assert data.catalog is not None
        assert data.catalog.display_name == "Qwen3 0.6B"
        assert data.manifest is not None
        assert data.manifest.task == "chat"

    def test_catalog_only_not_installed(self, empty_manager):
        entry = _catalog_model(hf_repo="Qwen/Qwen3-8B-GGUF")
        with (
            patch("lilbee.app.models._native_manifest_index", return_value={}),
            patch("lilbee.catalog.find_catalog_entry", return_value=entry),
        ):
            data = model_mod.show_model_data("Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
        assert data.installed is False
        assert data.catalog is not None
        assert data.manifest is None

    def test_unknown_ref_raises_not_found(self, empty_manager):
        with (
            patch("lilbee.app.models._native_manifest_index", return_value={}),
            patch("lilbee.catalog.find_catalog_entry", return_value=None),
            pytest.raises(ModelNotFoundError, match="model not found: ghost"),
        ):
            model_mod.show_model_data("ghost:latest")


class TestResolveNativePath:
    def test_returns_path_when_registry_resolves(self, tmp_path):
        fake_registry = MagicMock()
        fake_registry.resolve.return_value = tmp_path / "blob.gguf"
        with patch("lilbee.app.models.ModelRegistry", return_value=fake_registry):
            path = model_mod._resolve_native_path(_CHAT_REF)
        assert path == str(tmp_path / "blob.gguf")

    def test_suppresses_key_error_from_missing_blob(self):
        fake_registry = MagicMock()
        fake_registry.resolve.side_effect = KeyError("no blob")
        with patch("lilbee.app.models.ModelRegistry", return_value=fake_registry):
            path = model_mod._resolve_native_path(_CHAT_REF)
        assert path is None

    def test_suppresses_value_error_from_invalid_ref(self):
        fake_registry = MagicMock()
        fake_registry.resolve.side_effect = ValueError("bad ref")
        with patch("lilbee.app.models.ModelRegistry", return_value=fake_registry):
            path = model_mod._resolve_native_path(_CHAT_REF)
        assert path is None


class TestShowCmd:
    def test_human_output_installed(self, fake_manager, native_manifests):
        entry = _catalog_model()
        with (
            patch("lilbee.catalog.find_catalog_entry", return_value=entry),
            patch(
                "lilbee.app.models._resolve_native_path",
                return_value="/fake/path.gguf",
            ),
        ):
            result = runner.invoke(app, ["model", "show", _CHAT_REF])
        assert result.exit_code == 0, result.output
        assert "source:" in result.output
        assert "/fake/path.gguf" in result.output
        assert "downloaded:" in result.output

    def test_json_output_roundtrips(self, fake_manager, native_manifests):
        entry = _catalog_model()
        with (
            patch("lilbee.catalog.find_catalog_entry", return_value=entry),
            patch("lilbee.app.models._resolve_native_path", return_value="/p.gguf"),
        ):
            result = runner.invoke(app, ["--json", "model", "show", _CHAT_REF])
        assert result.exit_code == 0, result.output
        parsed = ShowModelResult.model_validate_json(result.output)
        assert parsed.installed is True
        assert parsed.catalog is not None
        assert parsed.catalog.display_name == "Qwen3 0.6B"
        assert parsed.path == "/p.gguf"

    def test_json_not_found_exits_1(self, empty_manager):
        with (
            patch("lilbee.app.models._native_manifest_index", return_value={}),
            patch("lilbee.catalog.find_catalog_entry", return_value=None),
        ):
            result = runner.invoke(app, ["--json", "model", "show", "ghost:1"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert "model not found" in payload["error"]

    def test_human_not_found_exits_1(self, empty_manager):
        with (
            patch("lilbee.app.models._native_manifest_index", return_value={}),
            patch("lilbee.catalog.find_catalog_entry", return_value=None),
        ):
            result = runner.invoke(app, ["model", "show", "ghost:1"])
        assert result.exit_code == 1
        assert "model not found" in result.output


class TestPullModelData:
    def test_already_installed_short_circuits(self, fake_manager, native_manifests):
        result = model_mod.pull_model_data(_CHAT_REF, ModelSource.NATIVE)
        assert isinstance(result, PullResult)
        assert result.status == PullStatus.ALREADY_INSTALLED
        assert fake_manager.pull_calls == []

    def test_pull_native_invokes_manager_and_callbacks(self, fake_manager, native_manifests):
        events = []
        target = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
        result = model_mod.pull_model_data(target, ModelSource.NATIVE, on_update=events.append)
        assert result.status == PullStatus.OK
        assert result.path == f"/fake/{target}.gguf"
        assert events
        assert events[0].percent == 50
        assert fake_manager.pull_calls == [(target, ModelSource.NATIVE)]


class TestAdoptEmbedder:
    """Adopting a downloaded index's embedder pulls it when missing and routes
    the switch through the settings boundary without a rebuild."""

    _NO_REINDEX = SettingsUpdateResult(updated=["embedding_model"], reindex_required=False)

    def test_adopts_and_pulls_when_missing(self, fake_manager):
        target = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic.Q4_K_M.gguf"
        with patch(
            "lilbee.app.settings.apply_settings_update", return_value=self._NO_REINDEX
        ) as apply:
            result = model_mod.adopt_embedder(target)
        assert result.status == AdoptStatus.ADOPTED
        assert result.reindex_required is False
        assert (target, ModelSource.NATIVE) in fake_manager.pull_calls
        apply.assert_called_once_with({"embedding_model": target})

    def test_already_active_skips_pull(self, fake_manager):
        original = cfg.embedding_model
        cfg.embedding_model = _CHAT_REF  # _CHAT_REF is installed in fake_manager
        try:
            with patch("lilbee.app.settings.apply_settings_update", return_value=self._NO_REINDEX):
                result = model_mod.adopt_embedder(_CHAT_REF)
        finally:
            cfg.embedding_model = original
        assert result.status == AdoptStatus.ALREADY_ACTIVE
        assert fake_manager.pull_calls == []


class TestPullCmd:
    def test_json_stream_emits_done_event(self, fake_manager, native_manifests):
        result = runner.invoke(
            app, ["--json", "model", "pull", "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"]
        )
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.splitlines() if line.strip()]
        parsed = [json.loads(line) for line in lines]
        assert parsed[-1]["event"] == PullEvent.DONE.value
        assert parsed[-1]["status"] == PullStatus.OK.value
        assert parsed[-1]["model"] == "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"

    def test_human_mode_prints_pulled(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "pull", "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"])
        assert result.exit_code == 0, result.output
        assert "Pulled" in result.output

    def test_human_already_installed_message(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "pull", _CHAT_REF])
        assert result.exit_code == 0
        assert "already installed" in result.output

    def test_runtime_error_json(self, native_manifests):
        manager = _FakeManager()
        manager.pull = MagicMock(side_effect=RuntimeError("no network"))
        with patch("lilbee.app.models.get_services", return_value=MagicMock(model_manager=manager)):
            result = runner.invoke(
                app, ["--json", "model", "pull", "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"]
            )
        assert result.exit_code == 1
        payload = json.loads(result.output.strip().splitlines()[-1])
        assert payload == {"error": "no network"}

    def test_runtime_error_human(self, native_manifests):
        manager = _FakeManager()
        manager.pull = MagicMock(side_effect=RuntimeError("boom"))
        with patch("lilbee.app.models.get_services", return_value=MagicMock(model_manager=manager)):
            result = runner.invoke(
                app, ["model", "pull", "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"]
            )
        assert result.exit_code == 1
        assert "boom" in result.output

    def test_invalid_source(self, fake_manager):
        result = runner.invoke(
            app, ["model", "pull", "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf", "--source", "bad"]
        )
        assert result.exit_code != 0


class TestRemoveModelData:
    def test_removes_and_reports_freed(self, fake_manager, native_manifests):
        result = model_mod.remove_model_data(_CHAT_REF)
        assert isinstance(result, RemoveResult)
        assert result.deleted is True
        assert result.freed_gb == 5.0
        assert fake_manager.remove_calls == [(_CHAT_REF, None)]

    def test_missing_manifest_returns_zero_freed(self, fake_manager):
        with patch("lilbee.app.models._native_manifest_index", return_value={}):
            result = model_mod.remove_model_data(_CHAT_REF)
        assert result.deleted is True
        assert result.freed_gb == 0.0


class TestRmCmd:
    def test_confirm_declined(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "rm", _CHAT_REF], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert fake_manager.remove_calls == []

    def test_confirm_accepted(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "rm", _CHAT_REF], input="y\n")
        assert result.exit_code == 0
        assert "5.00 GB freed" in result.output
        assert fake_manager.remove_calls == [(_CHAT_REF, None)]

    def test_yes_flag_skips_prompt(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "rm", "--yes", _CHAT_REF])
        assert result.exit_code == 0
        assert "Removed" in result.output

    def test_not_found_exits_1(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "rm", "--yes", "ghost:1.0"])
        assert result.exit_code == 1
        assert "Not found" in result.output

    def test_json_output_serializes_remove_result(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["--json", "model", "rm", _CHAT_REF])
        assert result.exit_code == 0
        parsed = RemoveResult.model_validate_json(result.output)
        assert parsed.deleted is True
        assert parsed.freed_gb == 5.0

    def test_json_not_found_exits_1(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["--json", "model", "rm", "ghost:1.0"])
        assert result.exit_code == 1
        parsed = RemoveResult.model_validate_json(result.output)
        assert parsed.deleted is False

    def test_read_only_refusal_exits_1(self, fake_manager, native_manifests):
        msg = "lilbee runs Ollama models but doesn't remove them. Manage them in Ollama instead."
        with patch("lilbee.cli.model.remove_model_data", side_effect=ValueError(msg)):
            result = runner.invoke(app, ["model", "rm", "--yes", "ollama/llama3:latest"])
        assert result.exit_code == 1
        assert "doesn't remove them" in result.output

    def test_json_read_only_refusal_exits_1(self, fake_manager, native_manifests):
        msg = "lilbee runs Ollama models but doesn't remove them. Manage them in Ollama instead."
        with patch("lilbee.cli.model.remove_model_data", side_effect=ValueError(msg)):
            result = runner.invoke(app, ["--json", "model", "rm", "--yes", "ollama/llama3:latest"])
        assert result.exit_code == 1
        assert "doesn't remove them" in result.output

    def test_invalid_source(self, fake_manager):
        result = runner.invoke(app, ["model", "rm", "--yes", _CHAT_REF, "--source", "bad"])
        assert result.exit_code != 0


class TestBrowseCmd:
    def test_json_mode_rejected_exit_2(self, fake_manager):
        result = runner.invoke(app, ["--json", "model", "browse"])
        assert result.exit_code == 2
        payload = json.loads(result.output)
        assert "interactive" in payload["error"]

    def test_non_tty_rejected_exit_1(self, fake_manager):
        result = runner.invoke(app, ["model", "browse"])
        assert result.exit_code == 1
        assert "terminal" in result.output

    def test_tty_launches_tui_with_catalog(self, fake_manager):
        with (
            patch("lilbee.cli.model._is_interactive_terminal", return_value=True),
            patch("lilbee.cli.tui.run_tui") as run_tui,
        ):
            result = runner.invoke(app, ["model", "browse"])
        assert result.exit_code == 0, result.output
        run_tui.assert_called_once_with(initial_view="Catalog")


class TestCatalogEntryDataFactory:
    def test_from_catalog_model_maps_fields(self):
        entry = _catalog_model()
        data = CatalogEntryData.from_catalog_model(entry)
        # CatalogModel.ref is now the bare hf_repo (catalog browse identity).
        assert data.ref == "Qwen/Qwen3-0.6B-GGUF"
        assert data.hf_repo == "Qwen/Qwen3-0.6B-GGUF"
        assert data.featured is True
        assert data.recommended is True


class TestManifestDataFactory:
    def test_from_manifest_computes_size_gb(self):
        manifest = _manifest(_CHAT_REPO, _CHAT_FILE, size=3 * 1024**3, task="chat")
        data = ManifestData.from_manifest(manifest)
        assert data.size_gb == 3.0
        assert data.ref == _CHAT_REF
        assert data.hf_repo == _CHAT_REPO
        assert data.gguf_filename == _CHAT_FILE


class TestNativeManifestIndex:
    def test_indexes_by_ref(self, tmp_path):
        fake_registry = MagicMock()
        fake_registry.list_installed.return_value = [
            _manifest(_CHAT_REPO, _CHAT_FILE, size=1024, task="chat"),
        ]
        with patch("lilbee.app.models.ModelRegistry", return_value=fake_registry):
            index = model_mod._native_manifest_index()
        assert _CHAT_REF in index
        assert index[_CHAT_REF].task == "chat"
