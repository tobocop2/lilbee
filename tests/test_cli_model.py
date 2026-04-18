"""Tests for the `lilbee model` CLI sub-app and its typed data helpers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lilbee.cli import app
from lilbee.cli import model as model_mod
from lilbee.cli.model import (
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
from lilbee.model_manager import ModelNotFoundError, ModelSource
from lilbee.registry import ModelManifest

runner = CliRunner()


def _manifest(name: str, tag: str, *, size: int, task: str) -> ModelManifest:
    return ModelManifest(
        name=name,
        tag=tag,
        size_bytes=size,
        task=task,
        source_repo=f"org/{name}-GGUF",
        source_filename=f"{name}.gguf",
        downloaded_at="2026-04-11T00:00:00+00:00",
        display_name=f"{name} {tag}",
    )


def _remote(name: str, task: str, parameter_size: str = "8B") -> MagicMock:
    rm = MagicMock()
    rm.name = name
    rm.task = task
    rm.parameter_size = parameter_size
    return rm


def _catalog_model(*, name: str = "qwen3", tag: str = "0.6b", task: str = "chat") -> MagicMock:
    entry = MagicMock()
    entry.name = name
    entry.tag = tag
    entry.ref = f"{name}:{tag}"
    entry.display_name = f"Qwen3 {tag.upper()}"
    entry.hf_repo = f"Qwen/Qwen3-{tag.upper()}-GGUF"
    entry.size_gb = 0.5
    entry.min_ram_gb = 2.0
    entry.description = "Tiny chat model"
    entry.task = task
    entry.featured = True
    entry.recommended = True
    return entry


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
            return ModelSource.LITELLM
        return None

    def pull(self, model, source, *, on_progress=None, on_bytes=None):
        self.pull_calls.append((model, source))
        if on_bytes is not None:
            on_bytes(50, 100)
        return f"/fake/{model}.gguf"

    def remove(self, model, source=None) -> bool:
        self.remove_calls.append((model, source))
        if (source is ModelSource.NATIVE or source is None) and model in self._native:
            self._native.remove(model)
            return True
        if (source is ModelSource.LITELLM or source is None) and model in self._litellm:
            self._litellm.remove(model)
            return True
        return False


@pytest.fixture
def fake_manager():
    manager = _FakeManager(native=["qwen3:0.6b"], litellm=["llama3:latest"])
    with patch("lilbee.model_manager.get_model_manager", return_value=manager):
        yield manager


@pytest.fixture
def empty_manager():
    manager = _FakeManager()
    with patch("lilbee.model_manager.get_model_manager", return_value=manager):
        yield manager


@pytest.fixture
def native_manifests():
    manifests = {
        "qwen3:0.6b": _manifest("qwen3", "0.6b", size=5 * 1024**3, task="chat"),
    }
    with patch("lilbee.cli.model._native_manifest_index", return_value=manifests):
        yield manifests


@pytest.fixture
def no_remote_classify():
    with patch("lilbee.model_manager.classify_remote_models", return_value=[]):
        yield


@pytest.fixture
def with_remote_classify():
    remote = [_remote("llama3:latest", task="chat", parameter_size="8B")]
    with patch("lilbee.model_manager.classify_remote_models", return_value=remote):
        yield remote


class TestModelEntryFactories:
    def test_from_native_populates_size_and_task(self):
        manifest = _manifest("qwen3", "0.6b", size=2 * 1024**3, task="chat")
        entry = ModelEntry.from_native("qwen3:0.6b", manifest)
        assert entry.source == "native"
        assert entry.size_gb == 2.0
        assert entry.task == "chat"
        assert entry.display_name == "qwen3 0.6b"

    def test_from_native_missing_manifest(self):
        entry = ModelEntry.from_native("qwen3:8b", None)
        assert entry.source == "native"
        assert entry.task is None
        assert entry.size_gb is None
        assert entry.display_name == ""

    def test_from_litellm_with_remote(self):
        remote = _remote("llama3:latest", task="chat", parameter_size="8B")
        entry = ModelEntry.from_litellm("llama3:latest", remote)
        assert entry.source == "litellm"
        assert entry.task == "chat"
        assert entry.display_name == "8B"

    def test_from_litellm_missing_remote(self):
        entry = ModelEntry.from_litellm("llama3:latest", None)
        assert entry.task is None
        assert entry.display_name == ""


class TestListModelsData:
    def test_default_lists_both_sources(self, fake_manager, native_manifests, with_remote_classify):
        data = model_mod.list_models_data()
        assert isinstance(data, ListModelsResult)
        assert data.total == 2
        sources = {e.source for e in data.models}
        assert sources == {"native", "litellm"}

    def test_filter_source_native_skips_litellm_http(self, fake_manager, native_manifests):
        with patch("lilbee.model_manager.classify_remote_models") as classify:
            data = model_mod.list_models_data(source=ModelSource.NATIVE)
        classify.assert_not_called()
        assert data.total == 1
        assert data.models[0].name == "qwen3:0.6b"

    def test_filter_source_litellm(self, fake_manager, native_manifests, with_remote_classify):
        data = model_mod.list_models_data(source=ModelSource.LITELLM)
        assert data.total == 1
        assert data.models[0].source == "litellm"

    def test_task_filter_drops_entries_without_matching_task(
        self, fake_manager, native_manifests, with_remote_classify
    ):
        data = model_mod.list_models_data(task="chat")
        assert {e.name for e in data.models} == {"qwen3:0.6b", "llama3:latest"}
        empty = model_mod.list_models_data(task="embedding")
        assert empty.total == 0

    def test_empty_when_no_models_installed(self, empty_manager, no_remote_classify):
        with patch("lilbee.cli.model._native_manifest_index", return_value={}):
            data = model_mod.list_models_data()
        assert data.total == 0


class TestListCmd:
    def test_human_output(self, fake_manager, native_manifests, with_remote_classify):
        result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0, result.output
        assert "qwen3:0.6b" in result.output
        assert "llama3:latest" in result.output

    def test_json_output_roundtrips(self, fake_manager, native_manifests, with_remote_classify):
        result = runner.invoke(app, ["--json", "model", "list"])
        assert result.exit_code == 0, result.output
        parsed = ListModelsResult.model_validate_json(result.output)
        assert parsed.total == 2
        assert {e.name for e in parsed.models} == {"qwen3:0.6b", "llama3:latest"}

    def test_empty_human_message(self, empty_manager, no_remote_classify):
        with patch("lilbee.cli.model._native_manifest_index", return_value={}):
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


class TestShowModelData:
    def test_catalog_and_installed_merged(self, fake_manager, native_manifests):
        entry = _catalog_model()
        with (
            patch("lilbee.catalog.find_catalog_entry", return_value=entry),
            patch(
                "lilbee.cli.model._resolve_native_path",
                return_value="/fake/path.gguf",
            ),
        ):
            data = model_mod.show_model_data("qwen3:0.6b")
        assert isinstance(data, ShowModelResult)
        assert data.installed is True
        assert data.source == "native"
        assert data.path == "/fake/path.gguf"
        assert data.catalog is not None
        assert data.catalog.display_name == "Qwen3 0.6B"
        assert data.manifest is not None
        assert data.manifest.task == "chat"

    def test_catalog_only_not_installed(self, empty_manager):
        entry = _catalog_model(name="qwen3", tag="8b")
        with (
            patch("lilbee.cli.model._native_manifest_index", return_value={}),
            patch("lilbee.catalog.find_catalog_entry", return_value=entry),
        ):
            data = model_mod.show_model_data("qwen3:8b")
        assert data.installed is False
        assert data.catalog is not None
        assert data.manifest is None

    def test_unknown_ref_raises_not_found(self, empty_manager):
        with (
            patch("lilbee.cli.model._native_manifest_index", return_value={}),
            patch("lilbee.catalog.find_catalog_entry", return_value=None),
            pytest.raises(ModelNotFoundError, match="model not found: ghost"),
        ):
            model_mod.show_model_data("ghost:latest")


class TestResolveNativePath:
    def test_returns_path_when_registry_resolves(self, tmp_path):
        fake_registry = MagicMock()
        fake_registry.resolve.return_value = tmp_path / "blob.gguf"
        with patch("lilbee.registry.ModelRegistry", return_value=fake_registry):
            path = model_mod._resolve_native_path("qwen3:0.6b")
        assert path == str(tmp_path / "blob.gguf")

    def test_suppresses_key_error_from_missing_blob(self):
        fake_registry = MagicMock()
        fake_registry.resolve.side_effect = KeyError("no blob")
        with patch("lilbee.registry.ModelRegistry", return_value=fake_registry):
            path = model_mod._resolve_native_path("qwen3:0.6b")
        assert path is None

    def test_suppresses_value_error_from_invalid_ref(self):
        fake_registry = MagicMock()
        fake_registry.resolve.side_effect = ValueError("bad ref")
        with patch("lilbee.registry.ModelRegistry", return_value=fake_registry):
            path = model_mod._resolve_native_path("qwen3:0.6b")
        assert path is None


class TestShowCmd:
    def test_human_output_installed(self, fake_manager, native_manifests):
        entry = _catalog_model()
        with (
            patch("lilbee.catalog.find_catalog_entry", return_value=entry),
            patch(
                "lilbee.cli.model._resolve_native_path",
                return_value="/fake/path.gguf",
            ),
        ):
            result = runner.invoke(app, ["model", "show", "qwen3:0.6b"])
        assert result.exit_code == 0, result.output
        assert "source:" in result.output
        assert "/fake/path.gguf" in result.output
        assert "downloaded:" in result.output

    def test_json_output_roundtrips(self, fake_manager, native_manifests):
        entry = _catalog_model()
        with (
            patch("lilbee.catalog.find_catalog_entry", return_value=entry),
            patch("lilbee.cli.model._resolve_native_path", return_value="/p.gguf"),
        ):
            result = runner.invoke(app, ["--json", "model", "show", "qwen3:0.6b"])
        assert result.exit_code == 0, result.output
        parsed = ShowModelResult.model_validate_json(result.output)
        assert parsed.installed is True
        assert parsed.catalog is not None
        assert parsed.catalog.display_name == "Qwen3 0.6B"
        assert parsed.path == "/p.gguf"

    def test_json_not_found_exits_1(self, empty_manager):
        with (
            patch("lilbee.cli.model._native_manifest_index", return_value={}),
            patch("lilbee.catalog.find_catalog_entry", return_value=None),
        ):
            result = runner.invoke(app, ["--json", "model", "show", "ghost:1"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert "model not found" in payload["error"]

    def test_human_not_found_exits_1(self, empty_manager):
        with (
            patch("lilbee.cli.model._native_manifest_index", return_value={}),
            patch("lilbee.catalog.find_catalog_entry", return_value=None),
        ):
            result = runner.invoke(app, ["model", "show", "ghost:1"])
        assert result.exit_code == 1
        assert "model not found" in result.output


class TestPullModelData:
    def test_already_installed_short_circuits(self, fake_manager, native_manifests):
        result = model_mod.pull_model_data("qwen3:0.6b", ModelSource.NATIVE)
        assert isinstance(result, PullResult)
        assert result.status == PullStatus.ALREADY_INSTALLED
        assert fake_manager.pull_calls == []

    def test_pull_native_invokes_manager_and_callbacks(self, fake_manager, native_manifests):
        events = []
        result = model_mod.pull_model_data("qwen3:8b", ModelSource.NATIVE, on_update=events.append)
        assert result.status == PullStatus.OK
        assert result.path == "/fake/qwen3:8b.gguf"
        assert events
        assert events[0].percent == 50
        assert fake_manager.pull_calls == [("qwen3:8b", ModelSource.NATIVE)]

    def test_build_pull_callbacks_none_when_no_on_update(self):
        dict_cb, bytes_cb = model_mod._build_pull_callbacks(None)
        assert dict_cb is None
        assert bytes_cb is None

    def test_pull_litellm_adapts_dict_events(self, native_manifests):
        events = []

        class _Litellm(_FakeManager):
            def pull(self, model, source, *, on_progress=None, on_bytes=None):
                self.pull_calls.append((model, source))
                if on_progress is not None:
                    on_progress({"status": "pulling", "completed": 25, "total": 100})
                return None

        manager = _Litellm()
        with patch("lilbee.model_manager.get_model_manager", return_value=manager):
            result = model_mod.pull_model_data(
                "llama3:latest", ModelSource.LITELLM, on_update=events.append
            )
        assert result.status == PullStatus.OK
        assert result.path is None
        assert events
        assert events[0].percent == 25
        assert events[0].detail == "pulling"


class TestPullCmd:
    def test_json_stream_emits_done_event(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["--json", "model", "pull", "qwen3:8b"])
        assert result.exit_code == 0, result.output
        lines = [line for line in result.output.splitlines() if line.strip()]
        parsed = [json.loads(line) for line in lines]
        assert parsed[-1]["event"] == PullEvent.DONE.value
        assert parsed[-1]["status"] == PullStatus.OK.value
        assert parsed[-1]["model"] == "qwen3:8b"

    def test_human_mode_prints_pulled(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "pull", "qwen3:8b"])
        assert result.exit_code == 0, result.output
        assert "Pulled" in result.output

    def test_human_already_installed_message(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "pull", "qwen3:0.6b"])
        assert result.exit_code == 0
        assert "already installed" in result.output

    def test_runtime_error_json(self, native_manifests):
        manager = _FakeManager()
        manager.pull = MagicMock(side_effect=RuntimeError("no network"))
        with patch("lilbee.model_manager.get_model_manager", return_value=manager):
            result = runner.invoke(app, ["--json", "model", "pull", "qwen3:8b"])
        assert result.exit_code == 1
        payload = json.loads(result.output.strip().splitlines()[-1])
        assert payload == {"error": "no network"}

    def test_runtime_error_human(self, native_manifests):
        manager = _FakeManager()
        manager.pull = MagicMock(side_effect=RuntimeError("boom"))
        with patch("lilbee.model_manager.get_model_manager", return_value=manager):
            result = runner.invoke(app, ["model", "pull", "qwen3:8b"])
        assert result.exit_code == 1
        assert "boom" in result.output

    def test_invalid_source(self, fake_manager):
        result = runner.invoke(app, ["model", "pull", "qwen3:8b", "--source", "bad"])
        assert result.exit_code != 0


class TestRemoveModelData:
    def test_removes_and_reports_freed(self, fake_manager, native_manifests):
        result = model_mod.remove_model_data("qwen3:0.6b")
        assert isinstance(result, RemoveResult)
        assert result.deleted is True
        assert result.freed_gb == 5.0
        assert fake_manager.remove_calls == [("qwen3:0.6b", None)]

    def test_missing_manifest_returns_zero_freed(self, fake_manager):
        with patch("lilbee.cli.model._native_manifest_index", return_value={}):
            result = model_mod.remove_model_data("qwen3:0.6b")
        assert result.deleted is True
        assert result.freed_gb == 0.0


class TestRmCmd:
    def test_confirm_declined(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "rm", "qwen3:0.6b"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert fake_manager.remove_calls == []

    def test_confirm_accepted(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "rm", "qwen3:0.6b"], input="y\n")
        assert result.exit_code == 0
        assert "5.00 GB freed" in result.output
        assert fake_manager.remove_calls == [("qwen3:0.6b", None)]

    def test_yes_flag_skips_prompt(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "rm", "--yes", "qwen3:0.6b"])
        assert result.exit_code == 0
        assert "Removed" in result.output

    def test_not_found_exits_1(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["model", "rm", "--yes", "ghost:1.0"])
        assert result.exit_code == 1
        assert "Not found" in result.output

    def test_json_output_serializes_remove_result(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["--json", "model", "rm", "qwen3:0.6b"])
        assert result.exit_code == 0
        parsed = RemoveResult.model_validate_json(result.output)
        assert parsed.deleted is True
        assert parsed.freed_gb == 5.0

    def test_json_not_found_exits_1(self, fake_manager, native_manifests):
        result = runner.invoke(app, ["--json", "model", "rm", "ghost:1.0"])
        assert result.exit_code == 1
        parsed = RemoveResult.model_validate_json(result.output)
        assert parsed.deleted is False

    def test_invalid_source(self, fake_manager):
        result = runner.invoke(app, ["model", "rm", "--yes", "qwen3:0.6b", "--source", "bad"])
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
        run_tui.assert_called_once_with(auto_sync=False, initial_view="Catalog")


class TestCatalogEntryDataFactory:
    def test_from_catalog_model_maps_fields(self):
        entry = _catalog_model()
        data = CatalogEntryData.from_catalog_model(entry)
        assert data.ref == "qwen3:0.6b"
        assert data.hf_repo == "Qwen/Qwen3-0.6B-GGUF"
        assert data.featured is True
        assert data.recommended is True


class TestManifestDataFactory:
    def test_from_manifest_computes_size_gb(self):
        manifest = _manifest("qwen3", "0.6b", size=3 * 1024**3, task="chat")
        data = ManifestData.from_manifest(manifest)
        assert data.size_gb == 3.0
        assert data.name == "qwen3:0.6b"
        assert data.source_repo == "org/qwen3-GGUF"


class TestNativeManifestIndex:
    def test_indexes_by_ref(self, tmp_path):
        fake_registry = MagicMock()
        fake_registry.list_installed.return_value = [
            _manifest("qwen3", "0.6b", size=1024, task="chat"),
        ]
        with patch("lilbee.registry.ModelRegistry", return_value=fake_registry):
            index = model_mod._native_manifest_index()
        assert "qwen3:0.6b" in index
        assert index["qwen3:0.6b"].task == "chat"
