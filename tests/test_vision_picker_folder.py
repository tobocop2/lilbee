"""A model folder is a vision model when its config has a vision_config and an engine serves it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import pytest

from lilbee.catalog import runnable
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.cli.tui.widgets.model_bar import ModelOption, _collect_native_models
from lilbee.core.config import cfg
from lilbee.modelhub.registry import ModelManifest, ModelRegistry
from lilbee.modelhub.role_validator import _installed_ref_and_task
from lilbee.providers import gguf_meta
from lilbee.providers.fleet import model_memory
from lilbee.providers.fleet.engines import mistral
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import EngineChoice, choose_engine, install_advice
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.host_caps import HostCaps, collect
from lilbee.providers.roles import WorkerRole
from tests.test_registry import _FAKE_REV, _seed_hf_cache

_WEIGHTS = "model.safetensors"
_CONTENT = b"weights"
_VISION = {"vision_config": {}}
_TEXT = {"architectures": ["Qwen3ForCausalLM"]}
_MLX = ModelFormat.MLX
_SAFETENSORS = ModelFormat.SAFETENSORS
_TOO_DEEP = "[" * 200000


def _ref(fmt: ModelFormat) -> str:
    return f"org/folder-{fmt.value}#{fmt.value}"


def _install_folder(
    root: Path, config: object, fmt: ModelFormat, task: ModelTask = ModelTask.CHAT
) -> None:
    """Install a model folder under *root* the way a pull lays it out."""
    repo = f"org/folder-{fmt.value}"
    _seed_hf_cache(root, repo=repo, filename=_WEIGHTS, content=_CONTENT)
    snapshot = root / f"models--org--folder-{fmt.value}" / "snapshots" / _FAKE_REV
    text = config if isinstance(config, str) else json.dumps(config)
    (snapshot / "config.json").write_text(text, encoding="utf-8")
    ModelRegistry(root)._write_manifest(
        ModelManifest(
            hf_repo=repo,
            gguf_filename=_WEIGHTS,
            size_bytes=len(_CONTENT),
            task=task,
            downloaded_at="2026-11-18T00:00:00+00:00",
            blob=hashlib.sha256(_CONTENT).hexdigest(),
            format=fmt,
        )
    )


def _vision_refs() -> list[str]:
    buckets: dict[ModelTask, list[ModelOption]] = {task: [] for task in ModelTask}
    _collect_native_models(buckets, set())
    return [option.ref for option in buckets[ModelTask.VISION]]


@pytest.fixture(autouse=True)
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The installed engines: mlx, mistral and tei. A test removes one to uninstall it."""
    monkeypatch.setattr(cfg, "models_dir", tmp_path)
    engines = {
        EngineName.MLX: frozenset({ModelFormat.MLX}),
        EngineName.MISTRAL: frozenset({ModelFormat.GGUF, ModelFormat.SAFETENSORS}),
        EngineName.TEI: frozenset({ModelFormat.SAFETENSORS}),
    }
    with mock.patch(
        "lilbee.providers.fleet.engines.choice.installed_engines", return_value=engines
    ):
        yield engines


class TestDeclaresVisionConfig:
    def test_a_vision_config_section_counts(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text('{"vision_config": {}}', encoding="utf-8")
        assert gguf_meta.declares_vision_config(tmp_path)

    @pytest.mark.parametrize(
        "text",
        ['{"architectures": []}', '{"vision_config": null}', "[1]", "not json", _TOO_DEEP],
        ids=["no-section", "null-section", "not-a-mapping", "unreadable", "nested-too-deep"],
    )
    def test_anything_else_does_not(self, tmp_path: Path, text: str) -> None:
        (tmp_path / "config.json").write_text(text, encoding="utf-8")
        assert not gguf_meta.declares_vision_config(tmp_path)

    def test_a_folder_with_no_config_does_not(self, tmp_path: Path) -> None:
        assert not gguf_meta.declares_vision_config(tmp_path)


class TestAnEngineMustServeTheFolderAsVision:
    def test_a_safetensors_folder_is_offered_and_accepted(self, tmp_path: Path) -> None:
        _install_folder(tmp_path, _VISION, _SAFETENSORS)
        ref = _ref(_SAFETENSORS)
        assert _vision_refs() == [ref]
        assert _installed_ref_and_task(ref, ModelTask.VISION) == (ref, ModelTask.VISION)

    def test_an_mlx_folder_is_neither_offered_nor_accepted(self, tmp_path: Path) -> None:
        _install_folder(tmp_path, _VISION, _MLX)
        ref = _ref(_MLX)
        assert _vision_refs() == []
        assert _installed_ref_and_task(ref, ModelTask.VISION) == (ref, ModelTask.CHAT)

    def test_a_text_folder_is_neither_offered_nor_accepted(self, tmp_path: Path) -> None:
        _install_folder(tmp_path, _TEXT, _SAFETENSORS)
        ref = _ref(_SAFETENSORS)
        assert _vision_refs() == []
        assert _installed_ref_and_task(ref, ModelTask.VISION) == (ref, ModelTask.CHAT)

    def test_a_safetensors_folder_with_mistral_not_installed_is_not_vision(
        self, tmp_path: Path, installed: dict[EngineName, frozenset[ModelFormat]]
    ) -> None:
        del installed[EngineName.MISTRAL]
        _install_folder(tmp_path, _VISION, _SAFETENSORS)
        ref = _ref(_SAFETENSORS)
        assert _vision_refs() == []
        assert _installed_ref_and_task(ref, ModelTask.VISION) == (ref, ModelTask.CHAT)

    @pytest.mark.parametrize("task", [ModelTask.EMBEDDING, ModelTask.RERANK])
    def test_a_folder_of_another_task_is_neither_offered_nor_accepted(
        self, tmp_path: Path, task: ModelTask
    ) -> None:
        _install_folder(tmp_path, _VISION, _SAFETENSORS, task)
        ref = _ref(_SAFETENSORS)
        assert _vision_refs() == []
        assert _installed_ref_and_task(ref, ModelTask.VISION) == (ref, task)

    def test_a_config_nested_too_deep_is_not_vision(self, tmp_path: Path) -> None:
        _install_folder(tmp_path, _TOO_DEEP, _SAFETENSORS)
        ref = _ref(_SAFETENSORS)
        assert not ModelRegistry(tmp_path).has_vision_projector(ref)
        assert _vision_refs() == []
        assert _installed_ref_and_task(ref, ModelTask.VISION) == (ref, ModelTask.CHAT)


def _no_host() -> HostCaps:
    raise AssertionError("the host is not read when no installed engine loads the format")


def _vision_choice(
    installed: dict[EngineName, frozenset[ModelFormat]],
    host: Callable[[], HostCaps] = runnable.session_host,
) -> EngineChoice:
    return choose_engine(
        fmt=ModelFormat.SAFETENSORS,
        host=host,
        installed=installed,
        preferred="auto",
        override=None,
        role=WorkerRole.VISION,
        model="org/folder",
    )


class TestTheMessageNamesTheEngineToInstall:
    def test_a_folder_loaded_by_an_engine_that_does_not_serve_vision(self) -> None:
        choice = _vision_choice({EngineName.TEI: frozenset({ModelFormat.SAFETENSORS})})
        assert choice.name is None
        assert choice.notice is not None
        assert choice.notice.startswith(
            "For vision: No installed engine serves the safetensors format of org/folder "
            "for vision."
        )
        assert f"{EngineName.MISTRAL.value} serves it." in choice.notice
        assert install_advice(mistral.MISTRAL, runnable.session_host()) in choice.notice
        assert "No installed engine loads" not in choice.notice

    def test_the_notice_reads_the_cached_session_host_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The install advice needs the host for the OS floor; it takes the session host that
        # the caller holds, so the live read (a child process) runs once, not per notice.
        live = runnable.detect
        reads: list[HostCaps] = []

        def counted() -> HostCaps:
            reads.append(live())
            return reads[-1]

        monkeypatch.setattr(runnable, "detect", counted)
        installed = {EngineName.TEI: frozenset({ModelFormat.SAFETENSORS})}
        first = _vision_choice(installed)
        assert _vision_choice(installed) == first
        assert first.notice is not None
        assert "No installed engine serves" in first.notice
        assert len(reads) == 1

    def test_a_server_the_host_refuses_is_not_called_missing(self) -> None:
        windows = collect(
            system="Windows", machine="AMD64", cpu_info={"flags": ["avx2"]}, nvidia=None
        )
        choice = choose_engine(
            fmt=ModelFormat.SAFETENSORS,
            host=lambda: windows,
            installed={EngineName.TEI: frozenset({ModelFormat.SAFETENSORS})},
            preferred="auto",
            override=None,
            role=WorkerRole.EMBED,
            model="org/folder",
        )
        assert choice.notice == (
            "For embed: Engine tei loads the safetensors format of org/folder "
            "but cannot run on this host. No build exists for Windows."
        )

    def test_no_role_names_the_refused_engine_without_a_role_prefix(self) -> None:
        # Only TEI is installed, and the host refuses it. Without a role no server is named.
        windows = collect(
            system="Windows", machine="AMD64", cpu_info={"flags": ["avx2"]}, nvidia=None
        )
        choice = choose_engine(
            fmt=ModelFormat.GGUF,
            host=lambda: windows,
            installed={EngineName.TEI: frozenset({ModelFormat.GGUF})},
            preferred="auto",
            override=None,
            model="org/folder",
        )
        assert choice.notice == (
            "Engine tei loads the gguf format of org/folder "
            "but cannot run on this host. No build exists for Windows."
        )

    def test_a_format_no_installed_engine_loads_keeps_the_load_message(self) -> None:
        choice = _vision_choice({}, _no_host)
        assert choice.notice is not None
        assert "No installed engine loads the safetensors format of org/folder." in choice.notice


@pytest.mark.parametrize(
    "read",
    [MLX.model_meta, model_memory.installed_config],
    ids=["mlx", "memory"],
)
def test_a_config_nested_too_deep_reads_as_no_config(
    tmp_path: Path, read: Callable[[Path], object]
) -> None:
    (tmp_path / "config.json").write_text(_TOO_DEEP, encoding="utf-8")
    assert read(tmp_path) is None
