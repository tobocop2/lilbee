"""The chat model list hides a format that no installed engine loads."""

from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.models import installed_chat_model_refs
from lilbee.app.services import set_services
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.providers.fleet.engines.base import EngineName
from tests.conftest import make_mock_services

ENGINES = "lilbee.providers.fleet.engines.choice.installed_engines"
GGUF_REF = "acme/Chat-GGUF/chat-Q4_K_M.gguf"
MLX_REF = "acme/Chat-4bit#mlx"
OPENVINO_REF = "acme/Chat-int4#openvino"
EMBED_REF = "acme/Embed-GGUF/embed-Q4_K_M.gguf"

LLAMA_ONLY = {EngineName.LLAMA: {ModelFormat.GGUF}}
LLAMA_AND_MLX = {
    EngineName.LLAMA: {ModelFormat.GGUF},
    EngineName.MLX: {ModelFormat.MLX},
}


def _manifest(ref: str, task: ModelTask) -> MagicMock:
    manifest = MagicMock()
    manifest.ref = ref
    manifest.task = task
    return manifest


@pytest.fixture
def registry():
    services = make_mock_services()
    services.registry.list_installed = MagicMock(
        return_value=[
            _manifest(MLX_REF, ModelTask.CHAT),
            _manifest(GGUF_REF, ModelTask.CHAT),
            _manifest(OPENVINO_REF, ModelTask.CHAT),
            _manifest(EMBED_REF, ModelTask.EMBEDDING),
        ]
    )
    set_services(services)
    yield services
    set_services(None)


def _list_with(installed) -> list[str]:
    with patch(ENGINES, return_value=installed):
        return installed_chat_model_refs()


def test_llama_cpp_only_hides_non_gguf_chat_models(registry) -> None:
    assert _list_with(LLAMA_ONLY) == [GGUF_REF]


def test_an_engine_that_loads_mlx_shows_the_mlx_chat_model(registry) -> None:
    assert _list_with(LLAMA_AND_MLX) == [MLX_REF, GGUF_REF]


def test_no_installed_engine_lists_nothing(registry) -> None:
    assert _list_with({}) == []


def _manifest_with_files(ref: str, task: ModelTask) -> MagicMock:
    manifest = _manifest(ref, task)
    manifest.hf_repo = ref.split("/")[0] + "/" + ref.split("/")[1].split("#")[0]
    manifest.gguf_filename = "model.gguf"
    manifest.downloaded_at = "2026-05-15T00:00:00+00:00"
    return manifest


def test_models_endpoint_hides_a_format_no_engine_loads(registry) -> None:
    from lilbee.server.chat_completions_api.routes import _build_models_list_payload

    registry.provider.served_chat_ctx.return_value = None
    registry.provider.served_chat_slots.return_value = None
    registry.registry.list_installed = MagicMock(
        return_value=[
            _manifest_with_files(MLX_REF, ModelTask.CHAT),
            _manifest_with_files(GGUF_REF, ModelTask.CHAT),
        ]
    )
    with patch(ENGINES, return_value=LLAMA_ONLY):
        ids = [m.id for m in _build_models_list_payload().data]
    assert GGUF_REF in ids
    assert MLX_REF not in ids


def test_list_installed_models_hides_a_format_no_engine_loads() -> None:
    from lilbee.modelhub import models

    manifests = [
        _manifest(MLX_REF, ModelTask.CHAT),
        _manifest(GGUF_REF, ModelTask.CHAT),
    ]
    with (
        patch.object(models.ModelRegistry, "list_installed", return_value=manifests),
        patch.object(models, "_remote_chat_model_names", return_value=[]),
        patch(ENGINES, return_value=LLAMA_ONLY),
    ):
        assert models.list_installed_models() == [GGUF_REF]


def test_model_bar_picker_hides_a_format_no_engine_loads() -> None:
    from lilbee.cli.tui.widgets import model_bar
    from lilbee.modelhub.registry import ModelRegistry

    manifests = [
        _manifest_with_files(MLX_REF, ModelTask.CHAT),
        _manifest_with_files(GGUF_REF, ModelTask.CHAT),
    ]
    buckets: dict[ModelTask, list] = {task: [] for task in ModelTask}
    with (
        patch.object(ModelRegistry, "list_installed", return_value=manifests),
        patch.object(model_bar, "_has_vision_sidecar", return_value=False),
        patch(ENGINES, return_value=LLAMA_ONLY),
    ):
        model_bar._collect_native_models(buckets, set())
    assert [o.ref for o in buckets[ModelTask.CHAT]] == [GGUF_REF]
