"""A model is offered for a role only when an installed engine serves that role for its format."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.app.models import installed_chat_model_refs
from lilbee.app.services import set_services
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.core.config import cfg
from lilbee.modelhub.install_state import InstallState
from lilbee.modelhub.model_manager.validation import _first_installed_local_ref
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import loadable_refs
from lilbee.providers.roles import WorkerRole
from tests.conftest import make_mock_services

ENGINES = "lilbee.providers.fleet.engines.choice.installed_engines"
SAFETENSORS_CHAT = "acme/Chat#safetensors"
SAFETENSORS_EMBED = "acme/Embed#safetensors"
GGUF_CHAT = "acme/Chat-GGUF/chat-Q4_K_M.gguf"

TEI_ONLY = {EngineName.TEI: frozenset({ModelFormat.SAFETENSORS})}
TEI_AND_LLAMA = {
    EngineName.TEI: frozenset({ModelFormat.SAFETENSORS}),
    EngineName.LLAMA: frozenset({ModelFormat.GGUF}),
}


def _manifest(ref: str, task: ModelTask) -> MagicMock:
    manifest = MagicMock()
    manifest.ref = ref
    manifest.task = task
    manifest.hf_repo = ref.split("#")[0]
    manifest.gguf_filename = "model.gguf"
    manifest.downloaded_at = "2026-05-15T00:00:00+00:00"
    return manifest


MANIFESTS = [
    _manifest(SAFETENSORS_CHAT, ModelTask.CHAT),
    _manifest(GGUF_CHAT, ModelTask.CHAT),
    _manifest(SAFETENSORS_EMBED, ModelTask.EMBEDDING),
]


@pytest.fixture
def registry():
    services = make_mock_services()
    services.registry.list_installed = MagicMock(return_value=MANIFESTS)
    set_services(services)
    yield services
    set_services(None)


class TestLoadableRefs:
    def test_an_embedding_engine_does_not_offer_a_chat_model(self) -> None:
        with patch(ENGINES, return_value=TEI_ONLY):
            assert loadable_refs([SAFETENSORS_CHAT], WorkerRole.CHAT) == []

    def test_an_embedding_engine_offers_an_embedding_model(self) -> None:
        with patch(ENGINES, return_value=TEI_ONLY):
            assert loadable_refs([SAFETENSORS_EMBED], WorkerRole.EMBED) == [SAFETENSORS_EMBED]

    def test_an_embedding_engine_offers_a_reranker(self) -> None:
        with patch(ENGINES, return_value=TEI_ONLY):
            assert loadable_refs([SAFETENSORS_CHAT], WorkerRole.RERANK) == [SAFETENSORS_CHAT]

    def test_no_role_keeps_the_format_only_check(self) -> None:
        with patch(ENGINES, return_value=TEI_ONLY):
            assert loadable_refs([SAFETENSORS_CHAT, GGUF_CHAT]) == [SAFETENSORS_CHAT]

    def test_an_engine_that_refuses_a_format_for_a_role_does_not_offer_it(self) -> None:
        mistral_only = {EngineName.MISTRAL: frozenset({ModelFormat.GGUF, ModelFormat.SAFETENSORS})}
        with patch(ENGINES, return_value=mistral_only):
            assert loadable_refs([GGUF_CHAT], WorkerRole.EMBED) == []
            assert loadable_refs([GGUF_CHAT], WorkerRole.CHAT) == [GGUF_CHAT]
            assert loadable_refs([SAFETENSORS_EMBED], WorkerRole.EMBED) == [SAFETENSORS_EMBED]

    def test_another_engine_keeps_its_own_models_in_the_chat_list(self) -> None:
        with patch(ENGINES, return_value=TEI_AND_LLAMA):
            refs = [SAFETENSORS_CHAT, GGUF_CHAT]
            assert loadable_refs(refs, WorkerRole.CHAT) == [GGUF_CHAT]


class TestEveryChatListing:
    def test_the_app_list_leaves_out_a_chat_model_only_tei_loads(self, registry) -> None:
        with patch(ENGINES, return_value=TEI_ONLY):
            assert installed_chat_model_refs() == []

    def test_the_app_list_keeps_a_gguf_chat_model_beside_tei(self, registry) -> None:
        with patch(ENGINES, return_value=TEI_AND_LLAMA):
            assert installed_chat_model_refs() == [GGUF_CHAT]

    def test_the_model_list_leaves_out_a_chat_model_only_tei_loads(self) -> None:
        from lilbee.modelhub import models

        with (
            patch.object(models.ModelRegistry, "list_installed", return_value=MANIFESTS),
            patch.object(models, "_remote_chat_model_names", return_value=[]),
            patch(ENGINES, return_value=TEI_ONLY),
        ):
            assert models.list_installed_models() == []

    def test_the_models_endpoint_leaves_out_a_chat_model_only_tei_loads(self, registry) -> None:
        from lilbee.server.chat_completions_api.routes import _build_models_list_payload

        registry.provider.served_chat_ctx.return_value = None
        registry.provider.served_chat_slots.return_value = None
        with patch(ENGINES, return_value=TEI_AND_LLAMA):
            ids = [m.id for m in _build_models_list_payload().data]
        assert ids == [GGUF_CHAT]


class TestSubstituteForARole:
    @pytest.fixture
    def repo(self, tmp_path: Path):
        from lilbee.modelhub.registry import ModelRegistry

        registry = ModelRegistry(tmp_path)
        with (
            patch.object(ModelRegistry, "list_installed", return_value=MANIFESTS),
            patch.object(cfg, "models_dir", tmp_path),
            patch(
                "lilbee.modelhub.model_manager.validation.install_state",
                return_value=InstallState.REGISTERED,
            ),
        ):
            yield registry

    def test_ask_finds_no_chat_substitute_when_only_tei_loads_the_model(self, repo) -> None:
        with patch.object(choice, "installed_engines", lambda: TEI_ONLY):
            assert _first_installed_local_ref(ModelTask.CHAT) is None

    def test_the_embedding_substitute_is_still_found_with_tei(self, repo) -> None:
        with patch.object(choice, "installed_engines", lambda: TEI_ONLY):
            assert _first_installed_local_ref(ModelTask.EMBEDDING) == SAFETENSORS_EMBED
