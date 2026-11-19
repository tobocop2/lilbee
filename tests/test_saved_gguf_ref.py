"""A model setting saved with the ``#gguf`` qualifier reads like the bare ref."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from conftest import PICKS_CHAT, install_fake_model
from lilbee.catalog.refs import canonical_ref
from lilbee.modelhub.role_validator import validate_model_task_assignment
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.model_ref import is_native_gguf_ref, routes_to_native_gguf
from lilbee.providers.serving import is_native_rerank_ref
from lilbee.server import handlers

_CHAT_REPO = "other/Chat-Model-GGUF"
_CHAT_FILE = "chat-Q4_K_M.gguf"
_RERANK_REPO = "gpustack/bge-reranker-v2-m3-GGUF"
_RERANK_FILE = "bge-Q4_K_M.gguf"
_EMBED_REPO = "embed/Test-Embedding-GGUF"
_EMBED_REF = f"{_EMBED_REPO}/test-embedding-Q4_K_M.gguf"


class TestCanonicalRef:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("org/repo#gguf", "org/repo"),
            ("org/repo/f.gguf#gguf", "org/repo/f.gguf"),
            ("org/repo", "org/repo"),
            ("org/repo/f.gguf", "org/repo/f.gguf"),
            ("org/repo#mlx", "org/repo#mlx"),
            ("ollama/qwen3:0.6b", "ollama/qwen3:0.6b"),
            ("org/repo#nope", "org/repo#nope"),
            ("/models/My#Models/f.gguf", "/models/My#Models/f.gguf"),
        ],
    )
    def test_reads_the_catalog_parser_and_keeps_a_malformed_ref(
        self, raw: str, expected: str
    ) -> None:
        assert canonical_ref(raw) == expected


class TestModelRefClassifier:
    @pytest.mark.parametrize(
        ("raw", "native"),
        [
            ("org/repo/f.gguf#gguf", True),
            ("openai/repo/f.gguf#gguf", True),
            ("/models/f.gguf#gguf", True),
            ("org/repo#gguf", False),
            ("org/repo/f.gguf#nope", False),
        ],
    )
    def test_a_gguf_qualified_ref_classifies_as_its_bare_ref(self, raw: str, native: bool) -> None:
        assert is_native_gguf_ref(raw) is native

    def test_a_gguf_qualified_ref_routes_native_unless_a_local_server_owns_it(self) -> None:
        assert routes_to_native_gguf("openai/repo/f.gguf#gguf") is True
        assert routes_to_native_gguf("ollama/repo/f.gguf#gguf") is False


class TestRoleValidator:
    def test_a_gguf_qualified_bare_repo_resolves_to_the_installed_quant(self) -> None:
        ref = install_fake_model(_CHAT_REPO, _CHAT_FILE, "chat")
        result = validate_model_task_assignment(
            "chat_model", f"{_CHAT_REPO}#gguf", allow_bypass=False
        )
        assert result == ref

    def test_a_gguf_qualified_file_ref_is_saved_as_the_bare_file_ref(self) -> None:
        ref = install_fake_model(_CHAT_REPO, _CHAT_FILE, "chat")
        result = validate_model_task_assignment("chat_model", f"{ref}#gguf", allow_bypass=False)
        assert result == ref

    def test_a_gguf_qualified_pick_file_ref_keeps_its_exact_quant(self) -> None:
        full = f"{PICKS_CHAT[0].hf_repo}/some-quant-Q4_K_M.gguf"
        result = validate_model_task_assignment("chat_model", f"{full}#gguf", allow_bypass=False)
        assert result == full

    def test_a_malformed_qualifier_is_refused_as_not_installed(self) -> None:
        install_fake_model(_CHAT_REPO, _CHAT_FILE, "chat")
        with pytest.raises(ValueError, match="not installed"):
            validate_model_task_assignment("chat_model", f"{_CHAT_REPO}#nope", allow_bypass=False)


class TestServingCheck:
    def test_a_gguf_qualified_bare_repo_reads_as_installed_native(self) -> None:
        install_fake_model(_RERANK_REPO, _RERANK_FILE, "rerank")
        assert is_native_rerank_ref(f"{_RERANK_REPO}#gguf", LLAMA) is True

    def test_a_gguf_qualified_file_ref_reads_as_native(self) -> None:
        ref = install_fake_model(_RERANK_REPO, _RERANK_FILE, "rerank")
        assert is_native_rerank_ref(f"{ref}#gguf", LLAMA) is True

    def test_a_malformed_qualifier_is_not_native(self) -> None:
        install_fake_model(_RERANK_REPO, _RERANK_FILE, "rerank")
        assert is_native_rerank_ref(f"{_RERANK_REPO}#nope", LLAMA) is False


class TestModelsHandler:
    @patch("lilbee.server.handlers.models.get_services")
    async def test_a_gguf_qualified_bare_repo_resolves_to_the_available_quant(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        result = await handlers.set_embedding_model(f"{_EMBED_REPO}#gguf")
        assert result.model == _EMBED_REF

    @patch("lilbee.server.handlers.models.get_services")
    async def test_a_gguf_qualified_file_ref_matches_the_available_file(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        result = await handlers.set_embedding_model(f"{_EMBED_REF}#gguf")
        assert result.model == _EMBED_REF

    @patch("lilbee.server.handlers.models.get_services")
    async def test_a_malformed_qualifier_is_not_available(self, mock_svc):
        mock_svc.return_value.provider.list_models.return_value = [_EMBED_REF]
        with pytest.raises(ValueError, match="not available"):
            await handlers.set_embedding_model(f"{_EMBED_REPO}#nope")
