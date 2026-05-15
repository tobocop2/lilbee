"""Tests for ``lilbee.modelhub.role_validator``."""

import os
from unittest import mock

import pytest

from tests.conftest import install_fake_model

_DEFAULT_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"


@pytest.fixture()
def _task_validation_enabled():
    """Unset the conftest-level bypass so validate_model_task_assignment fires."""
    prev = os.environ.pop("LILBEE_SKIP_MODEL_TASK_VALIDATION", None)
    try:
        yield
    finally:
        if prev is not None:
            os.environ["LILBEE_SKIP_MODEL_TASK_VALIDATION"] = prev


class TestValidateModelTaskAssignment:
    """The single write-boundary check for role-slot assignment."""

    def test_chat_slot_accepts_chat_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        result = validate_model_task_assignment("chat_model", _DEFAULT_CHAT_REF)
        assert result == _DEFAULT_CHAT_REF

    def test_chat_slot_rejects_vision_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        vision = "noctrex/LightOnOCR-2-1B-GGUF/lightonocr-Q4_K_M.gguf"
        with pytest.raises(ValueError, match="vision"):
            validate_model_task_assignment("chat_model", vision)

    def test_chat_slot_rejects_reranker_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        rerank = "gpustack/bge-reranker-v2-m3-GGUF/bge-reranker-Q4_K_M.gguf"
        with pytest.raises(ValueError, match="rerank"):
            validate_model_task_assignment("chat_model", rerank)

    def test_embedding_slot_rejects_chat_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with pytest.raises(ValueError, match="chat"):
            validate_model_task_assignment("embedding_model", _DEFAULT_CHAT_REF)

    def test_vision_slot_rejects_chat_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with pytest.raises(ValueError, match="chat"):
            validate_model_task_assignment("vision_model", _DEFAULT_CHAT_REF)

    def test_reranker_slot_rejects_vision_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        vision = "noctrex/LightOnOCR-2-1B-GGUF/lightonocr-Q4_K_M.gguf"
        with pytest.raises(ValueError, match="vision"):
            validate_model_task_assignment("reranker_model", vision)

    def test_empty_string_passes_through(self, _task_validation_enabled):
        """Empty or whitespace refs bypass validation (role unset)."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        assert validate_model_task_assignment("vision_model", "") == ""
        assert validate_model_task_assignment("reranker_model", "   ") == "   "

    def test_provider_prefix_bypasses_catalog(self, _task_validation_enabled):
        """Provider-prefixed refs (ollama/, openai/, ...) bypass the featured
        catalog check entirely; routing handles task taxonomy at the wire.
        """
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        ref = "ollama/qwen3:0.6b"
        assert validate_model_task_assignment("chat_model", ref) == ref

    def test_bare_hf_repo_canonicalizes_to_catalog_ref(self, _task_validation_enabled):
        """A bare ``hf_repo`` resolves to the catalog entry's ref (= the repo)."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        result = validate_model_task_assignment(
            "reranker_model", "gpustack/bge-reranker-v2-m3-GGUF"
        )
        assert result == "gpustack/bge-reranker-v2-m3-GGUF"

    def test_out_of_catalog_rejected(self, _task_validation_enabled):
        """Refs that are neither featured nor installed are rejected as not installed."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with pytest.raises(ValueError, match="not installed"):
            validate_model_task_assignment("chat_model", "totally-unknown-model:99b")

    def test_skip_env_var_disables_check(self, tmp_path):
        """LILBEE_SKIP_MODEL_TASK_VALIDATION bypasses the role check when pytest is imported."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with mock.patch.dict(os.environ, {"LILBEE_SKIP_MODEL_TASK_VALIDATION": "1"}):
            # Bypass: returns input unchanged, does not raise.
            result = validate_model_task_assignment("chat_model", "totally-unknown-model:99b")
        assert result == "totally-unknown-model:99b"

    def test_skip_env_var_alone_does_not_bypass_in_production(self, tmp_path):
        """Shell-level env var without the pytest sentinel must not bypass validation."""
        import sys

        from lilbee.modelhub.role_validator import validate_model_task_assignment

        saved_pytest = sys.modules.pop("pytest", None)
        try:
            with (
                mock.patch.dict(os.environ, {"LILBEE_SKIP_MODEL_TASK_VALIDATION": "1"}),
                pytest.raises(ValueError, match="not installed"),
            ):
                validate_model_task_assignment("chat_model", "totally-unknown-model:99b")
        finally:
            if saved_pytest is not None:
                sys.modules["pytest"] = saved_pytest

    def test_task_mismatch_carries_structured_fields(self, _task_validation_enabled):
        """TaskMismatchError carries the structured fields each surface needs to format messages."""
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.role_validator import TaskMismatchError, validate_model_task_assignment

        vision = "noctrex/LightOnOCR-2-1B-GGUF"
        with pytest.raises(TaskMismatchError) as exc_info:
            validate_model_task_assignment("chat_model", vision)

        err = exc_info.value
        assert err.ref == vision
        assert err.entry_task == ModelTask.VISION
        assert err.expected_task == ModelTask.CHAT

    def test_installed_non_featured_chat_model_accepted(self, _task_validation_enabled):
        """A non-featured chat model installed locally is a valid chat_model assignment."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        ref = install_fake_model(
            "MaziyarPanahi/Qwen3-1.7B-GGUF", "Qwen3-1.7B.Q4_K_M.gguf", task="chat"
        )
        assert validate_model_task_assignment("chat_model", ref) == ref

    def test_installed_non_featured_wrong_role_rejected(self, _task_validation_enabled):
        """An installed non-featured chat model in the reranker slot raises TaskMismatchError."""
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.role_validator import TaskMismatchError, validate_model_task_assignment

        ref = install_fake_model(
            "MaziyarPanahi/Qwen3-1.7B-GGUF", "Qwen3-1.7B.Q4_K_M.gguf", task="chat"
        )
        with pytest.raises(TaskMismatchError) as exc_info:
            validate_model_task_assignment("reranker_model", ref)
        assert exc_info.value.entry_task == ModelTask.CHAT
        assert exc_info.value.expected_task == ModelTask.RERANK
