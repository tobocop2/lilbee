"""Tests for providers.model_ref: model reference parsing and option translation."""

from __future__ import annotations

import pytest

from lilbee.modelhub.model_manager.types import RemoteModel
from lilbee.providers.model_ref import format_remote_ref, parse_model_ref, translate_options

# Canonical native HF ref for tests that need a local model.
_LOCAL_REF = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"


class TestParseModelRef:
    def test_native_hf_ref(self) -> None:
        ref = parse_model_ref(_LOCAL_REF)
        assert ref.provider == "local"
        assert ref.name == _LOCAL_REF
        assert ref.raw == _LOCAL_REF

    def test_bare_hf_repo_is_local(self) -> None:
        """A bare ``<org>/<repo>`` (no filename) is treated as a local ref."""
        ref = parse_model_ref("Qwen/Qwen3-8B-GGUF")
        assert ref.provider == "local"
        assert ref.name == "Qwen/Qwen3-8B-GGUF"

    def test_ollama_prefix(self) -> None:
        ref = parse_model_ref("ollama/qwen3:8b")
        assert ref.provider == "ollama"
        assert ref.name == "qwen3:8b"

    def test_ollama_prefix_bare_name(self) -> None:
        ref = parse_model_ref("ollama/qwen3")
        assert ref.provider == "ollama"
        assert ref.name == "qwen3:latest"

    def test_openai_prefix(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        assert ref.provider == "openai"
        assert ref.name == "gpt-4o"

    def test_anthropic_prefix(self) -> None:
        ref = parse_model_ref("anthropic/claude-sonnet-4-20250514")
        assert ref.provider == "anthropic"
        assert ref.name == "claude-sonnet-4-20250514"

    def test_gemini_prefix(self) -> None:
        ref = parse_model_ref("gemini/gemini-2.5-pro")
        assert ref.provider == "gemini"
        assert ref.name == "gemini-2.5-pro"

    def test_bare_name_tag_rejected(self) -> None:
        """A bare ``name:tag`` lacks a provider prefix and is rejected."""
        with pytest.raises(ValueError, match="must be a HuggingFace ref"):
            parse_model_ref("qwen3:0.6b")

    def test_unprefixed_bare_name_rejected(self) -> None:
        """A bare name with no ``/`` is rejected."""
        with pytest.raises(ValueError, match="must be a HuggingFace ref"):
            parse_model_ref("qwen3")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_model_ref("")


class TestProviderModelRefProperties:
    def test_api_model_is_api(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        assert ref.is_api is True
        assert ref.is_local is False
        assert ref.is_remote is True

    def test_local_model_is_local(self) -> None:
        ref = parse_model_ref(_LOCAL_REF)
        assert ref.is_local is True
        assert ref.is_api is False
        assert ref.is_remote is False

    def test_ollama_model_is_remote(self) -> None:
        ref = parse_model_ref("ollama/qwen3:8b")
        assert ref.is_remote is True
        assert ref.is_api is False
        assert ref.is_local is False

    def test_api_model_does_not_need_api_base(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        assert ref.needs_api_base is False

    def test_local_model_needs_api_base(self) -> None:
        ref = parse_model_ref(_LOCAL_REF)
        assert ref.needs_api_base is True

    def test_ollama_model_needs_api_base(self) -> None:
        ref = parse_model_ref("ollama/qwen3:8b")
        assert ref.needs_api_base is True


class TestForOpenaiPrefix:
    def test_ollama_model(self) -> None:
        ref = parse_model_ref("ollama/qwen3:8b")
        assert ref.for_openai_prefix() == "ollama/qwen3:8b"

    def test_openai_model(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        assert ref.for_openai_prefix() == "openai/gpt-4o"

    def test_anthropic_model(self) -> None:
        ref = parse_model_ref("anthropic/claude-sonnet-4-20250514")
        assert ref.for_openai_prefix() == "anthropic/claude-sonnet-4-20250514"

    def test_local_model(self) -> None:
        ref = parse_model_ref(_LOCAL_REF)
        assert ref.for_openai_prefix() == _LOCAL_REF


class TestForDisplay:
    def test_preserves_raw(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        assert ref.for_display() == "openai/gpt-4o"


class TestFormatRemoteRef:
    def test_openai_provider_lowercases_and_prefixes(self) -> None:
        model = RemoteModel(
            name="gpt-4o", task="chat", family="", parameter_size="", provider="OpenAI"
        )
        assert format_remote_ref(model.name, model.provider) == "openai/gpt-4o"

    def test_anthropic_provider(self) -> None:
        model = RemoteModel(
            name="claude-sonnet-4-20250514",
            task="chat",
            family="",
            parameter_size="",
            provider="Anthropic",
        )
        assert format_remote_ref(model.name, model.provider) == "anthropic/claude-sonnet-4-20250514"

    def test_ollama_provider_uses_ollama_prefix(self) -> None:
        model = RemoteModel(
            name="qwen3:8b",
            task="chat",
            family="",
            parameter_size="",
            provider="Ollama",
        )
        assert format_remote_ref(model.name, model.provider) == "ollama/qwen3:8b"


class TestTranslateOptions:
    def test_api_model_strips_local_options(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        opts = {"temperature": 0.7, "num_predict": 1024, "num_ctx": 4096, "top_k": 40}
        result = translate_options(opts, ref)
        assert result == {"temperature": 0.7, "max_tokens": 1024}
        assert "num_predict" not in result
        assert "num_ctx" not in result
        assert "top_k" not in result

    def test_local_model_keeps_options(self) -> None:
        ref = parse_model_ref(_LOCAL_REF)
        opts = {"temperature": 0.7, "num_predict": 1024, "num_ctx": 4096}
        result = translate_options(opts, ref)
        assert result == {"temperature": 0.7, "num_predict": 1024, "num_ctx": 4096}

    def test_api_model_without_num_predict(self) -> None:
        ref = parse_model_ref("anthropic/claude-sonnet-4-20250514")
        opts = {"temperature": 0.5}
        result = translate_options(opts, ref)
        assert result == {"temperature": 0.5}

    def test_empty_options(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        result = translate_options({}, ref)
        assert result == {}
