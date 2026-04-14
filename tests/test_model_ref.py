"""Tests for providers.model_ref — model reference parsing and option translation."""

from __future__ import annotations

from lilbee.providers.model_ref import parse_model_ref, translate_options


class TestParseModelRef:
    def test_local_bare_name(self) -> None:
        ref = parse_model_ref("qwen3")
        assert ref.provider == "local"
        assert ref.name == "qwen3:latest"
        assert ref.raw == "qwen3"

    def test_local_with_tag(self) -> None:
        ref = parse_model_ref("qwen3:0.6b")
        assert ref.provider == "local"
        assert ref.name == "qwen3:0.6b"

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

    def test_unknown_prefix_treated_as_local(self) -> None:
        ref = parse_model_ref("maternion/LightOnOCR-2")
        assert ref.provider == "local"
        assert ref.name == "maternion/LightOnOCR-2:latest"

    def test_unknown_prefix_with_tag(self) -> None:
        ref = parse_model_ref("maternion/LightOnOCR-2:v1")
        assert ref.provider == "local"
        assert ref.name == "maternion/LightOnOCR-2:v1"

    def test_empty_string(self) -> None:
        ref = parse_model_ref("")
        assert ref.provider == "local"
        assert ref.name == ":latest"


class TestProviderModelRefProperties:
    def test_api_model_is_api(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        assert ref.is_api is True
        assert ref.is_local is False
        assert ref.needs_litellm is True

    def test_local_model_is_local(self) -> None:
        ref = parse_model_ref("qwen3:8b")
        assert ref.is_local is True
        assert ref.is_api is False
        assert ref.needs_litellm is False

    def test_ollama_model_needs_litellm(self) -> None:
        ref = parse_model_ref("ollama/qwen3:8b")
        assert ref.needs_litellm is True
        assert ref.is_api is False
        assert ref.is_local is False

    def test_api_model_does_not_need_api_base(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        assert ref.needs_api_base is False

    def test_local_model_needs_api_base(self) -> None:
        ref = parse_model_ref("qwen3:8b")
        assert ref.needs_api_base is True

    def test_ollama_model_needs_api_base(self) -> None:
        ref = parse_model_ref("ollama/qwen3:8b")
        assert ref.needs_api_base is True


class TestForLitellm:
    def test_ollama_model(self) -> None:
        ref = parse_model_ref("ollama/qwen3:8b")
        assert ref.for_litellm() == "ollama/qwen3:8b"

    def test_openai_model(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        assert ref.for_litellm() == "openai/gpt-4o"

    def test_anthropic_model(self) -> None:
        ref = parse_model_ref("anthropic/claude-sonnet-4-20250514")
        assert ref.for_litellm() == "anthropic/claude-sonnet-4-20250514"

    def test_local_model(self) -> None:
        ref = parse_model_ref("qwen3:8b")
        assert ref.for_litellm() == "qwen3:8b"


class TestForDisplay:
    def test_preserves_raw(self) -> None:
        ref = parse_model_ref("openai/gpt-4o")
        assert ref.for_display() == "openai/gpt-4o"


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
        ref = parse_model_ref("qwen3:8b")
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
