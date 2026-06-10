"""Tests for providers.model_ref: model reference parsing and option translation."""

from __future__ import annotations

import pytest

from lilbee.modelhub.model_manager.types import RemoteModel
from lilbee.providers.model_ref import (
    format_remote_ref,
    parse_model_ref,
    routes_to_native_gguf,
    translate_options,
    with_configured_remote_chat,
)

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

    @pytest.mark.parametrize("org", ["openai", "mistral", "deepseek"])
    def test_native_gguf_shape_wins_over_api_provider_prefix(self, org: str) -> None:
        """A real HF org colliding with an API provider prefix still routes locally for GGUFs."""
        raw = f"{org}/Some-Model-GGUF/some-model-Q4_K_M.gguf"
        ref = parse_model_ref(raw)
        assert ref.provider == "local"
        assert ref.name == raw

    def test_lm_studio_gguf_path_id_routes_to_lm_studio(self) -> None:
        """LM Studio 0.2.x reports model ids as relative GGUF paths; the prefix wins."""
        ref = parse_model_ref("lm_studio/TheBloke/phi-2-GGUF/phi-2.Q4_K_M.gguf")
        assert ref.provider == "lm_studio"
        assert ref.name == "TheBloke/phi-2-GGUF/phi-2.Q4_K_M.gguf"

    def test_ollama_gguf_path_id_routes_to_ollama(self) -> None:
        """An Ollama-prefixed id that looks like a GGUF path stays with Ollama."""
        ref = parse_model_ref("ollama/some/dir/model.gguf")
        assert ref.provider == "ollama"

    def test_native_gguf_shape_check(self) -> None:
        from lilbee.providers.model_ref import is_native_gguf_ref

        assert is_native_gguf_ref("openai/Repo-GGUF/file.gguf") is True
        # Case-sensitive on purpose: hf_repo_from_ref only matches ".gguf".
        assert is_native_gguf_ref("openai/Repo-GGUF/sub/file.GGUF") is False
        assert is_native_gguf_ref("openai/gpt-4o") is False
        assert is_native_gguf_ref("ollama/llama3:8b") is False
        assert is_native_gguf_ref("file.gguf") is False

    def test_ollama_prefix(self) -> None:
        ref = parse_model_ref("ollama/qwen3:8b")
        assert ref.provider == "ollama"
        assert ref.name == "qwen3:8b"

    def test_ollama_prefix_bare_name(self) -> None:
        ref = parse_model_ref("ollama/qwen3")
        assert ref.provider == "ollama"
        assert ref.name == "qwen3:latest"

    def test_lm_studio_prefix(self) -> None:
        ref = parse_model_ref("lm_studio/qwen2.5-7b-instruct")
        assert ref.provider == "lm_studio"
        assert ref.name == "qwen2.5-7b-instruct"

    def test_lm_studio_prefix_does_not_append_latest_tag(self) -> None:
        """LM Studio ids are used verbatim; only Ollama appends ``:latest``."""
        ref = parse_model_ref("lm_studio/some-model")
        assert ref.provider == "lm_studio"
        assert ref.name == "some-model"

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

    def test_openrouter_prefix_carries_nested_path(self) -> None:
        """OpenRouter model ids embed a nested path; only the leading segment is the provider."""
        ref = parse_model_ref("openrouter/anthropic/claude-3.5-sonnet")
        assert ref.provider == "openrouter"
        assert ref.name == "anthropic/claude-3.5-sonnet"
        assert ref.is_api is True
        assert ref.for_openai_prefix() == "openrouter/anthropic/claude-3.5-sonnet"

    def test_mistral_prefix(self) -> None:
        ref = parse_model_ref("mistral/codestral-latest")
        assert ref.provider == "mistral"
        assert ref.name == "codestral-latest"
        assert ref.is_api is True

    def test_deepseek_prefix(self) -> None:
        ref = parse_model_ref("deepseek/deepseek-chat")
        assert ref.provider == "deepseek"
        assert ref.name == "deepseek-chat"
        assert ref.is_api is True

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


class TestRoutesToNativeGguf:
    """The exemption-aware shape check shared by parsing and rerank routing."""

    def test_native_gguf_shape_routes_native(self) -> None:
        assert routes_to_native_gguf(_LOCAL_REF) is True

    def test_local_server_prefix_is_exempt_from_shape_rule(self) -> None:
        assert routes_to_native_gguf("lm_studio/TheBloke/phi-2-GGUF/phi-2.Q4_K_M.gguf") is False
        assert routes_to_native_gguf("ollama/some/dir/model.gguf") is False

    def test_non_gguf_ref_does_not_route_native(self) -> None:
        assert routes_to_native_gguf("openai/gpt-4o") is False


class TestWithConfiguredRemoteChat:
    """A model listing includes a remote-configured chat model."""

    def test_remote_configured_chat_model_is_prepended(self) -> None:
        refs = with_configured_remote_chat(["a/b/c.gguf"], "ollama/qwen3:8b")
        assert refs == ["ollama/qwen3:8b", "a/b/c.gguf"]

    def test_native_configured_chat_model_is_untouched(self) -> None:
        assert with_configured_remote_chat(["a/b/c.gguf"], "a/b/c.gguf") == ["a/b/c.gguf"]

    def test_already_listed_remote_ref_is_not_duplicated(self) -> None:
        refs = with_configured_remote_chat(["ollama/qwen3:8b"], "ollama/qwen3:8b")
        assert refs == ["ollama/qwen3:8b"]


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

    def test_lm_studio_model_is_remote_and_needs_api_base(self) -> None:
        ref = parse_model_ref("lm_studio/qwen2.5-7b-instruct")
        assert ref.is_remote is True
        assert ref.is_api is False
        assert ref.is_local is False
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

    def test_lm_studio_model(self) -> None:
        ref = parse_model_ref("lm_studio/qwen2.5-7b-instruct")
        assert ref.for_openai_prefix() == "lm_studio/qwen2.5-7b-instruct"

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

    def test_lm_studio_display_name_normalizes_to_routing_key(self) -> None:
        """The ``"LM Studio"`` display name must map to the ``lm_studio/`` prefix.

        Lower-casing alone would yield ``"lm studio"`` and silently drop the
        prefix, corrupting the stored ``chat_model`` ref; the display->key
        normalization in ``format_remote_ref`` prevents that.
        """
        model = RemoteModel(
            name="qwen2.5-7b-instruct",
            task="chat",
            family="",
            parameter_size="",
            provider="LM Studio",
        )
        ref = format_remote_ref(model.name, model.provider)
        assert ref == "lm_studio/qwen2.5-7b-instruct"
        # Round-trips back through the parser without losing the prefix.
        assert parse_model_ref(ref).provider == "lm_studio"

    def test_lm_studio_routing_key_form(self) -> None:
        assert format_remote_ref("some-model", "lm_studio") == "lm_studio/some-model"

    def test_openrouter_with_nested_path(self) -> None:
        """OpenRouter model ids carry a vendor/model path that must round-trip."""
        ref = format_remote_ref("anthropic/claude-3.5-sonnet", "OpenRouter")
        assert ref == "openrouter/anthropic/claude-3.5-sonnet"
        # Round-trip back through the parser without double-prefixing.
        parsed = parse_model_ref(ref)
        assert parsed.provider == "openrouter"
        assert parsed.for_openai_prefix() == ref

    def test_mistral_provider(self) -> None:
        ref = format_remote_ref("codestral-latest", "Mistral")
        assert ref == "mistral/codestral-latest"

    def test_deepseek_provider(self) -> None:
        ref = format_remote_ref("deepseek-chat", "DeepSeek")
        assert ref == "deepseek/deepseek-chat"


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


# Options a chat caller can supply; both backends must agree on num_predict and
# never emit a key the receiving SDK errors on.
_SHARED_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "seed": 123,
    "num_predict": 1024,
    "repeat_penalty": 1.1,
    "num_ctx": 4096,
}

# The local chat-option translation emits these keys and nothing else from the
# option set; the llama-server request builder consumes exactly this set, so any
# extra key (num_ctx, num_predict) is a translation bug.
_LOCAL_CHAT_OPTION_KEYS = frozenset(
    {"temperature", "top_p", "top_k", "seed", "max_tokens", "repeat_penalty"}
)


class TestChatOptionTranslationParity:
    """Differential gate: same options through local-engine vs API translation.

    Pins the intentional divergence so the two paths can't silently drift:
    the local llama-server path keeps top_k/repeat_penalty and renames
    num_predict->max_tokens; the API path additionally strips top_k/num_ctx.
    Both rename num_predict consistently and neither leaks a key its backend
    would reject.
    """

    def _local(self) -> dict[str, object]:
        from lilbee.providers.engine_params import chat_options_to_kwargs

        return chat_options_to_kwargs(dict(_SHARED_OPTIONS))

    def _api(self) -> dict[str, object]:
        ref = parse_model_ref("openai/gpt-4o")
        return translate_options(dict(_SHARED_OPTIONS), ref)

    def test_num_predict_renamed_consistently(self) -> None:
        """Both backends speak max_tokens, not num_predict."""
        for translated in (self._local(), self._api()):
            assert translated["max_tokens"] == 1024
            assert "num_predict" not in translated

    def test_local_translation_keeps_local_only_params(self) -> None:
        """The local engine honors top_k and repeat_penalty, so they survive."""
        translated = self._local()
        assert translated["top_k"] == 40
        assert translated["repeat_penalty"] == 1.1

    def test_api_drops_top_k(self) -> None:
        """Hosted providers ignore top_k, so the API path strips it."""
        assert "top_k" not in self._api()

    def test_neither_path_leaks_num_ctx(self) -> None:
        """num_ctx is a model-load param; it must never reach a per-call request."""
        assert "num_ctx" not in self._local()
        assert "num_ctx" not in self._api()

    def test_local_translation_emits_only_supported_keys(self) -> None:
        """Every emitted key is one the llama-server request builder accepts."""
        translated = self._local()
        assert set(translated) <= _LOCAL_CHAT_OPTION_KEYS

    def test_api_translation_does_not_error_in_litellm(self) -> None:
        """litellm accepts the API-translated kwargs without raising.

        Confirms the audit finding empirically: litellm 1.x forwards
        repeat_penalty (into extra_body for OpenAI-compatible providers)
        rather than rejecting it, so keeping it is not a correctness bug.
        """
        litellm = pytest.importorskip("litellm")
        translated = self._api()
        params = litellm.utils.get_optional_params(
            model="gpt-4o", custom_llm_provider="openai", **translated
        )
        assert params["max_tokens"] == 1024
        # repeat_penalty is forwarded, not dropped or errored on.
        assert params.get("extra_body", {}).get("repeat_penalty") == 1.1
