"""Tests for per-model default generation settings."""

from __future__ import annotations

import pytest

from lilbee.config import cfg
from lilbee.model_defaults import (
    ModelDefaults,
    clear_cache,
    get_defaults,
    parse_ollama_parameters,
    read_gguf_defaults,
    set_defaults,
)


@pytest.fixture(autouse=True)
def _isolated_defaults(tmp_path):
    """Snapshot config and clear model defaults cache for each test."""
    snapshot = cfg.model_copy()
    cfg.apply_model_defaults(None)
    clear_cache()
    yield
    clear_cache()
    for field_name in type(cfg).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))
    cfg.clear_model_defaults()


class TestParseOllamaParameters:
    def test_empty_string(self):
        result = parse_ollama_parameters("")
        assert result == ModelDefaults()

    def test_valid_params(self):
        text = "temperature 0.7\ntop_p 0.9\nnum_ctx 4096\nrepeat_penalty 1.1"
        result = parse_ollama_parameters(text)
        assert result.temperature == 0.7
        assert result.top_p == 0.9
        assert result.num_ctx == 4096
        assert result.repeat_penalty == 1.1

    def test_unknown_keys_skipped(self):
        text = "temperature 0.7\nstop <|im_end|>\nmirostat 2\ntop_p 0.8"
        result = parse_ollama_parameters(text)
        assert result.temperature == 0.7
        assert result.top_p == 0.8
        assert result.num_ctx is None

    def test_invalid_values_skipped(self):
        text = "temperature abc\ntop_p 0.9"
        result = parse_ollama_parameters(text)
        assert result.temperature is None
        assert result.top_p == 0.9

    def test_whitespace_handling(self):
        text = "  temperature   0.5  \n\n  num_ctx   2048  \n"
        result = parse_ollama_parameters(text)
        assert result.temperature == 0.5
        assert result.num_ctx == 2048

    def test_single_word_lines_skipped(self):
        text = "temperature\ntop_p 0.9"
        result = parse_ollama_parameters(text)
        assert result.temperature is None
        assert result.top_p == 0.9

    def test_top_k_parsed_as_int(self):
        result = parse_ollama_parameters("top_k 40")
        assert result.top_k == 40
        assert isinstance(result.top_k, int)

    def test_max_tokens_parsed(self):
        result = parse_ollama_parameters("max_tokens 2048")
        assert result.max_tokens == 2048


class TestReadGgufDefaults:
    def test_empty_dict(self):
        result = read_gguf_defaults({})
        assert result == ModelDefaults()

    def test_valid_metadata(self):
        metadata = {
            "general.temperature": "0.8",
            "general.top_p": "0.95",
            "context_length": "8192",
        }
        result = read_gguf_defaults(metadata)
        assert result.temperature == 0.8
        assert result.top_p == 0.95
        assert result.num_ctx == 8192

    def test_missing_keys(self):
        metadata = {"general.architecture": "llama", "general.name": "test"}
        result = read_gguf_defaults(metadata)
        assert result == ModelDefaults()

    def test_repeat_penalty(self):
        metadata = {"general.repeat_penalty": "1.1"}
        result = read_gguf_defaults(metadata)
        assert result.repeat_penalty == 1.1

    def test_top_k(self):
        metadata = {"general.top_k": "40"}
        result = read_gguf_defaults(metadata)
        assert result.top_k == 40

    def test_invalid_value_skipped(self):
        metadata = {"general.temperature": "not_a_number", "context_length": "4096"}
        result = read_gguf_defaults(metadata)
        assert result.temperature is None
        assert result.num_ctx == 4096

    def test_invalid_context_length_skipped(self):
        metadata = {"context_length": "abc"}
        result = read_gguf_defaults(metadata)
        assert result.num_ctx is None


class TestCache:
    def test_get_returns_none_when_empty(self):
        assert get_defaults("nonexistent") is None

    def test_set_and_get(self):
        defaults = ModelDefaults(temperature=0.5)
        set_defaults("test-model", defaults)
        assert get_defaults("test-model") is defaults

    def test_clear_removes_all(self):
        set_defaults("a", ModelDefaults(temperature=0.1))
        set_defaults("b", ModelDefaults(temperature=0.2))
        clear_cache()
        assert get_defaults("a") is None
        assert get_defaults("b") is None

    def test_overwrite(self):
        set_defaults("m", ModelDefaults(temperature=0.1))
        set_defaults("m", ModelDefaults(temperature=0.9))
        assert get_defaults("m") == ModelDefaults(temperature=0.9)


class TestGenerationOptions3LayerMerge:
    def test_no_defaults_no_user_config(self):
        """All None -> empty dict (excluding max_tokens which has a default)."""
        cfg.max_tokens = None
        result = cfg.generation_options()
        assert result == {}

    def test_model_defaults_only(self):
        cfg.max_tokens = None
        cfg.apply_model_defaults(ModelDefaults(temperature=0.7, num_ctx=4096))
        result = cfg.generation_options()
        assert result == {"temperature": 0.7, "num_ctx": 4096}

    def test_user_config_overrides_model_defaults(self):
        cfg.apply_model_defaults(ModelDefaults(temperature=0.7, num_ctx=4096))
        cfg.temperature = 0.3
        cfg.max_tokens = None
        result = cfg.generation_options()
        assert result["temperature"] == 0.3
        assert result["num_ctx"] == 4096

    def test_per_call_overrides_win(self):
        cfg.apply_model_defaults(ModelDefaults(temperature=0.7))
        cfg.temperature = 0.3
        cfg.max_tokens = None
        result = cfg.generation_options(temperature=1.0)
        assert result["temperature"] == 1.0

    def test_none_fields_skipped_at_all_layers(self):
        cfg.apply_model_defaults(ModelDefaults(temperature=0.5))
        cfg.max_tokens = None
        result = cfg.generation_options()
        assert "top_p" not in result
        assert "num_ctx" not in result
        assert result == {"temperature": 0.5}

    def test_model_defaults_fill_gaps(self):
        """Model defaults provide values where user hasn't set any."""
        cfg.apply_model_defaults(ModelDefaults(temperature=0.7, top_p=0.9, num_ctx=8192))
        cfg.temperature = 0.5
        cfg.max_tokens = None
        result = cfg.generation_options()
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert result["num_ctx"] == 8192

    def test_clear_model_defaults(self):
        cfg.apply_model_defaults(ModelDefaults(temperature=0.7))
        cfg.clear_model_defaults()
        cfg.max_tokens = None
        result = cfg.generation_options()
        assert result == {}

    def test_top_k_sampling_remapped(self):
        """Config's top_k_sampling maps to 'top_k' in output."""
        cfg.top_k_sampling = 40
        cfg.max_tokens = None
        result = cfg.generation_options()
        assert result["top_k"] == 40
        assert "top_k_sampling" not in result

    def test_model_default_top_k_used(self):
        cfg.apply_model_defaults(ModelDefaults(top_k=50))
        cfg.max_tokens = None
        result = cfg.generation_options()
        assert result["top_k"] == 50

    def test_seed_passthrough(self):
        cfg.seed = 42
        cfg.max_tokens = None
        result = cfg.generation_options()
        assert result["seed"] == 42

    def test_max_tokens_from_config(self):
        cfg.max_tokens = 2048
        result = cfg.generation_options()
        assert result["max_tokens"] == 2048

    def test_max_tokens_from_model_defaults(self):
        cfg.max_tokens = None
        cfg.apply_model_defaults(ModelDefaults(max_tokens=1024))
        result = cfg.generation_options()
        assert result["max_tokens"] == 1024

    def test_all_three_layers(self):
        """Full 3-layer merge: model default -> user config -> per-call."""
        cfg.apply_model_defaults(
            ModelDefaults(temperature=0.7, top_p=0.9, num_ctx=4096, top_k=40)
        )
        cfg.temperature = 0.5
        cfg.max_tokens = None
        result = cfg.generation_options(num_ctx=8192)
        assert result["temperature"] == 0.5  # user config wins over model
        assert result["top_p"] == 0.9  # model default fills gap
        assert result["num_ctx"] == 8192  # per-call wins over model
        assert result["top_k"] == 40  # model default fills gap


class TestLiteLLMCacheIntegration:
    """Test that _cache_ollama_defaults correctly parses and stores."""

    def test_cache_ollama_defaults(self):
        from lilbee.providers.litellm_provider import _cache_ollama_defaults

        _cache_ollama_defaults("qwen3:8b", "temperature 0.7\nnum_ctx 32768\nstop <|im_end|>")
        defaults = get_defaults("qwen3:8b")
        assert defaults is not None
        assert defaults.temperature == 0.7
        assert defaults.num_ctx == 32768

    def test_cache_ollama_defaults_empty(self):
        from lilbee.providers.litellm_provider import _cache_ollama_defaults

        _cache_ollama_defaults("empty-model", "")
        defaults = get_defaults("empty-model")
        assert defaults is not None
        assert defaults == ModelDefaults()
