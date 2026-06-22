"""Tests for per-model default generation settings and the 3-layer merge."""

from __future__ import annotations

import pytest

from lilbee.core.config import cfg
from lilbee.providers.model_defaults import ModelDefaults


@pytest.fixture(autouse=True)
def _isolated_defaults():
    """Snapshot config and clear model defaults for each test.

    Resets all generation-option fields to None so tests aren't affected
    by the user's local config.toml (e.g. temperature=0.6).
    """
    snapshot = cfg.model_copy()
    cfg.apply_model_defaults(None)
    cfg.temperature = None
    cfg.top_p = None
    cfg.top_k_sampling = None
    cfg.repeat_penalty = None
    cfg.num_ctx = None
    cfg.seed = None
    cfg.max_tokens = None
    yield
    for field_name in type(cfg).model_fields:
        setattr(cfg, field_name, getattr(snapshot, field_name))
    cfg.clear_model_defaults()


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
        """Model defaults provide values where the user hasn't set any."""
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
        cfg.apply_model_defaults(ModelDefaults(temperature=0.7, top_p=0.9, num_ctx=4096, top_k=40))
        cfg.temperature = 0.5
        cfg.max_tokens = None
        result = cfg.generation_options(num_ctx=8192)
        assert result["temperature"] == 0.5  # user config wins over model
        assert result["top_p"] == 0.9  # model default fills gap
        assert result["num_ctx"] == 8192  # per-call wins over model
        assert result["top_k"] == 40  # model default fills gap
