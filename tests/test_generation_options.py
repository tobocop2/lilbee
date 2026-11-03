"""Tests for the generation options merge: user config, then per-call overrides."""

from __future__ import annotations

import pytest

from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def _unset_generation_fields():
    """Unset every generation field so the user's local config.toml cannot leak in."""
    snapshot = cfg.model_copy()
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


class TestGenerationOptionsMerge:
    def test_nothing_set_gives_empty_dict(self):
        assert cfg.generation_options() == {}

    def test_user_config_is_used(self):
        cfg.temperature = 0.3
        cfg.num_ctx = 4096
        assert cfg.generation_options() == {"temperature": 0.3, "num_ctx": 4096}

    def test_per_call_overrides_win(self):
        cfg.temperature = 0.3
        assert cfg.generation_options(temperature=1.0)["temperature"] == 1.0

    def test_none_override_keeps_user_config(self):
        cfg.temperature = 0.3
        assert cfg.generation_options(temperature=None) == {"temperature": 0.3}

    def test_top_k_sampling_remapped(self):
        """Config's top_k_sampling maps to 'top_k' in output."""
        cfg.top_k_sampling = 40
        result = cfg.generation_options()
        assert result["top_k"] == 40
        assert "top_k_sampling" not in result

    def test_seed_passthrough(self):
        cfg.seed = 42
        assert cfg.generation_options()["seed"] == 42

    def test_max_tokens_becomes_num_predict(self):
        cfg.max_tokens = 2048
        assert cfg.generation_options() == {"num_predict": 2048}
