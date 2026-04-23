"""Tests for Config (pydantic-settings BaseSettings) and env var overrides."""

import os
import re
from pathlib import Path
from unittest import mock

import pytest

from lilbee.config import (
    _DEFAULT_CORS_ORIGIN_REGEX,
    CHUNKS_TABLE,
    DEFAULT_IGNORE_DIRS,
    SOURCES_TABLE,
    Config,
    cfg,
)


def _clean_env(tmp_path: Path | None = None) -> dict[str, str]:
    """Return os.environ with all LILBEE_* and OLLAMA_HOST vars removed.

    If tmp_path is given, sets LILBEE_DATA to it so no existing config.toml
    is accidentally picked up. Sets ``LILBEE_SKIP_MODEL_TASK_VALIDATION=1``
    so tests using placeholder model names don't trip the per-role
    catalog-task validator; pop it explicitly to exercise that validator.
    """
    env = {
        k: v for k, v in os.environ.items() if not k.startswith("LILBEE_") and k != "OLLAMA_HOST"
    }
    env["LILBEE_SKIP_MODEL_TASK_VALIDATION"] = "1"
    if tmp_path is not None:
        env["LILBEE_DATA"] = str(tmp_path)
    return env


class TestFromEnvDefaults:
    def test_default_values(self, tmp_path):
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.chat_model == "qwen3:0.6b"
            assert c.embedding_model == "nomic-embed-text:v1.5"
            assert c.embedding_dim == 768
            assert c.chunk_size == 512
            assert c.chunk_overlap == 100
            assert c.max_embed_chars == 2000
            assert c.top_k == 10
            assert c.max_distance == 0.9
            assert c.json_mode is False

    def test_constants_unchanged(self):
        assert CHUNKS_TABLE == "chunks"
        assert SOURCES_TABLE == "_sources"
        assert "node_modules" in DEFAULT_IGNORE_DIRS

    def test_config_field_public_false_marker(self):
        """ConfigField(public=False) stores the flag in json_schema_extra."""
        from lilbee.config import ConfigField

        info = ConfigField(default="", writable=True, public=False)
        extra = info.json_schema_extra
        assert isinstance(extra, dict)
        assert extra.get("public") is False


class TestEnvVarOverrides:
    def test_lilbee_data_overrides_paths(self, tmp_path):
        with mock.patch.dict(os.environ, {"LILBEE_DATA": str(tmp_path)}):
            c = Config()
            assert c.data_root == tmp_path
            assert c.documents_dir == tmp_path / "documents"
            assert c.data_dir == tmp_path / "data"
            assert c.lancedb_dir == tmp_path / "data" / "lancedb"

    def test_data_root_default_uses_platform(self):
        env = _clean_env()
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert str(c.data_root).endswith("lilbee")

    def test_chat_model_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHAT_MODEL": "llama3"}):
            c = Config()
            assert c.chat_model == "llama3:latest"

    def test_chat_model_override_tagged(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHAT_MODEL": "llama3:8b"}):
            c = Config()
            assert c.chat_model == "llama3:8b"

    def test_embedding_model_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_EMBEDDING_MODEL": "mxbai-embed-large"}):
            c = Config()
            assert c.embedding_model == "mxbai-embed-large:latest"

    def test_model_tag_normalized_on_assignment(self):
        cfg.chat_model = "qwen3"
        assert cfg.chat_model == "qwen3:latest"
        cfg.chat_model = "qwen3:0.6b"
        assert cfg.chat_model == "qwen3:0.6b"

    def test_normalize_model_tag_empty_string_passthrough(self):
        """The validator's empty-string guard returns immediately."""
        cfg.vision_model = ""
        assert cfg.vision_model == ""

    def test_embedding_dim_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_EMBEDDING_DIM": "1024"}):
            c = Config()
            assert c.embedding_dim == 1024

    def test_chunk_size_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHUNK_SIZE": "256"}):
            c = Config()
            assert c.chunk_size == 256

    def test_chunk_size_below_minimum_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cfg.chunk_size = 5

    def test_chunk_overlap_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHUNK_OVERLAP": "50"}):
            c = Config()
            assert c.chunk_overlap == 50

    def test_top_k_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_TOP_K": "20"}):
            c = Config()
            assert c.top_k == 20

    def test_max_embed_chars_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_MAX_EMBED_CHARS": "3000"}):
            c = Config()
            assert c.max_embed_chars == 3000

    def test_max_distance_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_MAX_DISTANCE": "1.5"}):
            c = Config()
            assert c.max_distance == 1.5

    def test_system_prompt_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_SYSTEM_PROMPT": "You are a pirate."}):
            c = Config()
            assert c.system_prompt == "You are a pirate."


class TestTomlConfigFile:
    def test_toml_values_loaded(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('chat_model = "my-saved-model"\n')
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.chat_model == "my-saved-model:latest"

    def test_env_var_overrides_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('chat_model = "toml-model"\n')
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "env-model"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.chat_model == "env-model:latest"

    def test_no_toml_uses_defaults(self, tmp_path):
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.chat_model == "qwen3:0.6b"

    def test_corrupt_toml_uses_defaults(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("this is not valid TOML [[[")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.chat_model == "qwen3:0.6b"

    def test_deprecated_config_keys_log_warning_on_load(self, tmp_path, caplog):
        """Phase D dropped four wiki config keys; each should log a
        warning when present in config.toml, and neither survive as
        a Config attribute.
        """
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            "wiki_faithfulness_threshold = 0.7\n"
            'wiki_faithfulness_prompt = "old"\n'
            "wiki_faithfulness_max_tokens = 256\n"
            'wiki_concept_prompt = "old"\n'
        )
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with (
            mock.patch.dict(os.environ, env, clear=True),
            caplog.at_level("WARNING", logger="lilbee.config"),
        ):
            c = Config()
        assert not hasattr(c, "wiki_faithfulness_threshold")
        assert not hasattr(c, "wiki_faithfulness_prompt")
        assert not hasattr(c, "wiki_faithfulness_max_tokens")
        assert not hasattr(c, "wiki_concept_prompt")
        warning_text = " ".join(r.message for r in caplog.records)
        for key in (
            "wiki_faithfulness_threshold",
            "wiki_faithfulness_prompt",
            "wiki_faithfulness_max_tokens",
            "wiki_concept_prompt",
        ):
            assert key in warning_text

    def test_embedding_model_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('embedding_model = "my-embed"\n')
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.embedding_model == "my-embed:latest"

    def test_temperature_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("temperature = 0.5\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.temperature == 0.5

    def test_env_var_overrides_toml_for_temperature(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("temperature = 0.5\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_TEMPERATURE"] = "0.9"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.temperature == 0.9

    def test_system_prompt_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('system_prompt = "You are a pirate."\n')
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.system_prompt == "You are a pirate."

    def test_env_var_overrides_toml_for_system_prompt(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('system_prompt = "Be verbose."\n')
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_SYSTEM_PROMPT"] = "Be brief."
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.system_prompt == "Be brief."

    def test_enable_ocr_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("enable_ocr = true\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.enable_ocr is True

    def test_top_p_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("top_p = 0.9\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.top_p == 0.9

    def test_top_k_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("top_k = 20\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.top_k == 20

    def test_top_k_sampling_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("top_k_sampling = 40\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.top_k_sampling == 40

    def test_repeat_penalty_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("repeat_penalty = 1.2\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.repeat_penalty == 1.2

    def test_repeat_penalty_defaults_to_one_point_one(self, tmp_path):
        """Fresh Config defaults repeat_penalty to 1.1 so chat doesn't loop."""
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.repeat_penalty == 1.1

    def test_num_ctx_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("num_ctx = 4096\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.num_ctx == 4096

    def test_seed_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("seed = 123\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.seed == 123


class TestEnableOcrConfig:
    def test_default_is_none(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.enable_ocr is None

    def test_true_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_ENABLE_OCR": "true"}):
            c = Config()
            assert c.enable_ocr is True

    def test_false_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_ENABLE_OCR": "false"}):
            c = Config()
            assert c.enable_ocr is False

    def test_empty_string_means_auto(self, tmp_path) -> None:
        with mock.patch.dict(
            os.environ, {**_clean_env(tmp_path), "LILBEE_ENABLE_OCR": ""}, clear=True
        ):
            c = Config()
            assert c.enable_ocr is None

    def test_auto_string_means_none(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_ENABLE_OCR": "auto"}):
            c = Config()
            assert c.enable_ocr is None

    def test_yes_no_variants(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_ENABLE_OCR": "yes"}):
            c = Config()
            assert c.enable_ocr is True

        with mock.patch.dict(os.environ, {"LILBEE_ENABLE_OCR": "no"}):
            c = Config()
            assert c.enable_ocr is False

    def test_numeric_variants(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_ENABLE_OCR": "1"}):
            c = Config()
            assert c.enable_ocr is True

        with mock.patch.dict(os.environ, {"LILBEE_ENABLE_OCR": "0"}):
            c = Config()
            assert c.enable_ocr is False

    def test_case_insensitive(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_ENABLE_OCR": "TRUE"}):
            c = Config()
            assert c.enable_ocr is True

    def test_from_toml(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("enable_ocr = true\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.enable_ocr is True

    def test_garbage_value_coerces_via_bool(self) -> None:
        """Unrecognized string falls through _parse_bool and coerces via ``bool()``."""
        with mock.patch.dict(os.environ, {"LILBEE_ENABLE_OCR": "maybe"}):
            c = Config()
            assert c.enable_ocr is True  # bool("maybe") is True

    def test_whitespace_only_means_auto(self) -> None:
        """Whitespace-only strings hit the auto/none branch and return None."""
        with mock.patch.dict(os.environ, {"LILBEE_ENABLE_OCR": "   "}):
            c = Config()
            assert c.enable_ocr is None


class TestSemanticChunkingConfig:
    def test_default_is_false(self, tmp_path) -> None:
        """Semantic chunking is opt-in: default False, enabled via env/config."""
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.semantic_chunking is False

    def test_true_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_SEMANTIC_CHUNKING": "true"}):
            c = Config()
            assert c.semantic_chunking is True

    def test_false_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_SEMANTIC_CHUNKING": "false"}):
            c = Config()
            assert c.semantic_chunking is False

    def test_yes_no_variants(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_SEMANTIC_CHUNKING": "yes"}):
            assert Config().semantic_chunking is True
        with mock.patch.dict(os.environ, {"LILBEE_SEMANTIC_CHUNKING": "no"}):
            assert Config().semantic_chunking is False

    def test_numeric_variants(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_SEMANTIC_CHUNKING": "1"}):
            assert Config().semantic_chunking is True
        with mock.patch.dict(os.environ, {"LILBEE_SEMANTIC_CHUNKING": "0"}):
            assert Config().semantic_chunking is False

    def test_case_insensitive(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_SEMANTIC_CHUNKING": "FALSE"}):
            assert Config().semantic_chunking is False

    def test_invalid_falls_back_to_default(self, caplog) -> None:
        import logging

        with (
            mock.patch.dict(os.environ, {"LILBEE_SEMANTIC_CHUNKING": "banana"}),
            caplog.at_level(logging.WARNING, logger="lilbee.config"),
        ):
            c = Config()
            assert c.semantic_chunking is False
        assert any("banana" in rec.message for rec in caplog.records)

    def test_non_string_non_bool_coerced(self) -> None:
        """Validator coerces non-str, non-bool inputs via ``bool()``.

        Calls the validator directly because pydantic may pre-coerce via
        its own conversion before a mode="before" validator even sees
        simple types like int.
        """
        from lilbee.config import Config

        parse = Config._parse_semantic_chunking
        assert parse(1) is True
        assert parse(0) is False
        assert parse([1]) is True
        assert parse([]) is False

    def test_from_toml(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("semantic_chunking = false\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            assert Config().semantic_chunking is False

    def test_env_overrides_toml(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("semantic_chunking = false\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_SEMANTIC_CHUNKING"] = "true"
        with mock.patch.dict(os.environ, env, clear=True):
            assert Config().semantic_chunking is True


class TestTopicThresholdConfig:
    def test_default_is_0_75(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.topic_threshold == pytest.approx(0.75)

    def test_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_TOPIC_THRESHOLD": "0.5"}):
            assert Config().topic_threshold == pytest.approx(0.5)

    def test_accepts_boundaries(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_TOPIC_THRESHOLD": "0.0"}):
            assert Config().topic_threshold == 0.0
        with mock.patch.dict(os.environ, {"LILBEE_TOPIC_THRESHOLD": "1.0"}):
            assert Config().topic_threshold == 1.0

    def test_out_of_range_raises(self) -> None:
        from pydantic import ValidationError

        with (
            mock.patch.dict(os.environ, {"LILBEE_TOPIC_THRESHOLD": "1.5"}),
            pytest.raises(ValidationError),
        ):
            Config()

    def test_from_toml(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("topic_threshold = 0.42\n")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            assert Config().topic_threshold == pytest.approx(0.42)


class TestParseBool:
    def test_truthy_values(self) -> None:
        from lilbee.config import _parse_bool

        for truthy in ("true", "TRUE", "1", "yes", "  YES  "):
            assert _parse_bool(truthy) is True

    def test_falsy_values(self) -> None:
        from lilbee.config import _parse_bool

        for falsy in ("false", "FALSE", "0", "no", "  NO  "):
            assert _parse_bool(falsy) is False

    def test_invalid_raises(self) -> None:
        from lilbee.config import _parse_bool

        with pytest.raises(ValueError, match="Invalid boolean"):
            _parse_bool("maybe")


class TestOcrTimeoutConfig:
    def test_default_is_120(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.ocr_timeout == 120.0

    def test_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_OCR_TIMEOUT": "60.5"}):
            c = Config()
            assert c.ocr_timeout == 60.5

    def test_zero_means_no_limit(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_OCR_TIMEOUT": "0"}):
            c = Config()
            assert c.ocr_timeout == 0

    def test_invalid_raises(self) -> None:
        with (
            mock.patch.dict(os.environ, {"LILBEE_OCR_TIMEOUT": "abc"}),
            pytest.raises(ValueError),
        ):
            Config()


class TestCorsOriginsConfig:
    def test_cors_origins_from_env(self) -> None:
        with mock.patch.dict(
            os.environ, {"LILBEE_CORS_ORIGINS": "app://obsidian.md,https://my-app.com"}
        ):
            c = Config()
            assert c.cors_origins == ["app://obsidian.md", "https://my-app.com"]

    def test_cors_origins_default_empty(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.cors_origins == []

    def test_cors_origins_list_passthrough(self) -> None:
        """List values pass through the validator unchanged."""
        cfg.cors_origins = ["https://a.com", "https://b.com"]
        assert cfg.cors_origins == ["https://a.com", "https://b.com"]


class TestCorsOriginRegexConfig:
    def test_cors_origin_regex_default_matches_obsidian_desktop(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert pat.fullmatch("app://obsidian.md")

    def test_cors_origin_regex_default_matches_capacitor_localhost(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert pat.fullmatch("capacitor://localhost")

    def test_cors_origin_regex_default_matches_http_localhost_any_port(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert pat.fullmatch("http://localhost")
            assert pat.fullmatch("http://localhost:3000")
            assert pat.fullmatch("http://localhost:7433")
            assert pat.fullmatch("https://localhost:8443")

    def test_cors_origin_regex_default_matches_loopback_ipv4(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert pat.fullmatch("http://127.0.0.1:7433")
            assert pat.fullmatch("https://127.0.0.1")

    def test_cors_origin_regex_default_matches_loopback_ipv6(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert pat.fullmatch("http://[::1]:7433")
            assert pat.fullmatch("https://[::1]")

    def test_cors_origin_regex_default_rejects_random_remote(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert not pat.fullmatch("https://evil.example.com")
            assert not pat.fullmatch("http://not-localhost.example")
            assert not pat.fullmatch("app://some-other-app.md")

    def test_cors_origin_regex_from_env_overrides_default(self, tmp_path) -> None:
        env = _clean_env(tmp_path)
        env["LILBEE_CORS_ORIGIN_REGEX"] = r"^https://only-this\.example$"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.cors_origin_regex == r"^https://only-this\.example$"

    def test_cors_origin_regex_from_env_match_nothing_disables_default(self, tmp_path) -> None:
        # Empty env vars are ignored by _PlainEnvSource, so the documented opt-out is
        # to set a regex that matches nothing — e.g. ^$.
        env = _clean_env(tmp_path)
        env["LILBEE_CORS_ORIGIN_REGEX"] = "^$"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.cors_origin_regex == "^$"

    def test_cors_origin_regex_default_compiles(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            re.compile(c.cors_origin_regex)

    def test_cors_origin_regex_default_equals_constant(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.cors_origin_regex == _DEFAULT_CORS_ORIGIN_REGEX


class TestLocalDotLilbee:
    def test_local_lilbee_overrides_default(self, tmp_path):
        local = tmp_path / ".lilbee"
        local.mkdir()
        env = _clean_env()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.platform.find_local_root", return_value=local),
        ):
            c = Config()
            assert c.data_root == local
            assert c.documents_dir == local / "documents"
            assert c.lancedb_dir == local / "data" / "lancedb"

    def test_lilbee_data_takes_precedence_over_local(self, tmp_path):
        local = tmp_path / ".lilbee"
        local.mkdir()
        explicit = tmp_path / "explicit"
        with (
            mock.patch.dict(os.environ, {"LILBEE_DATA": str(explicit)}),
            mock.patch("lilbee.platform.find_local_root", return_value=local),
        ):
            c = Config()
            assert c.data_root == explicit

    def test_no_local_uses_platform_default(self):
        env = _clean_env()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.platform.find_local_root", return_value=None),
        ):
            c = Config()
            assert c.data_root.name == "lilbee"
            assert c.data_root.name != ".lilbee"


class TestGenerationOptions:
    def test_empty_when_all_none(self):
        c = Config()
        c.temperature = None
        c.top_p = None
        c.top_k_sampling = None
        c.repeat_penalty = None
        c.num_ctx = None
        c.seed = None
        c.max_tokens = None
        assert c.generation_options() == {}

    def test_includes_set_values(self):
        c = Config()
        c.temperature = 0.3
        c.seed = 42
        c.top_p = None
        c.top_k_sampling = None
        c.repeat_penalty = None
        c.num_ctx = None
        c.max_tokens = None
        opts = c.generation_options()
        assert opts == {"temperature": 0.3, "seed": 42}

    def test_includes_max_tokens(self):
        c = Config()
        opts = c.generation_options()
        assert opts["max_tokens"] == 4096

    def test_remaps_top_k_sampling(self):
        c = Config()
        c.temperature = None
        c.top_p = None
        c.top_k_sampling = 40
        c.repeat_penalty = None
        c.num_ctx = None
        c.seed = None
        c.max_tokens = None
        opts = c.generation_options()
        assert opts == {"top_k": 40}
        assert "top_k_sampling" not in opts

    def test_overrides_merge(self):
        c = Config()
        c.temperature = 0.5
        c.top_p = None
        c.top_k_sampling = None
        c.repeat_penalty = None
        c.num_ctx = None
        c.seed = None
        c.max_tokens = None
        opts = c.generation_options(temperature=0.9, num_ctx=4096)
        assert opts == {"temperature": 0.9, "num_ctx": 4096}

    def test_env_var_wiring(self):
        with mock.patch.dict(
            os.environ,
            {
                "LILBEE_TEMPERATURE": "0.3",
                "LILBEE_TOP_P": "0.95",
                "LILBEE_TOP_K_SAMPLING": "40",
                "LILBEE_REPEAT_PENALTY": "1.1",
                "LILBEE_NUM_CTX": "4096",
                "LILBEE_SEED": "123",
            },
        ):
            c = Config()
            assert c.temperature == 0.3
            assert c.top_p == 0.95
            assert c.top_k_sampling == 40
            assert c.repeat_penalty == 1.1
            assert c.num_ctx == 4096
            assert c.seed == 123


class TestIgnoreDirs:
    def test_default_ignore_dirs_contains_expected(self):
        c = Config()
        for name in ["node_modules", "__pycache__", "venv", "build", "dist"]:
            assert name in c.ignore_dirs

    def test_lilbee_ignore_dirs_env_adds_custom_entries(self):
        with mock.patch.dict(os.environ, {"LILBEE_IGNORE_DIRS": "output,generated"}):
            c = Config()
            assert "output" in c.ignore_dirs
            assert "generated" in c.ignore_dirs
            assert "node_modules" in c.ignore_dirs

    def test_lilbee_ignore_dirs_empty_string(self):
        env = _clean_env()
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.ignore_dirs == DEFAULT_IGNORE_DIRS

    def test_lilbee_ignore_dirs_strips_whitespace(self):
        with mock.patch.dict(os.environ, {"LILBEE_IGNORE_DIRS": " foo , bar "}):
            c = Config()
            assert "foo" in c.ignore_dirs
            assert "bar" in c.ignore_dirs


class TestConceptAllowedEntTypes:
    """A3 entity-type filter: spaCy NER labels kept by the wiki extractor."""

    def test_default_includes_core_wiki_types(self):
        c = Config()
        for label in ("PERSON", "ORG", "GPE", "PRODUCT", "FAC", "NORP"):
            assert label in c.concept_allowed_ent_types

    def test_default_excludes_quantitative_types(self):
        c = Config()
        for label in ("QUANTITY", "CARDINAL", "DATE", "TIME", "MONEY", "PERCENT"):
            assert label not in c.concept_allowed_ent_types

    def test_env_override_replaces_defaults(self):
        # Replace-semantics: narrowing the set should NOT re-union with
        # the defaults the way ``ignore_dirs`` does.
        with mock.patch.dict(os.environ, {"LILBEE_CONCEPT_ALLOWED_ENT_TYPES": "PERSON,ORG"}):
            c = Config()
            assert c.concept_allowed_ent_types == frozenset({"PERSON", "ORG"})

    def test_env_override_is_case_insensitive(self):
        with mock.patch.dict(os.environ, {"LILBEE_CONCEPT_ALLOWED_ENT_TYPES": "person,Org"}):
            c = Config()
            assert c.concept_allowed_ent_types == frozenset({"PERSON", "ORG"})

    def test_empty_env_falls_back_to_default(self):
        # Empty override should not silently deactivate the gate.
        env = _clean_env()
        env["LILBEE_CONCEPT_ALLOWED_ENT_TYPES"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert "PERSON" in c.concept_allowed_ent_types


class TestEmptyStringValidation:
    def test_empty_chat_model_rejected(self, tmp_path):
        with pytest.raises(Exception, match="at least 1 character"):
            Config(
                data_root=tmp_path,
                documents_dir=tmp_path / "docs",
                data_dir=tmp_path / "data",
                lancedb_dir=tmp_path / "data" / "lancedb",
                models_dir=tmp_path / "models",
                chat_model="",
                embedding_model="nomic-embed-text",
                embedding_dim=768,
                chunk_size=512,
                chunk_overlap=100,
                max_embed_chars=2000,
                top_k=10,
                max_distance=0.7,
                system_prompt="You are helpful.",
                ignore_dirs=frozenset(),
            )

    def test_empty_embedding_model_rejected(self, tmp_path):
        with pytest.raises(Exception, match="at least 1 character"):
            Config(
                data_root=tmp_path,
                documents_dir=tmp_path / "docs",
                data_dir=tmp_path / "data",
                lancedb_dir=tmp_path / "data" / "lancedb",
                models_dir=tmp_path / "models",
                chat_model="qwen3",
                embedding_model="",
                embedding_dim=768,
                chunk_size=512,
                chunk_overlap=100,
                max_embed_chars=2000,
                top_k=10,
                max_distance=0.7,
                system_prompt="You are helpful.",
                ignore_dirs=frozenset(),
            )

    def test_empty_system_prompt_rejected(self, tmp_path):
        with pytest.raises(Exception, match="at least 1 character"):
            Config(
                data_root=tmp_path,
                documents_dir=tmp_path / "docs",
                data_dir=tmp_path / "data",
                lancedb_dir=tmp_path / "data" / "lancedb",
                models_dir=tmp_path / "models",
                chat_model="qwen3",
                embedding_model="nomic-embed-text",
                embedding_dim=768,
                chunk_size=512,
                chunk_overlap=100,
                max_embed_chars=2000,
                top_k=10,
                max_distance=0.7,
                system_prompt="",
                ignore_dirs=frozenset(),
            )

    def test_enable_ocr_none_allowed(self, tmp_path):
        """enable_ocr is nullable, None means auto."""
        c = Config(
            data_root=tmp_path,
            documents_dir=tmp_path / "docs",
            data_dir=tmp_path / "data",
            lancedb_dir=tmp_path / "data" / "lancedb",
            models_dir=tmp_path / "models",
            chat_model="qwen3",
            embedding_model="nomic-embed-text",
            embedding_dim=768,
            chunk_size=512,
            chunk_overlap=100,
            max_embed_chars=2000,
            top_k=10,
            max_distance=0.7,
            system_prompt="You are helpful.",
            ignore_dirs=frozenset(),
            enable_ocr=None,
        )
        assert c.enable_ocr is None


class TestEmptyStringToNone:
    def test_empty_temperature_becomes_none(self, tmp_path):
        env = _clean_env(tmp_path)
        env["LILBEE_TEMPERATURE"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.temperature is None

    def test_whitespace_seed_becomes_none(self, tmp_path):
        env = _clean_env(tmp_path)
        env["LILBEE_SEED"] = "   "
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.seed is None


class TestIgnoreDirsFallback:
    def test_non_string_non_collection_returns_defaults(self, tmp_path):
        env = _clean_env(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config(ignore_dirs=42)  # type: ignore[arg-type]
        assert c.ignore_dirs == DEFAULT_IGNORE_DIRS


class TestOllamaHostFallback:
    def test_ollama_host_sets_litellm_base_url(self, tmp_path):
        env = _clean_env(tmp_path)
        env["OLLAMA_HOST"] = "http://custom:11434"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.litellm_base_url == "http://custom:11434"


class TestParseEnableOcrFallback:
    def test_non_string_non_bool_coerced_via_bool(self):
        """An integer like 42 falls through to bool(v)."""
        from lilbee.config import Config

        assert Config._parse_enable_ocr(42) is True
        assert Config._parse_enable_ocr(0) is False


class TestDefaultCrawlExcludePatterns:
    """The out-of-the-box default exclude list blocks common noise without
    accidentally rejecting real content URLs."""

    def _matches_any(self, url: str) -> bool:
        import re

        from lilbee.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS

        return any(re.search(p, url) for p in DEFAULT_CRAWL_EXCLUDE_PATTERNS)

    def test_every_pattern_compiles(self):
        """Every shipped default must be valid Python regex."""
        import re

        from lilbee.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS

        for pattern in DEFAULT_CRAWL_EXCLUDE_PATTERNS:
            re.compile(pattern)

    def test_every_category_contributes(self):
        """Each per-category tuple appears in the master default list."""
        from lilbee.config import (
            _ARCHIVE_EXCLUDE,
            _ATTACHMENT_EXCLUDE,
            _AUTH_EXCLUDE,
            _DUPLICATE_VIEW_EXCLUDE,
            _ECOMMERCE_EXCLUDE,
            _FEED_EXCLUDE,
            _META_EXCLUDE,
            _TRACKING_EXCLUDE,
            _WP_EXCLUDE,
            DEFAULT_CRAWL_EXCLUDE_PATTERNS,
        )

        for category in (
            _WP_EXCLUDE,
            _ARCHIVE_EXCLUDE,
            _FEED_EXCLUDE,
            _DUPLICATE_VIEW_EXCLUDE,
            _ATTACHMENT_EXCLUDE,
            _AUTH_EXCLUDE,
            _ECOMMERCE_EXCLUDE,
            _TRACKING_EXCLUDE,
            _META_EXCLUDE,
        ):
            assert len(category) >= 1
            for p in category:
                assert p in DEFAULT_CRAWL_EXCLUDE_PATTERNS

    def test_wordpress_noise_matches(self):
        for url in (
            "https://example.com/wp-admin/",
            "https://example.com/wp-login.php",
            "https://example.com/xmlrpc.php",
            "https://example.com/wp-json/wp/v2/posts",
            "https://example.com/wp-cron.php",
            "https://example.com/wp-includes/js/jquery/jquery.js",
            "https://example.com/wp-content/uploads/2024/06/banner.png",
            "https://example.com/?p=123",
            "https://example.com/?page_id=45",
            "https://example.com/?cat=7",
        ):
            assert self._matches_any(url), f"should exclude: {url}"

    def test_archive_and_pagination_matches(self):
        for url in (
            "https://example.com/page/5/",
            "https://example.com/?paged=3",
            "https://example.com/?page=2",
            "https://example.com/2024/06/",
            "https://example.com/2024/06/15/",
            "https://example.com/2024/",
            "https://example.com/tag/gardening/",
            "https://example.com/category/growing/",
            "https://example.com/author/tobias/",
            "https://example.com/archive/",
            "https://example.com/comment-page-2",
        ):
            assert self._matches_any(url), f"should exclude: {url}"

    def test_feed_matches(self):
        for url in (
            "https://example.com/feed/",
            "https://example.com/feed/atom/",
            "https://example.com/comments/feed/",
            "https://example.com/rss/",
        ):
            assert self._matches_any(url), f"should exclude: {url}"

    def test_duplicate_view_matches(self):
        for url in (
            "https://example.com/article/amp/",
            "https://example.com/article/?amp=1",
            "https://example.com/article/?print=1",
            "https://example.com/article/?preview=true",
            "https://example.com/article/print/",
        ):
            assert self._matches_any(url), f"should exclude: {url}"

    def test_auth_matches(self):
        for url in (
            "https://example.com/login",
            "https://example.com/logout",
            "https://example.com/register",
            "https://example.com/signup",
            "https://example.com/my-account/orders/",
            "https://example.com/profile/settings",
            "https://example.com/password-reset",
        ):
            assert self._matches_any(url), f"should exclude: {url}"

    def test_ecommerce_matches(self):
        for url in (
            "https://example.com/cart",
            "https://example.com/checkout/step1",
            "https://example.com/wishlist",
            "https://example.com/orders",
            "https://example.com/compare",
            "https://example.com/products.json",
        ):
            assert self._matches_any(url), f"should exclude: {url}"

    def test_tracking_param_matches(self):
        for url in (
            "https://example.com/article?utm_source=newsletter",
            "https://example.com/?fbclid=abc123",
            "https://example.com/?gclid=xyz",
            "https://example.com/?msclkid=1",
            "https://example.com/?mc_cid=campaign1",
            "https://example.com/?mkt_tok=token",
            "https://example.com/?_hsenc=enc",
            "https://example.com/?igshid=ig",
            "https://example.com/?pk_campaign=spring",
            "https://example.com/?affiliate=partner",
        ):
            assert self._matches_any(url), f"should exclude: {url}"

    def test_meta_and_static_matches(self):
        for url in (
            "https://example.com/sitemap.xml",
            "https://example.com/sitemap_index.xml",
            "https://example.com/robots.txt",
            "https://example.com/humans.txt",
            "https://example.com/favicon.ico",
            "https://example.com/.well-known/security.txt",
            "https://example.com/files/report.pdf",
            "https://example.com/img/logo.png",
            "https://example.com/video.mp4",
            "https://example.com/dist/app.js",
            "https://example.com/style.css",
        ):
            assert self._matches_any(url), f"should exclude: {url}"

    def test_content_urls_pass_through(self):
        """Real content URLs must NOT match any default pattern."""
        for url in (
            "https://example.com/",
            "https://example.com/blog/how-to-grow-basil",
            "https://example.com/docs/installation",
            "https://example.com/about-us/team",
            "https://example.com/products/widget-1000",
            "https://example.com/2024-annual-report",
            "https://example.com/tutorials/setup",
            "https://example.com/post/why-gardening-matters",
            "https://example.com/plant_problems/yellow-leaves",
        ):
            assert not self._matches_any(url), f"should NOT exclude: {url}"


class TestCrawlExcludePatternsValidator:
    def test_newline_separated_string_splits(self):
        """Env vars come in as strings; validator splits by newline."""
        from lilbee.config import Config

        result = Config._split_crawl_exclude_patterns("/page/\\d+\n/tag/\n/category/")
        assert result == ["/page/\\d+", "/tag/", "/category/"]

    def test_list_passes_through_unchanged(self):
        """TOML lists and programmatic lists pass through the validator."""
        from lilbee.config import Config

        result = Config._split_crawl_exclude_patterns(["/page/", "/tag/"])
        assert result == ["/page/", "/tag/"]

    def test_empty_string_yields_empty_list(self):
        """Empty env var collapses to an empty list, disabling the filter."""
        from lilbee.config import Config

        assert Config._split_crawl_exclude_patterns("") == []
        assert Config._split_crawl_exclude_patterns("\n\n  \n") == []


class TestPlainEnvSourceSkipsEmpty:
    def test_empty_chat_model_uses_default(self, tmp_path):
        env = _clean_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.chat_model == "qwen3:0.6b"  # default, not empty


def _validator_env(tmp_path: Path) -> dict[str, str]:
    """_clean_env plus removal of the skip bypass so the validator fires."""
    env = _clean_env(tmp_path)
    env.pop("LILBEE_SKIP_MODEL_TASK_VALIDATION", None)
    return env


class TestModelTaskFieldValidator:
    """Per-role catalog-task validation runs on every write path (construction + assignment)."""

    def test_chat_slot_accepts_chat_model(self, tmp_path):
        env = _validator_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "qwen3:0.6b"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.chat_model.endswith("qwen3:0.6b")

    def test_chat_slot_rejects_vision_model(self, tmp_path):
        env = _validator_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "lightonocr:2-1b"
        with (
            mock.patch.dict(os.environ, env, clear=True),
            pytest.raises(Exception, match="vision"),  # pydantic ValidationError wraps it
        ):
            Config()

    def test_chat_slot_rejects_reranker_model(self, tmp_path):
        env = _validator_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "bge-reranker-v2-m3:latest"
        with (
            mock.patch.dict(os.environ, env, clear=True),
            pytest.raises(Exception, match="rerank"),
        ):
            Config()

    def test_embedding_slot_rejects_chat_model(self, tmp_path):
        env = _validator_env(tmp_path)
        env["LILBEE_EMBEDDING_MODEL"] = "qwen3:0.6b"
        with (
            mock.patch.dict(os.environ, env, clear=True),
            pytest.raises(Exception, match="chat"),
        ):
            Config()

    def test_vision_slot_rejects_chat_model(self, tmp_path):
        env = _validator_env(tmp_path)
        env["LILBEE_VISION_MODEL"] = "qwen3:0.6b"
        with (
            mock.patch.dict(os.environ, env, clear=True),
            pytest.raises(Exception, match="chat"),
        ):
            Config()

    def test_reranker_slot_rejects_vision_model(self, tmp_path):
        env = _validator_env(tmp_path)
        env["LILBEE_RERANKER_MODEL"] = "lightonocr:2-1b"
        with (
            mock.patch.dict(os.environ, env, clear=True),
            pytest.raises(Exception, match="vision"),
        ):
            Config()

    def test_empty_vision_model_allowed(self, tmp_path):
        """Vision and reranker roles allow empty strings (unset)."""
        env = _validator_env(tmp_path)
        env["LILBEE_VISION_MODEL"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.vision_model == ""

    def test_assignment_path_rejects_wrong_task(self, tmp_path):
        """validate_assignment=True means cfg.X = ref also runs the validator."""
        env = _validator_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "qwen3:0.6b"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            with pytest.raises(Exception, match="vision"):
                c.chat_model = "lightonocr:2-1b"

    def test_provider_prefix_canonicalized_on_assignment(self, tmp_path):
        """Direct ``cfg.X = ref`` canonicalizes provider-prefixed refs.

        Mirrors PATCH /api/models behavior: the validator returns the
        catalog's canonical ``name:tag`` so stored refs match the
        registry key regardless of input variant. Covers R4-F2.
        """
        env = _validator_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "qwen3:0.6b"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            c.chat_model = "ollama/qwen3:0.6b"
            assert c.chat_model == "qwen3:0.6b"

    def test_hf_repo_canonicalized_on_assignment(self, tmp_path):
        """``hf_repo`` form canonicalizes to the catalog ``name:tag``."""
        env = _validator_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "qwen3:0.6b"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            c.reranker_model = "gpustack/bge-reranker-v2-m3-GGUF:latest"
            assert c.reranker_model == "bge-reranker-v2-m3:latest"

    def test_env_load_canonicalizes(self, tmp_path):
        """Env-var load path also canonicalizes (validator runs at construction)."""
        env = _validator_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "ollama/qwen3:0.6b"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.chat_model == "qwen3:0.6b"

    def test_out_of_catalog_rejected(self, tmp_path):
        """Out-of-catalog model names are rejected since we can't verify the role."""
        env = _validator_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "totally-unknown-model:99b"
        with (
            mock.patch.dict(os.environ, env, clear=True),
            pytest.raises(Exception, match="featured catalog"),
        ):
            Config()

    def test_skip_env_var_disables_check(self, tmp_path):
        """LILBEE_SKIP_MODEL_TASK_VALIDATION bypasses the role check when pytest is imported."""
        env = _validator_env(tmp_path)
        env["LILBEE_SKIP_MODEL_TASK_VALIDATION"] = "1"
        env["LILBEE_CHAT_MODEL"] = "totally-unknown-model:99b"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.chat_model.endswith("totally-unknown-model:99b")

    def test_skip_env_var_alone_does_not_bypass_in_production(self, tmp_path):
        """Shell-level env var without the pytest sentinel must not bypass validation."""
        import sys

        env = _validator_env(tmp_path)
        env["LILBEE_SKIP_MODEL_TASK_VALIDATION"] = "1"
        env["LILBEE_CHAT_MODEL"] = "totally-unknown-model:99b"
        saved_pytest = sys.modules.pop("pytest", None)
        try:
            with (
                mock.patch.dict(os.environ, env, clear=True),
                pytest.raises(Exception, match="featured catalog"),
            ):
                Config()
        finally:
            if saved_pytest is not None:
                sys.modules["pytest"] = saved_pytest

    def test_whitespace_only_model_normalized_to_empty(self, tmp_path):
        """Whitespace-only values normalize to empty instead of raising."""
        env = _validator_env(tmp_path)
        env["LILBEE_VISION_MODEL"] = "   "
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.vision_model == ""

    def test_whitespace_assignment_normalized_to_empty(self, tmp_path):
        """Direct assignment of whitespace also normalizes cleanly."""
        env = _validator_env(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            c.vision_model = " "
            assert c.vision_model == ""

    def test_whitespace_chat_model_rejected(self, tmp_path):
        """Whitespace-only chat_model is rejected (required field)."""
        from pydantic import ValidationError

        env = _validator_env(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            with pytest.raises(ValidationError, match="chat_model"):
                c.chat_model = "   "

    def test_whitespace_embedding_model_rejected(self, tmp_path):
        """Whitespace-only embedding_model is rejected (required field)."""
        from pydantic import ValidationError

        env = _validator_env(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            with pytest.raises(ValidationError, match="embedding_model"):
                c.embedding_model = "   "

    def test_task_mismatch_message_parity_with_handler(self, tmp_path):
        """Validator and handler produce identical 422 messages via shared helper."""
        from lilbee.models import ModelTask
        from lilbee.server.handlers import format_task_mismatch

        env = _validator_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "lightonocr:2-1b"
        with (
            mock.patch.dict(os.environ, env, clear=True),
            pytest.raises(Exception) as exc_info,
        ):
            Config()

        handler_message = format_task_mismatch("lightonocr:2-1b", ModelTask.VISION, ModelTask.CHAT)
        # Pydantic wraps the raw ValueError; check the core message is present.
        assert handler_message in str(exc_info.value)
