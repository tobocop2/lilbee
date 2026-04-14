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
    is accidentally picked up by pydantic-settings.
    """
    env = {
        k: v for k, v in os.environ.items() if not k.startswith("LILBEE_") and k != "OLLAMA_HOST"
    }
    if tmp_path is not None:
        env["LILBEE_DATA"] = str(tmp_path)
    return env


class TestFromEnvDefaults:
    def test_default_values(self, tmp_path):
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.chat_model == "qwen3:latest"
            assert c.embedding_model == "nomic-embed-text:latest"
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
        result = Config._normalize_model_tag("")
        assert result == ""

    def test_embedding_dim_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_EMBEDDING_DIM": "1024"}):
            c = Config()
            assert c.embedding_dim == 1024

    def test_chunk_size_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHUNK_SIZE": "256"}):
            c = Config()
            assert c.chunk_size == 256

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
            assert c.chat_model == "qwen3:latest"

    def test_corrupt_toml_uses_defaults(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("this is not valid TOML [[[")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.chat_model == "qwen3:latest"

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


class TestPlainEnvSourceSkipsEmpty:
    def test_empty_chat_model_uses_default(self, tmp_path):
        env = _clean_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.chat_model == "qwen3:latest"  # default, not empty
