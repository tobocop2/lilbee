"""Tests for Config (pydantic-settings BaseSettings) and env var overrides."""

import os
from pathlib import Path
from unittest import mock

import pytest

from lilbee.config import (
    CHUNKS_TABLE,
    DEFAULT_IGNORE_DIRS,
    SOURCES_TABLE,
    Config,
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
            assert c.chat_model == "qwen3:8b"
            assert c.embedding_model == "nomic-embed-text"
            assert c.embedding_dim == 768
            assert c.chunk_size == 512
            assert c.chunk_overlap == 100
            assert c.max_embed_chars == 2000
            assert c.top_k == 10
            assert c.max_distance == 0.7
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
            assert c.chat_model == "llama3"

    def test_embedding_model_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_EMBEDDING_MODEL": "mxbai-embed-large"}):
            c = Config()
            assert c.embedding_model == "mxbai-embed-large"

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
            assert c.chat_model == "my-saved-model"

    def test_env_var_overrides_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('chat_model = "toml-model"\n')
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_CHAT_MODEL"] = "env-model"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.chat_model == "env-model"

    def test_no_toml_uses_defaults(self, tmp_path):
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.chat_model == "qwen3:8b"

    def test_corrupt_toml_uses_defaults(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("this is not valid TOML [[[")
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.chat_model == "qwen3:8b"

    def test_embedding_model_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('embedding_model = "my-embed"\n')
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.embedding_model == "my-embed"

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

    def test_vision_model_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('vision_model = "maternion/LightOnOCR-2"\n')
        env = _clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.vision_model == "maternion/LightOnOCR-2"

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


class TestVisionModelConfig:
    def test_default_vision_model_is_empty(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.vision_model == ""

    def test_vision_model_env_override(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_VISION_MODEL": "minicpm-v"}):
            c = Config()
            assert c.vision_model == "minicpm-v"


class TestVisionTimeoutConfig:
    def test_valid_timeout_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_VISION_TIMEOUT": "60.5"}):
            c = Config()
            assert c.vision_timeout == 60.5

    def test_no_timeout_env_returns_default(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, _clean_env(tmp_path), clear=True):
            c = Config()
            assert c.vision_timeout == 120.0

    def test_zero_timeout_means_no_limit(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_VISION_TIMEOUT": "0"}):
            c = Config()
            assert c.vision_timeout == 0

    def test_invalid_timeout_raises(self) -> None:
        with (
            mock.patch.dict(os.environ, {"LILBEE_VISION_TIMEOUT": "abc"}),
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
        assert c.generation_options() == {}

    def test_includes_set_values(self):
        c = Config()
        c.temperature = 0.3
        c.seed = 42
        c.top_p = None
        c.top_k_sampling = None
        c.repeat_penalty = None
        c.num_ctx = None
        opts = c.generation_options()
        assert opts == {"temperature": 0.3, "seed": 42}

    def test_remaps_top_k_sampling(self):
        c = Config()
        c.temperature = None
        c.top_p = None
        c.top_k_sampling = 40
        c.repeat_penalty = None
        c.num_ctx = None
        c.seed = None
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
    def test_empty_chat_model_rejected(self):
        with pytest.raises(Exception, match="at least 1 character"):
            Config(
                data_root=Path("/tmp"),
                documents_dir=Path("/tmp/docs"),
                data_dir=Path("/tmp/data"),
                lancedb_dir=Path("/tmp/data/lancedb"),
                models_dir=Path("/tmp/models"),
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

    def test_empty_embedding_model_rejected(self):
        with pytest.raises(Exception, match="at least 1 character"):
            Config(
                data_root=Path("/tmp"),
                documents_dir=Path("/tmp/docs"),
                data_dir=Path("/tmp/data"),
                lancedb_dir=Path("/tmp/data/lancedb"),
                models_dir=Path("/tmp/models"),
                chat_model="qwen3:8b",
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

    def test_empty_system_prompt_rejected(self):
        with pytest.raises(Exception, match="at least 1 character"):
            Config(
                data_root=Path("/tmp"),
                documents_dir=Path("/tmp/docs"),
                data_dir=Path("/tmp/data"),
                lancedb_dir=Path("/tmp/data/lancedb"),
                models_dir=Path("/tmp/models"),
                chat_model="qwen3:8b",
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

    def test_empty_vision_model_allowed(self):
        """vision_model is nullable — empty string is valid."""
        c = Config(
            data_root=Path("/tmp"),
            documents_dir=Path("/tmp/docs"),
            data_dir=Path("/tmp/data"),
            lancedb_dir=Path("/tmp/data/lancedb"),
            models_dir=Path("/tmp/models"),
            chat_model="qwen3:8b",
            embedding_model="nomic-embed-text",
            embedding_dim=768,
            chunk_size=512,
            chunk_overlap=100,
            max_embed_chars=2000,
            top_k=10,
            max_distance=0.7,
            system_prompt="You are helpful.",
            ignore_dirs=frozenset(),
            vision_model="",
        )
        assert c.vision_model == ""




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


class TestPlainEnvSourceSkipsEmpty:
    def test_empty_chat_model_uses_default(self, tmp_path):
        env = _clean_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.chat_model == "qwen3:8b"  # default, not empty
