"""Tests for Config dataclass and env var overrides."""

import os
from unittest import mock

import pytest

from lilbee.config import CHUNKS_TABLE, DEFAULT_IGNORE_DIRS, SOURCES_TABLE, Config


class TestFromEnvDefaults:
    def test_default_values(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LILBEE_")}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            c = Config.from_env()
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
            c = Config.from_env()
            assert c.data_root == tmp_path
            assert c.documents_dir == tmp_path / "documents"
            assert c.data_dir == tmp_path / "data"
            assert c.lancedb_dir == tmp_path / "data" / "lancedb"

    def test_data_root_default_uses_platform(self):
        env = {k: v for k, v in os.environ.items() if k != "LILBEE_DATA"}
        env["LILBEE_DATA"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config.from_env()
            assert str(c.data_root).endswith("lilbee")

    def test_chat_model_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHAT_MODEL": "llama3"}):
            c = Config.from_env()
            assert c.chat_model == "llama3"

    def test_embedding_model_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_EMBEDDING_MODEL": "mxbai-embed-large"}):
            c = Config.from_env()
            assert c.embedding_model == "mxbai-embed-large"

    def test_embedding_dim_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_EMBEDDING_DIM": "1024"}):
            c = Config.from_env()
            assert c.embedding_dim == 1024

    def test_chunk_size_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHUNK_SIZE": "256"}):
            c = Config.from_env()
            assert c.chunk_size == 256

    def test_chunk_overlap_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHUNK_OVERLAP": "50"}):
            c = Config.from_env()
            assert c.chunk_overlap == 50

    def test_top_k_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_TOP_K": "20"}):
            c = Config.from_env()
            assert c.top_k == 20

    def test_max_embed_chars_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_MAX_EMBED_CHARS": "3000"}):
            c = Config.from_env()
            assert c.max_embed_chars == 3000

    def test_max_distance_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_MAX_DISTANCE": "1.5"}):
            c = Config.from_env()
            assert c.max_distance == 1.5

    def test_system_prompt_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_SYSTEM_PROMPT": "You are a pirate."}):
            c = Config.from_env()
            assert c.system_prompt == "You are a pirate."


class TestPersistedChatModel:
    def test_config_toml_used_when_no_env_var(self):
        env = {k: v for k, v in os.environ.items() if k != "LILBEE_CHAT_MODEL"}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.settings.get", return_value="my-saved-model"),
        ):
            c = Config.from_env()
            assert c.chat_model == "my-saved-model"

    def test_env_var_overrides_config_toml(self):
        # When LILBEE_CHAT_MODEL is set, settings.get must NOT be called for chat_model.
        # We also set LILBEE_VISION_MODEL to avoid settings reads.
        with (
            mock.patch.dict(
                os.environ,
                {
                    "LILBEE_CHAT_MODEL": "env-model",
                    "LILBEE_VISION_MODEL": "noop",
                },
            ),
            mock.patch("lilbee.settings.get") as mock_get,
        ):
            c = Config.from_env()
            mock_get.assert_not_called()
            assert c.chat_model == "env-model"

    def test_no_persisted_value_keeps_default(self):
        env = {k: v for k, v in os.environ.items() if k != "LILBEE_CHAT_MODEL"}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            c = Config.from_env()
            assert c.chat_model == "qwen3:8b"

    def test_corrupt_config_toml_keeps_default(self):
        env = {k: v for k, v in os.environ.items() if k != "LILBEE_CHAT_MODEL"}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.settings.get", side_effect=Exception("bad toml")),
        ):
            c = Config.from_env()
            assert c.chat_model == "qwen3:8b"


class TestVisionModelConfig:
    def test_default_vision_model_is_empty(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("LILBEE_")}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            c = Config.from_env()
            assert c.vision_model == ""

    def test_vision_model_env_override(self) -> None:
        with (
            mock.patch.dict(os.environ, {"LILBEE_VISION_MODEL": "minicpm-v"}),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            c = Config.from_env()
            assert c.vision_model == "minicpm-v"

    def test_vision_model_from_config_toml(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "LILBEE_VISION_MODEL"}

        def fake_get(root, key):
            if key == "vision_model":
                return "maternion/LightOnOCR-2"
            return None

        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.settings.get", side_effect=fake_get),
        ):
            c = Config.from_env()
            assert c.vision_model == "maternion/LightOnOCR-2"


class TestVisionTimeoutConfig:
    def test_valid_timeout_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_VISION_TIMEOUT": "60.5"}):
            c = Config.from_env()
            assert c.vision_timeout == 60.5

    def test_no_timeout_env_returns_default(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "LILBEE_VISION_TIMEOUT"}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            c = Config.from_env()
            assert c.vision_timeout == 120.0

    def test_zero_timeout_means_no_limit(self) -> None:
        with mock.patch.dict(os.environ, {"LILBEE_VISION_TIMEOUT": "0"}):
            c = Config.from_env()
            assert c.vision_timeout == 0

    def test_invalid_timeout_warns_and_returns_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        with (
            mock.patch.dict(os.environ, {"LILBEE_VISION_TIMEOUT": "abc"}),
            caplog.at_level(logging.WARNING, logger="lilbee.config"),
        ):
            c = Config.from_env()
        assert c.vision_timeout == 120.0
        assert any("Invalid LILBEE_VISION_TIMEOUT" in r.message for r in caplog.records)


class TestLocalDotLilbee:
    def test_local_lilbee_overrides_default(self, tmp_path):
        local = tmp_path / ".lilbee"
        local.mkdir()
        env = {k: v for k, v in os.environ.items() if k != "LILBEE_DATA"}
        env["LILBEE_DATA"] = ""
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.platform.find_local_root", return_value=local),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            c = Config.from_env()
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
            c = Config.from_env()
            assert c.data_root == explicit

    def test_no_local_uses_platform_default(self):
        env = {k: v for k, v in os.environ.items() if k != "LILBEE_DATA"}
        env["LILBEE_DATA"] = ""
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.platform.find_local_root", return_value=None),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            c = Config.from_env()
            assert c.data_root.name == "lilbee"
            assert c.data_root.name != ".lilbee"


class TestGenerationOptions:
    def test_empty_when_all_none(self):
        c = Config.from_env()
        c.temperature = None
        c.top_p = None
        c.top_k_sampling = None
        c.repeat_penalty = None
        c.num_ctx = None
        c.seed = None
        assert c.generation_options() == {}

    def test_includes_set_values(self):
        c = Config.from_env()
        c.temperature = 0.3
        c.seed = 42
        c.top_p = None
        c.top_k_sampling = None
        c.repeat_penalty = None
        c.num_ctx = None
        opts = c.generation_options()
        assert opts == {"temperature": 0.3, "seed": 42}

    def test_remaps_top_k_sampling(self):
        c = Config.from_env()
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
        c = Config.from_env()
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
            c = Config.from_env()
            assert c.temperature == 0.3
            assert c.top_p == 0.95
            assert c.top_k_sampling == 40
            assert c.repeat_penalty == 1.1
            assert c.num_ctx == 4096
            assert c.seed == 123


class TestIgnoreDirs:
    def test_default_ignore_dirs_contains_expected(self):
        c = Config.from_env()
        for name in ["node_modules", "__pycache__", "venv", "build", "dist"]:
            assert name in c.ignore_dirs

    def test_lilbee_ignore_env_adds_custom_entries(self):
        with mock.patch.dict(os.environ, {"LILBEE_IGNORE": "output,generated"}):
            c = Config.from_env()
            assert "output" in c.ignore_dirs
            assert "generated" in c.ignore_dirs
            assert "node_modules" in c.ignore_dirs

    def test_lilbee_ignore_empty_string(self):
        with mock.patch.dict(os.environ, {"LILBEE_IGNORE": ""}):
            c = Config.from_env()
            assert c.ignore_dirs == DEFAULT_IGNORE_DIRS

    def test_lilbee_ignore_strips_whitespace(self):
        with mock.patch.dict(os.environ, {"LILBEE_IGNORE": " foo , bar "}):
            c = Config.from_env()
            assert "foo" in c.ignore_dirs
            assert "bar" in c.ignore_dirs
