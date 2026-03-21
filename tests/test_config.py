"""Tests for Config dataclass and env var overrides."""

import os
from pathlib import Path
from unittest import mock

import pytest

from lilbee.config import CHUNKS_TABLE, DEFAULT_IGNORE_DIRS, SOURCES_TABLE, Config, _load_setting


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

        def fake_get(root, key):
            if key == "chat_model":
                return "my-saved-model"
            return None

        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.settings.get", side_effect=fake_get),
        ):
            c = Config.from_env()
            assert c.chat_model == "my-saved-model"

    def test_env_var_overrides_config_toml(self):
        # When LILBEE_CHAT_MODEL is set, _load_chat_model skips settings.get
        # for chat_model. Other fields still call settings.get (that's expected).
        with (
            mock.patch.dict(
                os.environ,
                {"LILBEE_CHAT_MODEL": "env-model"},
            ),
            mock.patch("lilbee.settings.get", return_value=None) as mock_get,
        ):
            c = Config.from_env()
            assert c.chat_model == "env-model"
            # settings.get should not be called with "chat_model" key
            # (but may be called for other keys like embedding_model)
            for call in mock_get.call_args_list:
                assert call[0][1] != "chat_model"

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
            mock.patch("lilbee.settings.get", side_effect=ValueError("bad toml")),
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


class TestCorsOriginsConfig:
    def test_cors_origins_from_env(self) -> None:
        with mock.patch.dict(
            os.environ, {"LILBEE_CORS_ORIGINS": "app://obsidian.md,https://my-app.com"}
        ):
            c = Config.from_env()
            assert c.cors_origins == ["app://obsidian.md", "https://my-app.com"]

    def test_cors_origins_default_empty(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "LILBEE_CORS_ORIGINS"}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            c = Config.from_env()
            assert c.cors_origins == []


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


class TestLoadSettingHelper:
    def test_returns_default_when_no_env_and_no_saved(self, tmp_path):
        with mock.patch("lilbee.settings.get", return_value=None):
            result = _load_setting(tmp_path, "temperature", "TEMPERATURE", 0.7, float)
        assert result == 0.7

    def test_returns_saved_when_no_env(self, tmp_path):
        with mock.patch("lilbee.settings.get", return_value="0.5"):
            result = _load_setting(tmp_path, "temperature", "TEMPERATURE", 0.7, float)
        assert result == 0.5

    def test_env_var_takes_precedence(self, tmp_path):
        with (
            mock.patch.dict(os.environ, {"LILBEE_TEMPERATURE": "0.3"}),
            mock.patch("lilbee.settings.get", return_value="0.5"),
        ):
            result = _load_setting(tmp_path, "temperature", "TEMPERATURE", 0.7, float)
        assert result == 0.3

    def test_corrupt_toml_falls_back_to_default(self, tmp_path):
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=ValueError("bad")),
        ):
            result = _load_setting(tmp_path, "temperature", "TEMPERATURE", 0.7, float)
        assert result == 0.7

    def test_os_error_falls_back_to_default(self, tmp_path):
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=OSError("disk error")),
        ):
            result = _load_setting(tmp_path, "temperature", "TEMPERATURE", 0.7, float)
        assert result == 0.7

    def test_empty_saved_value_returns_default(self, tmp_path):
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", return_value=""),
        ):
            result = _load_setting(tmp_path, "temperature", "TEMPERATURE", 0.7, float)
        assert result == 0.7

    def test_int_type_coercion(self, tmp_path):
        with mock.patch("lilbee.settings.get", return_value="42"):
            result = _load_setting(tmp_path, "seed", "SEED", 0, int)
        assert result == 42
        assert isinstance(result, int)

    def test_str_type_coercion(self, tmp_path):
        with mock.patch("lilbee.settings.get", return_value="my-model"):
            result = _load_setting(tmp_path, "embedding_model", "EMBEDDING_MODEL", "default", str)
        assert result == "my-model"

    def test_returns_none_default_for_optional(self, tmp_path):
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", return_value=None),
        ):
            result = _load_setting(tmp_path, "temperature", "TEMPERATURE", None, float)
        assert result is None


class TestPersistedSettings:
    """Test that all settings load from config.toml at startup."""

    def _fake_get(self, data):
        def fn(root, key):
            return data.get(key)

        return fn

    def test_embedding_model_from_config(self):
        saved = {"embedding_model": "my-embed"}
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.embedding_model == "my-embed"

    def test_temperature_from_config(self):
        saved = {"temperature": "0.5"}
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.temperature == 0.5

    def test_top_p_from_config(self):
        saved = {"top_p": "0.9"}
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.top_p == 0.9

    def test_top_k_from_config(self):
        saved = {"top_k": "20"}
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.top_k == 20

    def test_top_k_sampling_from_config(self):
        saved = {"top_k_sampling": "40"}
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.top_k_sampling == 40

    def test_repeat_penalty_from_config(self):
        saved = {"repeat_penalty": "1.2"}
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.repeat_penalty == 1.2

    def test_num_ctx_from_config(self):
        saved = {"num_ctx": "4096"}
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.num_ctx == 4096

    def test_seed_from_config(self):
        saved = {"seed": "123"}
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.seed == 123

    def test_system_prompt_from_config(self):
        saved = {"system_prompt": "You are a pirate."}
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.system_prompt == "You are a pirate."

    def test_env_var_overrides_config_toml_for_temperature(self):
        saved = {"temperature": "0.5"}
        with (
            mock.patch.dict(os.environ, {"LILBEE_TEMPERATURE": "0.9"}),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.temperature == 0.9

    def test_env_var_overrides_config_toml_for_system_prompt(self):
        saved = {"system_prompt": "Be verbose."}
        with (
            mock.patch.dict(os.environ, {"LILBEE_SYSTEM_PROMPT": "Be brief."}),
            mock.patch("lilbee.settings.get", side_effect=self._fake_get(saved)),
        ):
            c = Config.from_env()
            assert c.system_prompt == "Be brief."


class TestEmptyStringValidation:
    def test_empty_chat_model_rejected(self):
        with pytest.raises(Exception, match="at least 1 character"):
            Config(
                data_root=Path("/tmp"),
                documents_dir=Path("/tmp/docs"),
                data_dir=Path("/tmp/data"),
                lancedb_dir=Path("/tmp/data/lancedb"),
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
