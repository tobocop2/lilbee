"""Tests for config path resolution and env var overrides."""

import importlib
import os
from unittest import mock


def _reload_config():
    """Reload config module to pick up env/platform changes."""
    import lilbee.config

    return importlib.reload(lilbee.config)


class TestDefaultDataDir:
    def test_darwin_uses_application_support(self):
        with (
            mock.patch.dict(os.environ, {"LILBEE_DATA": ""}, clear=False),
            mock.patch("sys.platform", "darwin"),
        ):
            cfg = _reload_config()
            assert "Application Support" in str(cfg._default_data_dir())
            assert str(cfg._default_data_dir()).endswith("lilbee")

    def test_linux_uses_xdg_data_home(self):
        with (
            mock.patch.dict(
                os.environ, {"LILBEE_DATA": "", "XDG_DATA_HOME": "/tmp/xdg"}, clear=False
            ),
            mock.patch("sys.platform", "linux"),
        ):
            cfg = _reload_config()
            assert cfg._default_data_dir().parts[-1] == "lilbee"

    def test_linux_defaults_to_local_share(self):
        env = {k: v for k, v in os.environ.items() if k != "XDG_DATA_HOME"}
        env["LILBEE_DATA"] = ""
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("sys.platform", "linux"),
        ):
            cfg = _reload_config()
            assert ".local/share/lilbee" in str(cfg._default_data_dir())

    def test_windows_uses_localappdata(self, tmp_path):
        with (
            mock.patch.dict(
                os.environ, {"LILBEE_DATA": "", "LOCALAPPDATA": str(tmp_path)}, clear=False
            ),
            mock.patch("sys.platform", "win32"),
        ):
            cfg = _reload_config()
            assert str(tmp_path) in str(cfg._default_data_dir())

    def test_windows_fallback_without_localappdata(self):
        env = {k: v for k, v in os.environ.items() if k != "LOCALAPPDATA"}
        env["LILBEE_DATA"] = ""
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("sys.platform", "win32"),
        ):
            cfg = _reload_config()
            assert "lilbee" in str(cfg._default_data_dir())


class TestEnvVarOverrides:
    def test_lilbee_data_overrides_paths(self, tmp_path):
        with mock.patch.dict(os.environ, {"LILBEE_DATA": str(tmp_path)}):
            cfg = _reload_config()
            assert tmp_path / "documents" == cfg.DOCUMENTS_DIR
            assert tmp_path / "data" / "lancedb" == cfg.LANCEDB_DIR

    def test_chat_model_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHAT_MODEL": "llama3"}):
            cfg = _reload_config()
            assert cfg.CHAT_MODEL == "llama3"

    def test_embedding_model_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_EMBEDDING_MODEL": "mxbai-embed-large"}):
            cfg = _reload_config()
            assert cfg.EMBEDDING_MODEL == "mxbai-embed-large"

    def test_embedding_dim_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_EMBEDDING_DIM": "1024"}):
            cfg = _reload_config()
            assert cfg.EMBEDDING_DIM == 1024

    def test_chunk_size_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHUNK_SIZE": "256"}):
            cfg = _reload_config()
            assert cfg.CHUNK_SIZE == 256

    def test_chunk_overlap_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHUNK_OVERLAP": "50"}):
            cfg = _reload_config()
            assert cfg.CHUNK_OVERLAP == 50

    def test_top_k_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_TOP_K": "10"}):
            cfg = _reload_config()
            assert cfg.TOP_K == 10


class TestMaxEmbedChars:
    def test_max_embed_chars_default(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LILBEE_")}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = _reload_config()
            assert cfg.MAX_EMBED_CHARS == 2000

    def test_max_embed_chars_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_MAX_EMBED_CHARS": "3000"}):
            cfg = _reload_config()
            assert cfg.MAX_EMBED_CHARS == 3000


class TestDefaults:
    def test_default_values(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LILBEE_")}
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = _reload_config()
            assert cfg.CHAT_MODEL == "mistral"
            assert cfg.EMBEDDING_MODEL == "nomic-embed-text"
            assert cfg.EMBEDDING_DIM == 768
            assert cfg.CHUNK_SIZE == 512
            assert cfg.CHUNK_OVERLAP == 100
            assert cfg.MAX_EMBED_CHARS == 2000
            assert cfg.TOP_K == 10
            assert cfg.CHUNKS_TABLE == "chunks"
            assert cfg.SOURCES_TABLE == "_sources"


class TestHelpers:
    def test_env_returns_default(self):
        from lilbee.config import _env

        with mock.patch.dict(os.environ, {}, clear=True):
            assert _env("NONEXISTENT", "fallback") == "fallback"

    def test_env_returns_override(self):
        from lilbee.config import _env

        with mock.patch.dict(os.environ, {"LILBEE_NONEXISTENT": "override"}):
            assert _env("NONEXISTENT", "fallback") == "override"

    def test_env_int_returns_default(self):
        from lilbee.config import _env_int

        with mock.patch.dict(os.environ, {}, clear=True):
            assert _env_int("NONEXISTENT", 42) == 42

    def test_env_int_returns_override(self):
        from lilbee.config import _env_int

        with mock.patch.dict(os.environ, {"LILBEE_NONEXISTENT": "99"}):
            assert _env_int("NONEXISTENT", 42) == 99
