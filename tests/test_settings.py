"""Tests for persistent settings (config.toml)."""

from unittest import mock

import pytest

from lilbee.core import settings


class TestChunkSizeOverlapInvariant:
    def test_lowering_chunk_size_below_existing_overlap_is_rejected(self, monkeypatch):
        from lilbee.app import settings as appset

        monkeypatch.setattr(appset.cfg, "chunk_size", 512)
        monkeypatch.setattr(appset.cfg, "chunk_overlap", 100)
        with pytest.raises(ValueError, match="chunk_overlap"):
            appset._validate({"chunk_size": 64})

    def test_lowering_chunk_size_above_existing_overlap_is_allowed(self, monkeypatch):
        from lilbee.app import settings as appset

        monkeypatch.setattr(appset.cfg, "chunk_size", 512)
        monkeypatch.setattr(appset.cfg, "chunk_overlap", 50)
        appset._validate({"chunk_size": 256})  # 50 < 256, no raise


class TestApplySettingsRollback:
    def test_parse_error_during_persist_restores_snapshot(self, monkeypatch):
        """A non-OSError (e.g. corrupt-toml parse) during persist must roll the
        in-memory snapshot back, not leave cfg holding unpersisted values."""
        from lilbee.app import settings as appset

        original = appset.cfg.chunk_size

        def _boom(*_a, **_k):
            raise ValueError("corrupt config.toml")

        monkeypatch.setattr(appset.persistent_settings, "update_values", _boom)
        with pytest.raises(ValueError):
            appset.apply_settings_update({"chunk_size": original + 64})
        assert appset.cfg.chunk_size == original


class TestLoad:
    def test_load_missing_file_returns_empty(self, tmp_path):
        assert settings.load(tmp_path) == {}

    def test_load_existing_file(self, tmp_path):
        (tmp_path / "config.toml").write_text('chat_model = "llama3"\n')
        assert settings.load(tmp_path) == {"chat_model": "llama3"}


class TestSave:
    def test_save_creates_file(self, tmp_path):
        settings.save(tmp_path, {"chat_model": "llama3"})
        assert (tmp_path / "config.toml").exists()
        assert 'chat_model = "llama3"' in (tmp_path / "config.toml").read_text()

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "nested" / "dir"
        settings.save(nested, {"key": "value"})
        assert (nested / "config.toml").exists()

    def test_save_load_roundtrip(self, tmp_path):
        settings.save(tmp_path, {"chat_model": "phi3", "top_k": "20"})
        result = settings.load(tmp_path)
        assert result == {"chat_model": "phi3", "top_k": "20"}


class TestGet:
    def test_get_existing_key(self, tmp_path):
        (tmp_path / "config.toml").write_text('chat_model = "llama3"\n')
        assert settings.get(tmp_path, "chat_model") == "llama3"

    def test_get_missing_key(self, tmp_path):
        (tmp_path / "config.toml").write_text('chat_model = "llama3"\n')
        assert settings.get(tmp_path, "nonexistent") is None

    def test_get_missing_file(self, tmp_path):
        assert settings.get(tmp_path, "anything") is None


class TestSetValue:
    def test_set_value_creates_file(self, tmp_path):
        settings.set_value(tmp_path, "chat_model", "mistral")
        assert settings.get(tmp_path, "chat_model") == "mistral"

    def test_set_value_preserves_existing(self, tmp_path):
        (tmp_path / "config.toml").write_text('existing = "keep"\n')
        settings.set_value(tmp_path, "chat_model", "phi3")
        result = settings.load(tmp_path)
        assert result == {"existing": "keep", "chat_model": "phi3"}

    def test_set_value_overwrites_key(self, tmp_path):
        settings.set_value(tmp_path, "chat_model", "llama3")
        settings.set_value(tmp_path, "chat_model", "mistral")
        assert settings.get(tmp_path, "chat_model") == "mistral"


class TestDeleteValue:
    def test_delete_existing_key(self, tmp_path):
        settings.set_value(tmp_path, "temperature", "0.5")
        settings.delete_value(tmp_path, "temperature")
        assert settings.get(tmp_path, "temperature") is None

    def test_delete_preserves_other_keys(self, tmp_path):
        settings.set_value(tmp_path, "chat_model", "llama3")
        settings.set_value(tmp_path, "temperature", "0.5")
        settings.delete_value(tmp_path, "temperature")
        assert settings.get(tmp_path, "chat_model") == "llama3"

    def test_delete_missing_key_is_noop(self, tmp_path):
        settings.set_value(tmp_path, "chat_model", "llama3")
        settings.delete_value(tmp_path, "nonexistent")
        assert settings.get(tmp_path, "chat_model") == "llama3"

    def test_delete_from_empty_file(self, tmp_path):
        settings.delete_value(tmp_path, "anything")
        assert settings.load(tmp_path) == {}


class TestTomlEscaping:
    def test_escape_double_quotes(self, tmp_path):
        settings.set_value(tmp_path, "prompt", 'say "hello"')
        assert settings.get(tmp_path, "prompt") == 'say "hello"'

    def test_escape_backslashes(self, tmp_path):
        settings.set_value(tmp_path, "path", r"C:\Users\test")
        assert settings.get(tmp_path, "path") == r"C:\Users\test"

    def test_escape_newlines(self, tmp_path):
        settings.set_value(tmp_path, "msg", "line1\nline2")
        assert settings.get(tmp_path, "msg") == "line1\nline2"

    def test_escape_tab(self, tmp_path):
        settings.set_value(tmp_path, "msg", "col1\tcol2")
        assert settings.get(tmp_path, "msg") == "col1\tcol2"

    def test_escape_mixed(self, tmp_path):
        val = 'He said "hello" at C:\\home\n'
        settings.set_value(tmp_path, "mixed", val)
        assert settings.get(tmp_path, "mixed") == val

    def test_escape_preserves_normal_values(self, tmp_path):
        settings.set_value(tmp_path, "model", "qwen3:8b")
        assert settings.get(tmp_path, "model") == "qwen3:8b"

    def test_escape_empty_string(self, tmp_path):
        settings.set_value(tmp_path, "key", "")
        assert settings.get(tmp_path, "key") == ""

    def test_escape_toml_string_function(self):
        from lilbee.core.settings import _escape_toml_string

        assert _escape_toml_string('say "hi"') == r"say \"hi\""
        assert _escape_toml_string(r"C:\path") == r"C:\\path"
        assert _escape_toml_string("a\nb") == r"a\nb"
        assert _escape_toml_string("a\tb") == r"a\tb"
        assert _escape_toml_string("normal") == "normal"
        assert _escape_toml_string("") == ""


class TestRerankerConfig:
    """Reranker mode + prompt config fields."""

    def test_reranker_type_defaults_auto(self):
        from lilbee.core.config import Config
        from lilbee.core.config.enums import RerankerType

        assert Config().reranker_type == RerankerType.AUTO

    def test_reranker_type_rejects_junk(self):
        import pydantic
        import pytest

        from lilbee.core.config import Config

        with pytest.raises(pydantic.ValidationError):
            Config(reranker_type="bogus")

    def test_reranker_prompt_defaults_empty(self):
        from lilbee.core.config import Config

        assert Config().reranker_prompt == ""

    def test_reranker_type_is_load_affecting(self):
        from lilbee.core.config.keys import LOAD_AFFECTING_KEYS

        assert "reranker_type" in LOAD_AFFECTING_KEYS

    def test_reranker_fields_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP

        assert "reranker_type" in SETTINGS_MAP
        assert "reranker_prompt" in SETTINGS_MAP
        assert SETTINGS_MAP["reranker_type"].choices == ("auto", "cross_encoder", "llm")


class TestMemoryTuningSettingsMap:
    """The dynamic-ctx tuning knobs are surfaced in the TUI settings map."""

    def test_num_ctx_max_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP, get_default

        defn = SETTINGS_MAP["num_ctx_max"]
        assert defn.writable is True
        assert defn.nullable is True  # None = use model training_ctx as ceiling
        assert defn.group == "Generation"
        assert get_default("num_ctx_max") is None

    def test_chat_n_ctx_target_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP, get_default

        defn = SETTINGS_MAP["chat_n_ctx_target"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.group == "Generation"
        with mock.patch(
            "lilbee.core.system._read_total_memory_bytes",
            return_value=8 * 1024**3,
        ):
            assert get_default("chat_n_ctx_target") == 8192

    def test_flash_attention_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP, get_default

        defn = SETTINGS_MAP["flash_attention"]
        assert defn.writable is True
        assert defn.nullable is True  # tri-state: None=auto
        assert defn.type is bool
        assert get_default("flash_attention") is None

    def test_kv_cache_type_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP
        from lilbee.core.config.enums import KvCacheType

        defn = SETTINGS_MAP["kv_cache_type"]
        assert defn.writable is True
        assert defn.choices == tuple(t.value for t in KvCacheType)

    def test_n_gpu_layers_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP, get_default

        defn = SETTINGS_MAP["n_gpu_layers"]
        assert defn.writable is True
        assert defn.nullable is True  # None = auto/all
        assert get_default("n_gpu_layers") is None

    def test_vision_ocr_max_tokens_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP, get_default

        defn = SETTINGS_MAP["vision_ocr_max_tokens"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is int
        assert defn.group == "Ingest"
        assert get_default("vision_ocr_max_tokens") == 4096

    def test_vision_ocr_concurrency_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP, get_default

        defn = SETTINGS_MAP["vision_ocr_concurrency"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is int
        assert defn.group == "Ingest"
        assert get_default("vision_ocr_concurrency") == 4

    def test_crawl_render_mode_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP
        from lilbee.core.config.enums import CrawlRenderMode

        defn = SETTINGS_MAP["crawl_render_mode"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.choices == tuple(m.value for m in CrawlRenderMode)

    def test_crawl_render_mode_is_writable_for_programmatic_surfaces(self):
        from lilbee.config_meta import WRITABLE_CONFIG_FIELDS

        # The TUI checkbox persists the choice via apply_settings_update, so the
        # field must be writable through the HTTP / MCP / programmatic contract.
        assert "crawl_render_mode" in WRITABLE_CONFIG_FIELDS

    def test_browser_memory_levers_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP, get_default

        recycle = SETTINGS_MAP["crawl_browser_recycle_pages"]
        assert recycle.writable is True
        assert recycle.type is int
        assert get_default("crawl_browser_recycle_pages") == 50

        extra = SETTINGS_MAP["crawl_browser_extra_args"]
        assert extra.writable is True
        assert extra.type is list
        assert get_default("crawl_browser_extra_args") == [
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ]


class TestCrawlRenderModeConfig:
    def test_default_is_http(self):
        from lilbee.core.config.enums import CrawlRenderMode
        from lilbee.core.config.model import Config

        assert Config().crawl_render_mode is CrawlRenderMode.HTTP

    def test_env_var_overrides_to_browser(self, monkeypatch):
        from lilbee.core.config.enums import CrawlRenderMode
        from lilbee.core.config.model import Config

        monkeypatch.setenv("LILBEE_CRAWL_RENDER_MODE", "browser")
        assert Config().crawl_render_mode is CrawlRenderMode.BROWSER

    def test_invalid_value_is_rejected(self, monkeypatch):
        import pytest
        from pydantic import ValidationError

        from lilbee.core.config.model import Config

        monkeypatch.setenv("LILBEE_CRAWL_RENDER_MODE", "bogus")
        with pytest.raises(ValidationError):
            Config()

    def test_browser_memory_lever_defaults(self):
        from lilbee.core.config.model import Config

        c = Config()
        assert c.crawl_browser_recycle_pages == 50
        assert c.crawl_browser_extra_args == ["--disable-dev-shm-usage", "--disable-gpu"]


class TestOverlayPersistedSettings:
    def test_empty_string_value_is_skipped(self, tmp_path):
        """Legacy persisted empty strings (None written as "") skip overlay
        instead of corrupting the in-memory config or spamming warnings."""
        from lilbee.core.config import cfg

        original = cfg.chat_model
        try:
            (tmp_path / "config.toml").write_text('chat_model = ""\n')
            settings.overlay_persisted_settings(tmp_path)
            assert cfg.chat_model == original
        finally:
            cfg.chat_model = original
