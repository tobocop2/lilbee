"""Tests for persistent settings (config.toml)."""

from unittest import mock

import pytest

from lilbee.app.settings_map import SETTINGS_MAP, get_default
from lilbee.config_meta import WRITABLE_CONFIG_FIELDS
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

    def test_scalars_keep_their_toml_types(self, tmp_path):
        """bb-s9xc: booleans and numbers must not be written as quoted strings."""
        settings.save(tmp_path, {"chat_compaction": True, "chat_n_ctx_target": 2560})

        written = (tmp_path / "config.toml").read_text()
        assert "chat_compaction = true" in written
        assert "chat_n_ctx_target = 2560" in written
        assert '"True"' not in written
        assert '"2560"' not in written

    def test_a_load_save_round_trip_does_not_stringify(self, tmp_path):
        """The drift path: load then save used to quote every value it had read."""
        (tmp_path / "config.toml").write_text("chat_compaction = true\nchat_n_ctx_target = 2560\n")

        settings.save(tmp_path, settings.load(tmp_path))

        assert settings.load(tmp_path) == {"chat_compaction": True, "chat_n_ctx_target": 2560}

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


class TestMutateValue:
    def test_reads_persisted_value_inside_the_lock(self, tmp_path):
        settings.set_value(tmp_path, "linked_roots", {"a": "/x"})

        seen = {}

        def _fn(current):
            seen["current"] = current
            return {**(current or {}), "b": "/y"}, "done"

        result = settings.mutate_value(tmp_path, "linked_roots", _fn)
        assert seen["current"] == {"a": "/x"}  # the persisted value, not None
        assert result == "done"
        assert settings.load(tmp_path)["linked_roots"] == {"a": "/x", "b": "/y"}

    def test_passes_none_when_key_absent(self, tmp_path):
        captured = {}

        def _fn(current):
            captured["current"] = current
            return {"only": "/z"}, None

        settings.mutate_value(tmp_path, "linked_roots", _fn)
        assert captured["current"] is None
        assert settings.load(tmp_path)["linked_roots"] == {"only": "/z"}

    def test_preserves_sibling_keys(self, tmp_path):
        settings.set_value(tmp_path, "chat_model", "keep-me")
        settings.mutate_value(tmp_path, "linked_roots", lambda cur: ({"a": "/x"}, None))
        loaded = settings.load(tmp_path)
        assert loaded["chat_model"] == "keep-me"
        assert loaded["linked_roots"] == {"a": "/x"}


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

    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("escape", "before\x1bafter"),
            ("nul", "before\x00after"),
            ("vertical_tab", "before\x0bafter"),
            ("bell", "before\x07after"),
            ("delete", "before\x7fafter"),
            ("every_c0", "".join(chr(c) for c in range(0x20))),
        ],
    )
    def test_control_characters_round_trip(self, tmp_path, label, value):
        """TOML forbids raw control characters; writing one used to break the whole file."""
        settings.set_value(tmp_path, label, value)
        assert settings.get(tmp_path, label) == value

    def test_one_control_character_does_not_discard_the_rest_of_the_config(self, tmp_path):
        """A parse failure makes the reader drop every setting, not just the bad key."""
        settings.set_value(tmp_path, "model", "qwen3:8b")
        settings.set_value(tmp_path, "reranker_prompt", "rank\x1bthese")
        assert settings.load(tmp_path) == {"model": "qwen3:8b", "reranker_prompt": "rank\x1bthese"}

    @pytest.mark.parametrize(
        "value",
        ['say "hi"', r"C:\path", "a\nb", "a\tb", "normal", "", "a\x1bb", "a\x00b", "a\x7fb"],
    )
    def test_a_value_survives_the_round_trip_verbatim(self, tmp_path, value):
        """Escaping is only correct if the reader gives the string back unchanged."""
        settings.set_value(tmp_path, "reranker_prompt", value)
        assert settings.load(tmp_path)["reranker_prompt"] == value

    def test_a_list_value_round_trips_as_a_list(self, tmp_path):
        """The hand-rolled emitter stringified anything non-scalar, so a list
        was persisted as the quoted repr "['a', 'b']" and read back as text."""
        settings.set_value(tmp_path, "exclude", ["a", "b"])
        assert settings.load(tmp_path)["exclude"] == ["a", "b"]

    def test_a_none_value_is_dropped_rather_than_written_as_text(self, tmp_path):
        """It used to land as the string "None", which then read back as a
        truthy setting rather than an absent one."""
        settings.set_value(tmp_path, "model", "qwen3:8b")
        settings.set_value(tmp_path, "reranker_prompt", None)
        assert settings.load(tmp_path) == {"model": "qwen3:8b"}


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

    def test_flash_attention_is_load_affecting(self):
        # flash_attention bakes into the llama-server argv, so it must reload the
        # engine and gate cross-process sharing (it feeds the engine pin signature).
        from lilbee.core.config.keys import LOAD_AFFECTING_KEYS

        assert "flash_attention" in LOAD_AFFECTING_KEYS

    def test_reranker_fields_in_settings_map(self):

        assert "reranker_type" in SETTINGS_MAP
        assert "reranker_prompt" in SETTINGS_MAP
        assert SETTINGS_MAP["reranker_type"].choices == ("auto", "cross_encoder", "llm")

    def test_neighbor_expansion_in_settings_map(self):

        defn = SETTINGS_MAP["neighbor_expansion"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.group == "Retrieval"
        assert get_default("neighbor_expansion") == 0

    def test_fusion_knobs_in_settings_map(self):
        """The four adaptive-fusion / structural-filter knobs (which gate the
        on-by-default fusion behavior) are on the settings surface with their
        shipped defaults, so a dropped or typo'd entry fails CI."""

        assert get_default("lexical_fusion_weight") == 1.0
        assert get_default("adaptive_fusion") is False
        assert get_default("adaptive_fusion_margin") == 0.15
        assert get_default("filter_structural_chunks") is False
        for key in (
            "lexical_fusion_weight",
            "adaptive_fusion",
            "adaptive_fusion_margin",
            "filter_structural_chunks",
        ):
            assert SETTINGS_MAP[key].writable is True, key
            assert SETTINGS_MAP[key].group == "Retrieval", key


class TestReplicaDefaults:
    """embed/vision replica counts default to 0 = auto (one per GPU at placement)."""

    def test_replicas_default_to_auto_zero(self):
        from lilbee.core.config import Config

        assert Config().embed_replicas == 0
        assert Config().vision_replicas == 0

    def test_replicas_accept_zero(self):
        from lilbee.core.config import Config

        assert Config(embed_replicas=0, vision_replicas=0).embed_replicas == 0

    def test_replicas_reject_negative(self):
        import pydantic
        import pytest

        from lilbee.core.config import Config

        with pytest.raises(pydantic.ValidationError):
            Config(embed_replicas=-1)


class TestTableExtractionSetting:
    """The table-extraction flag is writable, grouped with ingest, and reindex-marked."""

    def test_table_extraction_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP, get_default

        defn = SETTINGS_MAP["table_extraction"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is bool
        assert defn.group == "Ingest"
        assert get_default("table_extraction") is False

    def test_table_extraction_requires_reindex(self):
        from lilbee.config_meta import REINDEX_FIELDS, WRITABLE_CONFIG_FIELDS

        assert "table_extraction" in WRITABLE_CONFIG_FIELDS
        assert "table_extraction" in REINDEX_FIELDS


class TestLayoutDetectionSetting:
    """The layout-detection flag is writable, grouped with ingest, and reindex-marked."""

    def test_layout_detection_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP, get_default

        defn = SETTINGS_MAP["layout_detection"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is bool
        assert defn.group == "Ingest"
        assert get_default("layout_detection") is True

    def test_layout_detection_requires_reindex(self):
        from lilbee.config_meta import REINDEX_FIELDS, WRITABLE_CONFIG_FIELDS

        assert "layout_detection" in WRITABLE_CONFIG_FIELDS
        assert "layout_detection" in REINDEX_FIELDS


class TestTableModelSetting:
    """The table-model choice is writable, grouped with ingest, and reindex-marked."""

    def test_table_model_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP, get_default

        defn = SETTINGS_MAP["table_model"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is str
        assert defn.group == "Ingest"
        assert defn.choices == (
            "disabled",
            "tatr",
            "slanet_auto",
            "slanet_plus",
            "slanet_wired",
            "slanet_wireless",
        )
        assert get_default("table_model") == "slanet_auto"

    def test_table_model_requires_reindex(self):
        from lilbee.config_meta import REINDEX_FIELDS, WRITABLE_CONFIG_FIELDS

        assert "table_model" in WRITABLE_CONFIG_FIELDS
        assert "table_model" in REINDEX_FIELDS


class TestMemoryTuningSettingsMap:
    """The dynamic-ctx tuning knobs are surfaced in the TUI settings map."""

    def test_num_ctx_max_in_settings_map(self):

        defn = SETTINGS_MAP["num_ctx_max"]
        assert defn.writable is True
        assert defn.nullable is True  # None = use model training_ctx as ceiling
        assert defn.group == "Generation"
        assert get_default("num_ctx_max") is None

    def test_chat_n_ctx_target_in_settings_map(self):

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

        defn = SETTINGS_MAP["flash_attention"]
        assert defn.writable is True
        assert defn.nullable is True  # tri-state: None=auto
        assert defn.type is bool
        assert get_default("flash_attention") is None

    def test_kv_cache_type_in_settings_map(self):
        from lilbee.core.config.enums import KvCacheType

        defn = SETTINGS_MAP["kv_cache_type"]
        assert defn.writable is True
        assert defn.choices == tuple(t.value for t in KvCacheType)

    def test_n_gpu_layers_in_settings_map(self):

        defn = SETTINGS_MAP["n_gpu_layers"]
        assert defn.writable is True
        assert defn.nullable is True  # None = auto/all
        assert get_default("n_gpu_layers") is None

    def test_vision_ocr_max_tokens_in_settings_map(self):

        defn = SETTINGS_MAP["vision_ocr_max_tokens"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is int
        assert defn.group == "Ingest"
        assert get_default("vision_ocr_max_tokens") == 4096

    def test_vision_ocr_concurrency_in_settings_map(self):

        defn = SETTINGS_MAP["vision_ocr_concurrency"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is int
        assert defn.group == "Ingest"
        assert get_default("vision_ocr_concurrency") == 4

    def test_crawl_render_mode_in_settings_map(self):
        from lilbee.core.config.enums import CrawlRenderMode

        defn = SETTINGS_MAP["crawl_render_mode"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.choices == tuple(m.value for m in CrawlRenderMode)

    def test_crawl_render_mode_is_writable_for_programmatic_surfaces(self):

        # The TUI checkbox persists the choice via apply_settings_update, so the
        # field must be writable through the HTTP / MCP / programmatic contract.
        assert "crawl_render_mode" in WRITABLE_CONFIG_FIELDS

    def test_browser_memory_levers_in_settings_map(self):

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
    def test_empty_string_value_is_skipped(self, tmp_path, monkeypatch):
        """Legacy persisted empty strings (None written as "") skip overlay
        instead of corrupting the in-memory config or spamming warnings."""
        from lilbee.core.config import cfg

        monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
        original = cfg.chat_model
        try:
            (tmp_path / "config.toml").write_text('chat_model = ""\n')
            settings.overlay_persisted_settings(tmp_path)
            assert cfg.chat_model == original
        finally:
            cfg.chat_model = original

    def test_env_var_wins_over_config_toml(self, tmp_path, monkeypatch):
        """An explicit LILBEE_<FIELD> env var overrides config.toml, as documented."""
        from lilbee.core.config import cfg

        original = cfg.vision_replicas
        try:
            monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
            cfg.vision_replicas = 4  # value as loaded from LILBEE_VISION_REPLICAS
            monkeypatch.setenv("LILBEE_VISION_REPLICAS", "4")
            (tmp_path / "config.toml").write_text("vision_replicas = 2\n")
            settings.overlay_persisted_settings(tmp_path)
            assert cfg.vision_replicas == 4
        finally:
            cfg.vision_replicas = original

    def test_empty_env_var_does_not_suppress_config_toml(self, tmp_path, monkeypatch):
        """An empty LILBEE_<FIELD> env var is treated as unset; config.toml wins."""
        from lilbee.core.config import cfg

        original = cfg.vision_replicas
        try:
            monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
            monkeypatch.setenv("LILBEE_VISION_REPLICAS", "")
            cfg.vision_replicas = 1
            (tmp_path / "config.toml").write_text("vision_replicas = 3\n")
            settings.overlay_persisted_settings(tmp_path)
            assert cfg.vision_replicas == 3
        finally:
            cfg.vision_replicas = original

    def test_config_toml_applies_when_env_absent(self, tmp_path, monkeypatch):
        """Without the env var, config.toml is still overlaid onto cfg."""
        from lilbee.core.config import cfg

        original = cfg.vision_replicas
        try:
            monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
            monkeypatch.delenv("LILBEE_VISION_REPLICAS", raising=False)
            cfg.vision_replicas = 1
            (tmp_path / "config.toml").write_text("vision_replicas = 3\n")
            settings.overlay_persisted_settings(tmp_path)
            assert cfg.vision_replicas == 3
        finally:
            cfg.vision_replicas = original

    def test_skip_toml_config_makes_overlay_noop(self, tmp_path, monkeypatch):
        """LILBEE_SKIP_TOML_CONFIG=1 disables the overlay path too, so the escape
        hatch is honored consistently with the pydantic-settings source (the CLI
        and MCP overlay must not re-read config.toml behind the skip flag)."""
        from lilbee.core.config import cfg

        original = cfg.vision_replicas
        try:
            monkeypatch.setenv("LILBEE_SKIP_TOML_CONFIG", "1")
            cfg.vision_replicas = 1
            (tmp_path / "config.toml").write_text("vision_replicas = 3\n")
            settings.overlay_persisted_settings(tmp_path)
            assert cfg.vision_replicas == 1  # config.toml ignored while skipping
        finally:
            cfg.vision_replicas = original


class TestAutoSyncConfig:
    def test_auto_sync_defaults_true(self):
        from lilbee.core.config import Config

        assert Config().auto_sync is True

    def test_auto_sync_is_writable(self):

        assert "auto_sync" in WRITABLE_CONFIG_FIELDS

    def test_auto_sync_in_settings_map(self):

        assert "auto_sync" in SETTINGS_MAP


class TestListSettingRegexMarker:
    def test_only_regex_list_validates_as_regex(self):

        assert SETTINGS_MAP["crawl_exclude_patterns"].validate_regex is True
        # Chromium flag list must not be regex-validated.
        assert SETTINGS_MAP["crawl_browser_extra_args"].validate_regex is False


class TestUtf8RoundTrip:
    """save() writes UTF-8 and load() reads it back correctly (finding #2)."""

    def test_non_ascii_value_round_trips(self, tmp_path) -> None:
        settings.save(tmp_path, {"model": "qwen3-中文"})
        result = settings.load(tmp_path)
        assert result == {"model": "qwen3-中文"}

    def test_file_is_utf8_encoded(self, tmp_path) -> None:
        settings.save(tmp_path, {"key": "éàü"})
        raw = (tmp_path / "config.toml").read_bytes()
        decoded = raw.decode("utf-8")
        assert "key" in decoded

    def test_overwriting_an_existing_file_round_trips_unicode(self, tmp_path) -> None:
        """The atomic replace must not lose the UTF-8 encoding on a rewrite."""
        settings.save(tmp_path, {"key": "value"})
        settings.save(tmp_path, {"key": "value", "unicode": "é"})
        result = settings.load(tmp_path)
        assert result["unicode"] == "é"


class TestTitleSearchSettings:
    """The title-arm knobs are exposed on every settings surface."""

    def test_title_search_in_settings_map(self):

        defn = SETTINGS_MAP["title_search"]
        assert defn.writable is True
        assert defn.type is bool
        assert defn.group == "Retrieval"
        assert get_default("title_search") is False

    def test_title_search_weight_in_settings_map(self):

        defn = SETTINGS_MAP["title_search_weight"]
        assert defn.writable is True
        assert defn.type is float
        assert defn.group == "Retrieval"
        assert get_default("title_search_weight") == 0.5

    def test_title_search_fields_are_writable_for_programmatic_surfaces(self):

        assert "title_search" in WRITABLE_CONFIG_FIELDS
        assert "title_search_weight" in WRITABLE_CONFIG_FIELDS


class TestConcurrentConfigWrites:
    """A server, a CLI run, and an MCP process share one data root."""

    def test_a_second_process_does_not_drop_the_first_processes_key(self, tmp_path):
        """Cross-process read-modify-write must serialize, not interleave.

        A threading.Lock only covers one interpreter, so two processes could
        both load the same snapshot and each save it back without the other's
        key. This drives real subprocesses, which a thread test cannot.
        """
        import subprocess
        import sys
        import textwrap

        # Each process holds the read-modify-write open for a beat. Under the
        # lock they queue up and every key survives; without it they all load
        # the same snapshot and the last writer wins.
        script = textwrap.dedent(
            f"""
            import sys, time
            from pathlib import Path
            from lilbee.core import settings

            real_save = settings.save
            def slow_save(root, values):
                time.sleep(0.3)
                real_save(root, values)
            settings.save = slow_save

            settings.set_value(Path({str(tmp_path)!r}), sys.argv[1], sys.argv[1])
            """
        )
        procs = [subprocess.Popen([sys.executable, "-c", script, f"key{i}"]) for i in range(4)]
        for proc in procs:
            assert proc.wait(timeout=120) == 0

        result = settings.load(tmp_path)
        assert sorted(result) == [f"key{i}" for i in range(4)]

    def test_a_stale_lock_does_not_block_the_write(self, tmp_path, monkeypatch, caplog):
        """Losing an update to an abandoned lock file is worse than the race."""
        from filelock import FileLock

        from lilbee.core import settings as settings_mod

        monkeypatch.setattr(settings_mod, "_CONFIG_LOCK_TIMEOUT_S", 0.01)
        held = FileLock(str(tmp_path / "config.toml") + ".lock")
        held.acquire()
        try:
            with caplog.at_level("WARNING"):
                settings.set_value(tmp_path, "key", "value")
        finally:
            held.release()
        assert settings.get(tmp_path, "key") == "value"
        assert "Timed out waiting" in caplog.text
