"""Tests for persistent settings (config.toml)."""

from dataclasses import fields as dataclass_fields
from unittest import mock

import pytest

from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.config_meta import WRITABLE_CONFIG_FIELDS
from lilbee.core import settings
from lilbee.core.config.resolve import builtin_value
from lilbee.core.project_state import ProjectState


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

    def test_every_enum_typed_setting_carries_its_enums_values(self):
        """An enum-typed field cannot reach a client without its value set."""
        import enum
        from typing import get_args

        from lilbee.core.config import Config

        checked = 0
        for key, definition in SETTINGS_MAP.items():
            annotation = Config.model_fields[key].annotation
            for candidate in get_args(annotation) or (annotation,):
                if isinstance(candidate, type) and issubclass(candidate, enum.Enum):
                    assert definition.choices == tuple(str(m.value) for m in candidate), key
                    checked += 1
        assert checked >= 11

    def test_a_collection_field_gets_no_value_set(self):
        """A picker is only for a scalar, so a list field stays free text."""
        from lilbee.core.config.schema import field_value_set

        assert field_value_set("ignore_dirs") is None
        assert SETTINGS_MAP["ocr_language"].choices is None

    def test_reranker_fields_in_settings_map(self):

        assert "reranker_type" in SETTINGS_MAP
        assert "reranker_prompt" in SETTINGS_MAP
        assert SETTINGS_MAP["reranker_type"].choices == ("auto", "cross_encoder", "llm")

    def test_neighbor_expansion_in_settings_map(self):

        defn = SETTINGS_MAP["neighbor_expansion"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.group == "Retrieval"
        assert builtin_value("neighbor_expansion") == 0

    def test_fusion_knobs_in_settings_map(self):
        """The four adaptive-fusion / structural-filter knobs (which gate the
        on-by-default fusion behavior) are on the settings surface with their
        shipped defaults, so a dropped or typo'd entry fails CI."""

        assert builtin_value("lexical_fusion_weight") == 1.0
        assert builtin_value("adaptive_fusion") is False
        assert builtin_value("adaptive_fusion_margin") == 0.15
        assert builtin_value("filter_structural_chunks") is False
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
        from lilbee.app.settings_map import SETTINGS_MAP

        defn = SETTINGS_MAP["table_extraction"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is bool
        assert defn.group == "Ingest"
        assert builtin_value("table_extraction") is False

    def test_table_extraction_requires_reindex(self):
        from lilbee.config_meta import REINDEX_FIELDS, WRITABLE_CONFIG_FIELDS

        assert "table_extraction" in WRITABLE_CONFIG_FIELDS
        assert "table_extraction" in REINDEX_FIELDS


class TestLayoutDetectionSetting:
    """The layout-detection flag is writable, grouped with ingest, and reindex-marked."""

    def test_layout_detection_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP

        defn = SETTINGS_MAP["layout_detection"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is bool
        assert defn.group == "Ingest"
        assert builtin_value("layout_detection") is False

    def test_layout_detection_requires_reindex(self):
        from lilbee.config_meta import REINDEX_FIELDS, WRITABLE_CONFIG_FIELDS

        assert "layout_detection" in WRITABLE_CONFIG_FIELDS
        assert "layout_detection" in REINDEX_FIELDS


class TestTableModelSetting:
    """The table-model choice is writable, grouped with ingest, and reindex-marked."""

    def test_table_model_in_settings_map(self):
        from lilbee.app.settings_map import SETTINGS_MAP

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
        assert builtin_value("table_model") == "slanet_auto"

    def test_table_model_requires_reindex(self):
        from lilbee.config_meta import REINDEX_FIELDS, WRITABLE_CONFIG_FIELDS

        assert "table_model" in WRITABLE_CONFIG_FIELDS
        assert "table_model" in REINDEX_FIELDS

    def test_table_model_change_flags_no_reindex_while_layout_detection_is_off(self, monkeypatch):
        """xberg reads table_model only inside layout detection, so the change is inert."""
        from lilbee.app import settings as appset

        monkeypatch.setattr(appset.cfg, "layout_detection", False)
        monkeypatch.setattr(appset.persistent_settings, "update_values", lambda *_a, **_k: None)
        result = appset.apply_settings_update({"table_model": "tatr"})
        assert result.updated == ["table_model"]
        assert result.reindex_required is False

    def test_table_model_change_flags_a_reindex_once_layout_detection_is_on(self, monkeypatch):
        from lilbee.app import settings as appset

        monkeypatch.setattr(appset.persistent_settings, "update_values", lambda *_a, **_k: None)
        monkeypatch.setattr(appset.cfg, "layout_detection", True)
        assert appset.apply_settings_update({"table_model": "tatr"}).reindex_required is True
        monkeypatch.setattr(appset.cfg, "layout_detection", False)
        both = appset.apply_settings_update({"table_model": "disabled", "layout_detection": True})
        assert both.reindex_required is True


class TestOcrPageSelectionSettings:
    """The PDF OCR page-selection settings are writable and survive a config.toml reload."""

    def test_settings_map_entries(self):
        assert SETTINGS_MAP["ocr_strategy"].choices == ("auto", "scanned_pages")
        assert SETTINGS_MAP["ocr_scan_confidence"].type is float
        assert SETTINGS_MAP["force_ocr_pages"].type is list
        for key in ("ocr_strategy", "ocr_scan_confidence", "force_ocr_pages"):
            assert SETTINGS_MAP[key].group == "Ingest"
            assert key in WRITABLE_CONFIG_FIELDS
        assert builtin_value("ocr_strategy") == "auto"
        assert builtin_value("ocr_scan_confidence") == 0.7
        assert builtin_value("force_ocr_pages") == []

    def test_update_persists_and_reloads(self, tmp_path, monkeypatch):
        from lilbee.app import settings as appset

        monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
        monkeypatch.setattr(appset.cfg, "data_root", tmp_path)
        monkeypatch.setattr(appset.cfg, "force_ocr_pages", [])
        monkeypatch.setattr(appset.cfg, "ocr_strategy", "auto")
        monkeypatch.setattr(appset.cfg, "ocr_scan_confidence", 0.7)
        appset.apply_settings_update(
            {
                "force_ocr_pages": "3,1",
                "ocr_strategy": "scanned_pages",
                "ocr_scan_confidence": 0.5,
            }
        )
        assert appset.cfg.force_ocr_pages == [1, 3]
        assert settings.load(tmp_path)["force_ocr_pages"] == "1\n3"

        appset.cfg.force_ocr_pages = []
        appset.cfg.ocr_strategy = "auto"
        appset.cfg.ocr_scan_confidence = 0.7
        settings.overlay_persisted_settings(tmp_path)
        assert appset.cfg.force_ocr_pages == [1, 3]
        assert appset.cfg.ocr_strategy == "scanned_pages"
        assert appset.cfg.ocr_scan_confidence == 0.5

    def test_an_invalid_page_rolls_the_update_back(self, monkeypatch):
        from lilbee.app import settings as appset

        monkeypatch.setattr(appset.cfg, "force_ocr_pages", [2])
        monkeypatch.setattr(appset.persistent_settings, "update_values", lambda *_a, **_k: None)
        with pytest.raises(ValueError, match="page numbers start at 1"):
            appset.apply_settings_update({"force_ocr_pages": [0]})
        assert appset.cfg.force_ocr_pages == [2]


class TestMemoryTuningSettingsMap:
    """The dynamic-ctx tuning knobs are surfaced in the TUI settings map."""

    def test_num_ctx_max_in_settings_map(self):

        defn = SETTINGS_MAP["num_ctx_max"]
        assert defn.writable is True
        assert defn.nullable is True  # None = use model training_ctx as ceiling
        assert defn.group == "Generation"
        assert builtin_value("num_ctx_max") is None

    def test_chat_n_ctx_target_in_settings_map(self):

        defn = SETTINGS_MAP["chat_n_ctx_target"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.group == "Generation"
        with mock.patch(
            "lilbee.core.system._read_total_memory_bytes",
            return_value=8 * 1024**3,
        ):
            assert builtin_value("chat_n_ctx_target") == 8192

    def test_flash_attention_in_settings_map(self):

        defn = SETTINGS_MAP["flash_attention"]
        assert defn.writable is True
        assert defn.nullable is True  # tri-state: None=auto
        assert defn.type is bool
        assert builtin_value("flash_attention") is None

    def test_kv_cache_type_in_settings_map(self):
        from lilbee.core.config.enums import KvCacheType

        defn = SETTINGS_MAP["kv_cache_type"]
        assert defn.writable is True
        assert defn.choices == tuple(t.value for t in KvCacheType)

    def test_n_gpu_layers_in_settings_map(self):

        defn = SETTINGS_MAP["n_gpu_layers"]
        assert defn.writable is True
        assert defn.nullable is True  # None = auto/all
        assert builtin_value("n_gpu_layers") is None

    def test_vision_ocr_max_tokens_in_settings_map(self):

        defn = SETTINGS_MAP["vision_ocr_max_tokens"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is int
        assert defn.group == "Ingest"
        assert builtin_value("vision_ocr_max_tokens") == 4096

    def test_vision_ocr_concurrency_in_settings_map(self):

        defn = SETTINGS_MAP["vision_ocr_concurrency"]
        assert defn.writable is True
        assert defn.nullable is False
        assert defn.type is int
        assert defn.group == "Ingest"
        assert builtin_value("vision_ocr_concurrency") == 4

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
        assert builtin_value("crawl_browser_recycle_pages") == 50

        extra = SETTINGS_MAP["crawl_browser_extra_args"]
        assert extra.writable is True
        assert extra.type is list
        assert builtin_value("crawl_browser_extra_args") == [
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

    def test_empty_vision_model_env_keeps_config_toml_from_restoring_it(
        self, tmp_path, monkeypatch
    ):
        """An empty LILBEE_VISION_MODEL clears the role; config.toml's other keys still apply."""
        from lilbee.core.config import cfg

        original_vision, original_top_k = cfg.vision_model, cfg.top_k
        try:
            monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
            monkeypatch.setenv("LILBEE_VISION_MODEL", "")
            cfg.vision_model = ""
            cfg.top_k = 5
            (tmp_path / "config.toml").write_text(
                'vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"\ntop_k = 9\n',
                encoding="utf-8",
            )
            settings.overlay_persisted_settings(tmp_path)
            assert cfg.vision_model == ""
            assert cfg.top_k == 9
        finally:
            cfg.vision_model, cfg.top_k = original_vision, original_top_k

    def test_empty_persisted_vision_model_clears_the_ambient_one(self, tmp_path, monkeypatch):
        """A vision_model cleared into config.toml stays cleared; an empty chunk_size is unset."""
        from lilbee.core.config import cfg

        originals = cfg.vision_model, cfg.chunk_size, cfg.top_k
        try:
            monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
            monkeypatch.delenv("LILBEE_VISION_MODEL", raising=False)
            monkeypatch.delenv("LILBEE_CHUNK_SIZE", raising=False)
            cfg.vision_model = "org/Ambient-Vision-GGUF/ambient-Q4_K_M.gguf"
            (tmp_path / "config.toml").write_text(
                'vision_model = ""\nchunk_size = ""\ntop_k = 9\n'
                "[profile.values]\nchunk_size = 900\n",
                encoding="utf-8",
            )
            settings.overlay_persisted_settings(tmp_path)
            assert cfg.vision_model == ""
            assert cfg.chunk_size == 900
            assert cfg.top_k == 9
        finally:
            cfg.vision_model, cfg.chunk_size, cfg.top_k = originals

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


def test_the_model_roles_that_can_be_off_are_the_clearable_ones():
    from lilbee.config_meta import MODEL_ROLE_FIELDS
    from lilbee.core.config.model import CLEARABLE_MODEL_FIELDS

    nullable_roles = {key for key in MODEL_ROLE_FIELDS if SETTINGS_MAP[key].nullable}
    assert nullable_roles == CLEARABLE_MODEL_FIELDS == {"vision_model", "reranker_model"}


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
        assert builtin_value("title_search") is False

    def test_title_search_weight_in_settings_map(self):

        defn = SETTINGS_MAP["title_search_weight"]
        assert defn.writable is True
        assert defn.type is float
        assert defn.group == "Retrieval"
        assert builtin_value("title_search_weight") == 0.5

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


class TestCredentialsAreMaskable:
    """Every credential must reach the settings surface marked as one.

    ``llm_api_key`` was writable but absent from SETTINGS_MAP, so the remote
    provider's key could not be set from the TUI at all, and nothing marked it
    for masking.
    """

    @staticmethod
    def _write_only_fields() -> set[str]:
        from lilbee.core.config.model import Config

        found = set()
        for name, info in Config.model_fields.items():
            extra = info.json_schema_extra or {}
            if isinstance(extra, dict) and extra.get("write_only"):
                found.add(name)
        return found

    def test_every_write_only_field_is_editable(self):
        missing = self._write_only_fields() - set(SETTINGS_MAP)
        assert not missing, f"credentials absent from the settings surface: {sorted(missing)}"

    def test_every_write_only_field_is_marked_secret(self):
        unmarked = {f for f in self._write_only_fields() if not SETTINGS_MAP[f].secret}
        assert not unmarked, f"credentials that would render in the clear: {sorted(unmarked)}"

    def test_no_credential_is_exposed_by_the_public_config_api(self):
        from lilbee.config_meta import PUBLIC_CONFIG_FIELDS

        leaked = self._write_only_fields() & set(PUBLIC_CONFIG_FIELDS)
        assert not leaked, f"credentials returned by GET /api/config: {sorted(leaked)}"


class TestEmbedReindexRequired:
    """The setter's answer comes from the same verdict search refuses on."""

    _OTHER = "acme/other-GGUF/other.gguf"

    @pytest.fixture()
    def real_store(self, tmp_path, monkeypatch):
        from lilbee.app import settings as appset
        from lilbee.app.services import set_services
        from lilbee.data.store import Store

        monkeypatch.setattr(appset.cfg, "lancedb_dir", tmp_path / "lancedb")
        monkeypatch.setattr(appset.cfg, "embedding_model", "acme/built-GGUF/built.gguf")
        monkeypatch.setattr(appset.cfg, "embedding_dim", 4)
        store = Store(appset.cfg)
        set_services(mock.MagicMock(store=store))
        yield store

    @staticmethod
    def _index_one_chunk(store):
        store.add_chunks(
            [
                {
                    "source": "doc.md",
                    "content_type": "text",
                    "chunk_type": "raw",
                    "page_start": 0,
                    "page_end": 0,
                    "line_start": 0,
                    "line_end": 0,
                    "chunk": "text",
                    "chunk_index": 0,
                    "vector": [0.5, 0.5, 0.5, 0.5],
                }
            ]
        )

    def test_no_index_needs_no_reindex(self, real_store):
        from lilbee.app import settings as appset

        assert appset._embed_reindex_required() is False

    def test_a_name_change_requires_reindex(self, real_store, monkeypatch):
        from lilbee.app import settings as appset

        self._index_one_chunk(real_store)
        monkeypatch.setattr(appset.cfg, "embedding_model", self._OTHER)
        assert appset._embed_reindex_required() is True

    def test_a_dimension_change_requires_reindex(self, real_store, monkeypatch):
        """The persisted width is compared against the new model's, so a same-name
        embedder of another width is not mistaken for the one that built the index."""
        from lilbee.app import settings as appset

        self._index_one_chunk(real_store)
        monkeypatch.setattr(appset.cfg, "embedding_dim", 8)
        assert appset._embed_reindex_required() is True

    def test_switching_back_to_the_building_model_needs_no_reindex(self, real_store, monkeypatch):
        from lilbee.app import settings as appset

        self._index_one_chunk(real_store)
        monkeypatch.setattr(appset.cfg, "embedding_model", self._OTHER)
        assert appset._embed_reindex_required() is True
        monkeypatch.setattr(appset.cfg, "embedding_model", "acme/built-GGUF/built.gguf")
        assert appset._embed_reindex_required() is False


class TestResolverIsTheOnlyWriter:
    """Every settings writer leaves cfg equal to what a fresh Config() resolves."""

    @staticmethod
    def _write_config(text: str):
        from lilbee.core.config import cfg

        cfg.data_root.mkdir(parents=True, exist_ok=True)
        (cfg.data_root / "config.toml").write_text(text, encoding="utf-8")
        return cfg.data_root

    def test_live_cfg_matches_fresh_config_after_updates_and_nulls(self, monkeypatch):
        from lilbee.app import settings as appset
        from lilbee.core.config import Config, cfg
        from lilbee.core.config.resolve import ROOT_DERIVED_FIELDS
        from lilbee.providers.roles import MODEL_ROLE_FIELDS

        root = self._write_config(
            'top_k = 7\n[profile]\nname = "x"\n[profile.values]\n'
            "rerank_min_score = 0.7\nchunk_overlap = 50\n"
        )
        monkeypatch.setenv("LILBEE_MAX_TOKENS", "2048")
        settings.overlay_persisted_settings(root)
        appset.apply_settings_update(
            {"top_k": 9, "rerank_min_score": 0.3, "max_tokens": 1000, "seed": 5}
        )
        appset.apply_settings_update({"rerank_min_score": None, "seed": None})

        fresh = Config()
        keys = sorted((set(WRITABLE_CONFIG_FIELDS) | MODEL_ROLE_FIELDS) - ROOT_DERIVED_FIELDS)
        assert len(keys) > 100
        diverged = {k: (getattr(cfg, k), getattr(fresh, k)) for k in keys}
        diverged = {k: pair for k, pair in diverged.items() if pair[0] != pair[1]}
        assert diverged == {}
        assert (cfg.top_k, cfg.rerank_min_score, cfg.max_tokens, cfg.seed) == (9, 0.7, 2048, None)
        assert cfg.chunk_overlap == 50

    def test_update_under_env_keeps_env_value_in_cfg(self, monkeypatch):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        monkeypatch.setenv("LILBEE_TOP_K", "9")
        cfg.top_k = 9
        cfg.chunk_size = 512
        appset.apply_settings_update({"top_k": 7, "chunk_size": 900})
        assert cfg.top_k == 9
        assert cfg.chunk_size == 900
        assert settings.load(cfg.data_root)["top_k"] == 7

    @pytest.mark.parametrize("env_model", ["acme/a-GGUF/a.gguf", None])
    def test_embed_swap_keeps_the_width_of_the_live_model(self, monkeypatch, env_model):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        widths = {"acme/a-GGUF/a.gguf": 768, "acme/b-GGUF/b.gguf": 1024}
        if env_model is not None:
            monkeypatch.setenv("LILBEE_EMBEDDING_MODEL", env_model)
        cfg.embedding_model = "acme/a-GGUF/a.gguf"
        cfg.embedding_dim = 768
        monkeypatch.setattr(
            appset, "_embedder_dim_from_gguf", lambda ref, registry=None: widths[ref]
        )
        monkeypatch.setattr(appset, "_pin_legacy_store_meta", lambda: None)
        monkeypatch.setattr(appset, "_invalidate_caches", lambda keys: None)
        monkeypatch.setattr(appset, "_embed_reindex_required", lambda: False)
        appset.apply_settings_update({"embedding_model": "acme/b-GGUF/b.gguf"})
        expected_model = env_model or "acme/b-GGUF/b.gguf"
        assert (cfg.embedding_model, cfg.embedding_dim) == (expected_model, widths[expected_model])
        assert settings.load(cfg.data_root)["embedding_model"] == "acme/b-GGUF/b.gguf"

    def test_null_update_resolves_to_profile_value(self):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        self._write_config("[profile.values]\nrerank_min_score = 0.7\n")
        appset.apply_settings_update({"rerank_min_score": 0.3})
        assert cfg.rerank_min_score == 0.3
        appset.apply_settings_update({"rerank_min_score": None})
        assert cfg.rerank_min_score == 0.7
        assert "rerank_min_score" not in settings.load(cfg.data_root)

    def test_blank_string_update_clears_a_sampling_field_like_null(self):
        """REST and MCP forward raw JSON with no blank-stripping of their own;
        the field validator is what makes "" behave like an explicit null."""
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        # temperature carries no profile scope, so this [profile.values] entry
        # is dropped and the clear falls to the built-in default, same as null.
        self._write_config("[profile.values]\ntemperature = 0.7\n")
        appset.apply_settings_update({"temperature": 0.3})
        assert cfg.temperature == 0.3
        appset.apply_settings_update({"temperature": ""})
        assert cfg.temperature == 0.1
        assert "temperature" not in settings.load(cfg.data_root)

    def test_null_update_over_invalid_profile_value_warns_and_keeps_cfg_valid(self, caplog):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        self._write_config('[profile.values]\nrerank_min_score = "banana"\n')
        with caplog.at_level("WARNING", logger="lilbee.core.settings"):
            appset.apply_settings_update({"rerank_min_score": None})
        assert cfg.rerank_min_score is None
        assert any("rerank_min_score" in record.getMessage() for record in caplog.records)

    def test_overlay_resets_key_absent_from_new_root(self, tmp_path):
        from lilbee.core.config import cfg

        first = tmp_path / "first"
        first.mkdir()
        (first / "config.toml").write_text("top_k = 7\nchunk_size = 900\n", encoding="utf-8")
        second = tmp_path / "second"
        second.mkdir()
        (second / "config.toml").write_text("chunk_size = 700\n", encoding="utf-8")

        settings.overlay_persisted_settings(first)
        assert (cfg.top_k, cfg.chunk_size) == (7, 900)
        settings.overlay_persisted_settings(second)
        assert cfg.top_k == 12
        assert cfg.chunk_size == 700

    def test_overlay_keeps_the_root_derived_documents_dir(self, tmp_path):
        from lilbee.core.config import cfg

        root = tmp_path / "root"
        root.mkdir()
        (root / "config.toml").write_text("top_k = 7\n", encoding="utf-8")
        cfg.documents_dir = root / "documents"
        settings.overlay_persisted_settings(root)
        assert cfg.documents_dir == root / "documents"
        assert cfg.top_k == 7

    def test_profile_table_survives_update_and_delete(self):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        self._write_config('[profile]\nname = "scanned"\n[profile.values]\ntop_k = 4\n')
        appset.apply_settings_update({"chunk_size": 900})
        appset.apply_settings_update({"seed": 3})
        appset.apply_settings_update({"seed": None})
        stored = settings.load(cfg.data_root)
        assert stored["profile"] == {"name": "scanned", "values": {"top_k": 4}}
        assert stored["chunk_size"] == 900
        assert "seed" not in stored

    def test_profile_is_not_a_settable_key(self):
        from lilbee.app import settings as appset

        with pytest.raises(ValueError, match="Unknown or read-only setting: profile"):
            appset.apply_settings_update({"profile": {"name": "x"}})

    def test_no_config_field_holds_project_state(self):
        from lilbee.core.config import Config
        from lilbee.core.config.resolve import PROFILE_TABLE

        fields = set(Config.model_fields)
        state_keys = {field.name for field in dataclass_fields(ProjectState)}
        assert "top_k" in fields
        assert state_keys == {"analyzed_at", "tip_dismissed"}
        assert not state_keys & fields
        assert PROFILE_TABLE not in fields
        assert not any(name.startswith("profile") for name in fields)
        assert PROFILE_TABLE not in WRITABLE_CONFIG_FIELDS


class TestResetRemovesTheUserValue:
    """Reset deletes the key from config.toml and cfg takes the resolver's next source."""

    @staticmethod
    def _write_config(text: str):
        from lilbee.core.config import cfg

        cfg.data_root.mkdir(parents=True, exist_ok=True)
        (cfg.data_root / "config.toml").write_text(text, encoding="utf-8")
        return cfg.data_root

    def test_reset_leaves_cfg_equal_to_a_fresh_config(self, monkeypatch):
        from lilbee.app import settings as appset
        from lilbee.core.config import Config, cfg
        from lilbee.core.config.resolve import ROOT_DERIVED_FIELDS
        from lilbee.providers.roles import MODEL_ROLE_FIELDS

        root = self._write_config(
            "top_k = 7\nmax_distance = 0.3\nmax_tokens = 1000\nchunk_size = 900\n"
            "[profile.values]\nmax_distance = 0.7\n"
        )
        monkeypatch.setenv("LILBEE_MAX_TOKENS", "2048")
        settings.overlay_persisted_settings(root)
        appset.reset_settings(["top_k", "max_distance", "max_tokens", "seed"])

        fresh = Config()
        keys = sorted((set(WRITABLE_CONFIG_FIELDS) | MODEL_ROLE_FIELDS) - ROOT_DERIVED_FIELDS)
        assert len(keys) > 100
        diverged = {k: (getattr(cfg, k), getattr(fresh, k)) for k in keys}
        diverged = {k: pair for k, pair in diverged.items() if pair[0] != pair[1]}
        assert diverged == {}
        assert (cfg.top_k, cfg.max_distance, cfg.max_tokens) == (12, 0.7, 2048)
        assert settings.load(root) == {
            "chunk_size": 900,
            "profile": {"values": {"max_distance": 0.7}},
        }

    def test_reset_under_env_pin_keeps_env_and_removes_user_key(self, monkeypatch):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        root = self._write_config("top_k = 7\nchunk_size = 900\n")
        monkeypatch.setenv("LILBEE_TOP_K", "9")
        cfg.top_k = 9
        result = appset.reset_settings(["top_k"])
        assert result.updated == ["top_k"]
        assert cfg.top_k == 9
        assert settings.load(root) == {"chunk_size": 900}

    def test_reset_resolves_to_the_profile_value(self):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        root = self._write_config(
            "max_distance = 0.3\ntop_k = 7\n[profile.values]\nmax_distance = 0.7\n"
        )
        cfg.max_distance = 0.3
        cfg.top_k = 7
        appset.reset_settings(["max_distance"])
        assert cfg.max_distance == 0.7
        assert cfg.top_k == 7
        stored = settings.load(root)
        assert "max_distance" not in stored
        assert stored["top_k"] == 7
        assert stored["profile"] == {"values": {"max_distance": 0.7}}

    def test_reset_of_a_key_not_in_config_toml_writes_nothing(self):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        root = self._write_config("chunk_size = 900\n")
        path = root / "config.toml"
        before = path.stat().st_mtime_ns
        cfg.top_k = 99
        result = appset.reset_settings(["top_k"])
        assert result.updated == ["top_k"]
        assert cfg.top_k == 12
        assert path.stat().st_mtime_ns == before
        assert settings.load(root) == {"chunk_size": 900}

    def test_reset_with_no_config_file_creates_none(self):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        cfg.data_root.mkdir(parents=True, exist_ok=True)
        cfg.seed = 5
        appset.reset_settings(["seed"])
        assert cfg.seed is None
        assert not (cfg.data_root / "config.toml").exists()

    def test_reset_refuses_a_chunk_size_below_the_kept_overlap(self):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        root = self._write_config("chunk_size = 2048\nchunk_overlap = 1000\n")
        cfg.chunk_size = 2048
        cfg.chunk_overlap = 1000
        with pytest.raises(
            ValueError, match=r"chunk_overlap \(1000\) must be < chunk_size \(512\)"
        ):
            appset.reset_settings(["chunk_size"])
        assert (cfg.chunk_size, cfg.chunk_overlap) == (2048, 1000)
        assert settings.load(root) == {"chunk_size": 2048, "chunk_overlap": 1000}

    def test_reset_refuses_model_roles_when_the_surface_owns_them_elsewhere(self):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        root = self._write_config('chat_model = "acme/a-GGUF/a.gguf"\n')
        with pytest.raises(ValueError, match="'chat_model' must be set through the dedicated"):
            appset.reset_settings(["top_k", "chat_model"], allow_model_roles=False)
        assert settings.load(root) == {"chat_model": "acme/a-GGUF/a.gguf"}
        assert cfg.top_k == 12

    def test_reset_embedding_model_pins_meta_first_and_reports_reindex(self, monkeypatch):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        root = self._write_config('embedding_model = "acme/b-GGUF/b.gguf"\n')
        cfg.embedding_model = "acme/b-GGUF/b.gguf"
        calls: list[str] = []

        def pin() -> None:
            calls.append(settings.load(root).get("embedding_model", "<gone>"))

        monkeypatch.setattr(appset, "_pin_legacy_store_meta", pin)
        monkeypatch.setattr(appset, "_embedder_dim_from_gguf", lambda ref, registry=None: None)
        monkeypatch.setattr(appset, "_invalidate_caches", lambda keys: None)
        monkeypatch.setattr(appset, "_embed_reindex_required", lambda: True)
        result = appset.reset_settings(["embedding_model"])
        assert calls == ["acme/b-GGUF/b.gguf"]
        assert result.reindex_required is True
        assert cfg.embedding_model == builtin_value("embedding_model")
        assert "embedding_model" not in settings.load(root)

    def test_reset_validates_the_profile_value_it_falls_back_to(self):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        root = self._write_config(
            "chunk_size = 2048\nchunk_overlap = 1000\n[profile.values]\nchunk_size = 1500\n"
        )
        cfg.chunk_size = 2048
        cfg.chunk_overlap = 1000
        appset.reset_settings(["chunk_size"])
        assert (cfg.chunk_size, cfg.chunk_overlap) == (1500, 1000)
        assert "chunk_size" not in settings.load(root)

    def test_reset_refuses_an_invalid_profile_value_and_changes_nothing(self):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        root = self._write_config("max_distance = 0.3\n[profile.values]\nmax_distance = -5.0\n")
        cfg.max_distance = 0.3
        with pytest.raises(
            ValueError, match=r"Cannot reset 'max_distance': its profile value -5\.0 is invalid"
        ):
            appset.reset_settings(["max_distance"])
        assert cfg.max_distance == 0.3
        assert settings.load(root)["max_distance"] == 0.3

    def test_reset_with_duplicate_keys_removes_the_key_once(self):
        from lilbee.app import settings as appset
        from lilbee.core.config import cfg

        root = self._write_config("top_k = 7\nseed = 3\n")
        cfg.top_k = 7
        assert appset.reset_settings(["top_k", "top_k"]).updated == ["top_k"]
        assert settings.load(root) == {"seed": 3}
        assert cfg.top_k == 12

    def test_reset_keeps_refusing_documents_dir(self):
        from lilbee.app import settings as appset

        with pytest.raises(
            ValueError, match="'documents_dir' has no default to reset to; set a folder path"
        ):
            appset.reset_settings(["documents_dir"])
        assert appset.reset_settings(["documents_dir"], skip_unresettable=True).updated == []

    def test_reset_unknown_key_is_refused(self):
        from lilbee.app import settings as appset

        with pytest.raises(ValueError, match="Unknown or read-only setting: nope"):
            appset.reset_settings(["nope"])


class TestDeleteValues:
    def test_absent_keys_write_nothing(self, tmp_path):
        settings.delete_values(tmp_path, ["top_k"])
        assert not (tmp_path / "config.toml").exists()

    def test_present_key_is_removed_and_others_kept(self, tmp_path):
        settings.update_values(tmp_path, {"top_k": 7, "seed": 3})
        settings.delete_values(tmp_path, ["top_k", "absent"])
        assert settings.load(tmp_path) == {"seed": 3}
