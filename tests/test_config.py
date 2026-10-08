"""Tests for Config (pydantic-settings BaseSettings) and env var overrides."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from conftest import PICKS_CHAT, PICKS_RERANK, PICKS_VISION, clean_env
from lilbee.core.config import (
    CHUNKS_TABLE,
    DEFAULT_IGNORE_DIRS,
    SOURCES_TABLE,
    Config,
    cfg,
    config_scope,
    validate_ocr_timeout,
)
from lilbee.core.config.defaults import DEFAULT_CORS_ORIGIN_REGEX
from lilbee.core.config.enums import ChatMode, FtsLanguage, KvCacheType, OcrMode
from lilbee.core.config.model import _TomlSource, value_is_set
from lilbee.runtime.progress import OcrBackendUsed

_SAMPLE_CHAT_REF = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
_SAMPLE_EMBED_REF = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
_SAMPLE_VISION_REF = "Qwen/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"


class TestFromEnvDefaults:
    def test_default_values(self, tmp_path):
        with (
            mock.patch.dict(os.environ, clean_env(tmp_path), clear=True),
            mock.patch(
                "lilbee.core.system._read_total_memory_bytes",
                return_value=8 * 1024**3,
            ),
        ):
            c = Config()
            assert c.chat_model == ""
            assert c.embedding_model == ""
            assert c.embedding_dim == 768
            assert c.chunk_size == 512
            assert c.chunk_overlap == 100
            assert c.max_embed_chars == 2000
            assert c.top_k == 12
            assert c.max_distance == 0.75
            assert c.json_mode is False
            # Ingest workers auto-size to one per GPU; 1 keeps ingest in-process.
            assert c.ingest_processes == 0
            # Memory-budget defaults: q8_0 KV halves per-token cost vs f16,
            # 8K target keeps the working window inside chat-with-RAG needs,
            # and ``None`` num_ctx_max lets the model's training_ctx be the
            # only ceiling on hosts with the RAM to back it.
            from lilbee.core.config.enums import KvCacheType

            assert c.kv_cache_type is KvCacheType.Q8_0
            assert c.chat_n_ctx_target == 8192
            assert c.num_ctx_max is None
            # Wiki is opt-in: the Wiki view tab and the chat ModelBar's
            # scope picker only appear when the user explicitly enables it.
            assert c.wiki is False
            # Local-server URLs default blank; the resolver fills the spec
            # default so the literal lives only in the spec.
            assert c.ollama_base_url == ""
            assert c.lm_studio_base_url == ""

    def test_local_server_urls_strip_trailing_slash(self):
        c = Config(
            ollama_base_url="http://box:11434/",
            lm_studio_base_url="http://lm:1234/v1/",
        )
        assert c.ollama_base_url == "http://box:11434"
        assert c.lm_studio_base_url == "http://lm:1234/v1"

    def test_constants_unchanged(self):
        assert CHUNKS_TABLE == "chunks"
        assert SOURCES_TABLE == "_sources"
        assert "node_modules" in DEFAULT_IGNORE_DIRS

    def test_config_field_public_false_marker(self):
        """ConfigField(public=False) stores the flag in json_schema_extra."""
        from lilbee.core.config import ConfigField

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

    def test_data_root_expands_user_home(self):
        """A ~ in LILBEE_DATA_ROOT expands: systemd/.env deliver a literal '~'
        that would otherwise create a './~' tree and split a path-keyed lock."""
        env = clean_env()
        env.pop("LILBEE_DATA", None)
        env["LILBEE_SKIP_TOML_CONFIG"] = "1"
        env["LILBEE_DATA_ROOT"] = "~/lilbee_expanduser_probe"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.data_root == Path.home() / "lilbee_expanduser_probe"
            assert c.lancedb_dir == Path.home() / "lilbee_expanduser_probe" / "data" / "lancedb"

    def test_symlinked_data_root_keys_the_same_paths_as_its_target(self, tmp_path):
        """Two spellings of one directory derive one set of lock paths."""
        real = tmp_path / "real_root"
        real.mkdir()
        link = tmp_path / "link_root"
        link.symlink_to(real)

        with mock.patch.dict(os.environ, {"LILBEE_DATA": str(real)}):
            direct = Config()
        with mock.patch.dict(os.environ, {"LILBEE_DATA": str(link)}):
            through_link = Config()

        assert through_link.data_root == direct.data_root
        assert through_link.data_dir == direct.data_dir
        assert through_link.lancedb_dir == direct.lancedb_dir

    def test_padded_data_env_finds_the_same_dir_for_root_and_config(
        self, tmp_path, overlay_reads_config_toml
    ):
        """A padded LILBEE_DATA sends the root and its config.toml to one dir."""
        (tmp_path / "config.toml").write_text("top_k = 7\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"LILBEE_DATA": f"  {tmp_path}  "}):
            c = Config()
        assert c.data_root == tmp_path
        assert c.top_k == 7

    def test_relative_data_root_resolves_absolute(self, tmp_path, monkeypatch):
        """A relative root must not re-key on the process working directory."""
        (tmp_path / "kb").mkdir()
        monkeypatch.chdir(tmp_path)
        with mock.patch.dict(os.environ, {"LILBEE_DATA": "kb"}):
            c = Config()
        assert c.data_root.is_absolute()
        assert c.data_root == (tmp_path / "kb").resolve()

    def test_empty_data_root_falls_back_to_default_not_cwd(self):
        """An empty LILBEE_DATA_ROOT must resolve to the platform default, not
        the process cwd (which would make the data dir move with the launcher)."""
        env = clean_env()
        env.pop("LILBEE_DATA", None)
        env["LILBEE_SKIP_TOML_CONFIG"] = "1"
        env["LILBEE_DATA_ROOT"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.data_root != Path()
            assert c.data_root != Path.cwd()
            assert str(c.data_root).endswith("lilbee")
            # A raw blank string (direct construction / a string env source) hits
            # the same fall-through instead of resolving to Path(".") = cwd.
            c2 = Config(data_root="   ")
            assert c2.data_root != Path()
            assert c2.data_root != Path.cwd()
            assert str(c2.data_root).endswith("lilbee")

    def test_unresolvable_home_still_loads_config(self):
        """An unresolvable ~ still yields a usable root instead of raising.

        os.path.expanduser returns an unknown ~user unchanged; Path.expanduser
        raises.
        """
        from lilbee.core.system import canonical_data_root

        root = canonical_data_root("~nosuchuser_lilbee_probe/lilbee")
        assert root.is_absolute()
        assert str(root).endswith("lilbee")
        # The normal case still expands to the real home.
        assert canonical_data_root("~/lilbee") == Path.home() / "lilbee"

    def test_local_server_urls_from_env(self, tmp_path):
        env = clean_env(tmp_path)
        env["LILBEE_OLLAMA_BASE_URL"] = "http://box:11434"
        env["LILBEE_LM_STUDIO_BASE_URL"] = "http://lm:1234/v1"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.ollama_base_url == "http://box:11434"
            assert c.lm_studio_base_url == "http://lm:1234/v1"

    def test_data_root_default_uses_platform(self):
        env = clean_env()
        # Skip the platform-default config.toml: a dev's persisted state
        # could carry refs the new validators reject.
        env["LILBEE_SKIP_TOML_CONFIG"] = "1"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert str(c.data_root).endswith("lilbee")

    def test_module_import_canonicalizes_lilbee_data_env(self):
        """``lilbee.core.config`` exports ``LILBEE_DATA`` so spawned workers inherit it.

        Spawn-context workers re-import lilbee in a fresh process. The
        canonicalization at cfg construction means a worker spawned
        after lilbee has been imported once always sees a populated
        ``LILBEE_DATA`` and routes ``worker-*.log`` accordingly.
        """
        import lilbee.core.config  # noqa: F401  # ensure module-level side effect ran

        assert os.environ.get("LILBEE_DATA")

    def test_chat_model_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_CHAT_MODEL": "ollama/llama3:8b"}):
            c = Config()
            assert c.chat_model == "ollama/llama3:8b"

    def test_chat_model_override_native_hf_ref(self):
        ref = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
        with mock.patch.dict(os.environ, {"LILBEE_CHAT_MODEL": ref}):
            c = Config()
            assert c.chat_model == ref


class TestOcrLanguage:
    def test_defaults_to_english(self, tmp_path):
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            assert Config().ocr_language == ["eng"]

    def test_env_plus_separated(self, tmp_path):
        env = clean_env(tmp_path) | {"LILBEE_OCR_LANGUAGE": "eng+deu"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert Config().ocr_language == ["eng", "deu"]

    def test_env_comma_separated(self, tmp_path):
        env = clean_env(tmp_path) | {"LILBEE_OCR_LANGUAGE": "deu, fra"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert Config().ocr_language == ["deu", "fra"]

    def test_direct_list(self):
        assert Config(ocr_language=["spa"]).ocr_language == ["spa"]

    def test_persisted_newline_form_round_trips(self):
        """app.settings joins list values with '\\n' before writing config.toml;
        the validator must split on it so a multi-language value reloads intact."""
        persisted = "\n".join(["eng", "deu"])
        assert Config(ocr_language=persisted).ocr_language == ["eng", "deu"]

    def test_empty_list_falls_back_to_english(self):
        assert Config(ocr_language=[]).ocr_language == ["eng"]

    def test_blank_string_falls_back_to_english(self):
        assert Config(ocr_language="").ocr_language == ["eng"]

    def test_blank_entries_are_dropped(self):
        assert Config(ocr_language=["", "eng", "  "]).ocr_language == ["eng"]

    @pytest.mark.parametrize("code", ["en", "eng", "chi_sim", "jpn_vert", "aze_cyrl"])
    def test_accepts_tesseract_language_codes(self, code):
        assert Config(ocr_language=[code]).ocr_language == [code]

    @pytest.mark.parametrize("value", ["1 lines", "eng+1 lines", ["eng", "English"], "e"])
    def test_rejects_values_that_are_not_language_codes(self, value):
        """A settings-screen summary once landed here; xberg then refused every ingest."""
        with pytest.raises(ValueError, match="language code"):
            Config(ocr_language=value)

    def test_chat_mode_defaults_to_search_when_none_or_empty(self):
        """The validator coerces None / "" to 'search' so old configs round-trip."""
        from lilbee.core.config.model import Config as ConfigCls

        assert ConfigCls._normalize_chat_mode(None) == "search"
        assert ConfigCls._normalize_chat_mode("") == "search"

    def test_chat_mode_rejects_unknown_value(self):
        from lilbee.core.config.model import Config as ConfigCls

        with pytest.raises(ValueError, match="chat_mode must be"):
            ConfigCls._normalize_chat_mode("rag")

    @pytest.mark.parametrize(
        ("given", "expected"),
        [("search", "search"), ("SEARCH", "search"), ("  Search ", "search"), ("CHAT", "chat")],
    )
    def test_chat_mode_normalizes_case_and_padding(self, given, expected):
        """Every casing and padding the field accepted before is still accepted."""
        assert Config(chat_mode=given).chat_mode == expected

    def test_embedding_model_override(self):
        ref = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
        with mock.patch.dict(os.environ, {"LILBEE_EMBEDDING_MODEL": ref}):
            c = Config()
            assert c.embedding_model == ref

    def test_bare_name_tag_rejected_on_assignment(self):
        """Bare ``name:tag`` strings are not accepted by the cfg validator."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must be a HuggingFace ref"):
            cfg.chat_model = "qwen3:0.6b"

    def test_normalize_model_tag_empty_string_passthrough(self):
        """The validator's empty-string guard returns immediately."""
        cfg.vision_model = ""
        assert cfg.vision_model == ""

    def test_normalize_model_tag_blank_clears_role(self):
        """A blank ref normalizes to empty: the role is unconfigured, not invalid."""
        original_chat, original_embed = cfg.chat_model, cfg.embedding_model
        try:
            cfg.chat_model = "   "
            cfg.embedding_model = "\t"
            assert cfg.chat_model == ""
            assert cfg.embedding_model == ""
        finally:
            cfg.chat_model = original_chat or ""
            cfg.embedding_model = original_embed or ""

    def test_fusion_config_fields_enforce_their_bounds(self):
        """The new fusion/expansion knobs reject out-of-range values, so a bad
        config surfaces at assignment instead of silently mis-weighting fusion."""
        from pydantic import ValidationError

        for field, bad in [
            ("lexical_fusion_weight", 1.5),
            ("lexical_fusion_weight", -0.1),
            ("adaptive_fusion_margin", 2.5),
            ("adaptive_fusion_margin", -0.1),
            ("title_search_weight", 1.5),
            ("title_search_weight", -0.1),
            ("neighbor_expansion", -1),
            ("neighbor_expansion", 101),  # upper bound guards against a token-count misread
        ]:
            with pytest.raises(ValidationError):
                setattr(cfg, field, bad)

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

    def test_rag_system_prompt_override(self):
        with mock.patch.dict(os.environ, {"LILBEE_RAG_SYSTEM_PROMPT": "You are a pirate."}):
            c = Config()
            assert c.rag_system_prompt == "You are a pirate."


class TestForceOcrPages:
    def test_defaults_to_no_pages(self, tmp_path):
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            assert Config().force_ocr_pages == []

    def test_env_comma_separated(self, tmp_path):
        env = clean_env(tmp_path) | {"LILBEE_FORCE_OCR_PAGES": "1,3"}
        with mock.patch.dict(os.environ, env, clear=True):
            assert Config().force_ocr_pages == [1, 3]

    def test_direct_list(self):
        assert Config(force_ocr_pages=[2, 5]).force_ocr_pages == [2, 5]

    def test_persisted_newline_form_is_sorted_and_deduplicated(self):
        """app.settings joins list values with '\n' before writing config.toml."""
        assert Config(force_ocr_pages="3\n1\n3").force_ocr_pages == [1, 3]

    @pytest.mark.parametrize("value", [["1,3"], ["3", "1,3"]])
    def test_comma_separated_items_inside_a_list_are_split(self, value):
        assert Config(force_ocr_pages=value).force_ocr_pages == [1, 3]

    def test_a_bare_int_is_one_page(self):
        assert Config(force_ocr_pages=3).force_ocr_pages == [3]

    def test_a_bare_zero_is_rejected(self):
        with pytest.raises(ValueError, match="page numbers start at 1"):
            Config(force_ocr_pages=0)

    @pytest.mark.parametrize("value", [{"page": 1}, 1.5, True])
    def test_a_bare_non_page_value_is_rejected(self, value):
        with pytest.raises(ValueError, match="not a page number"):
            Config(force_ocr_pages=value)

    def test_string_items_from_the_list_editor_are_parsed(self):
        assert Config(force_ocr_pages=["4", " 2 "]).force_ocr_pages == [2, 4]

    def test_blank_string_is_no_pages(self):
        assert Config(force_ocr_pages="").force_ocr_pages == []

    @pytest.mark.parametrize("value", [[0], "-2", "1,0"])
    def test_rejects_pages_below_one(self, value):
        with pytest.raises(ValueError, match="page numbers start at 1"):
            Config(force_ocr_pages=value)

    @pytest.mark.parametrize("value", ["1,x", ["two"], [1.5], [True]])
    def test_rejects_values_that_are_not_page_numbers(self, value):
        with pytest.raises(ValueError, match="not a page number"):
            Config(force_ocr_pages=value)


class TestOcrStrategy:
    def test_defaults_to_auto(self, tmp_path):
        from lilbee.core.config.enums import OcrPageStrategy

        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            config = Config()
        assert config.ocr_strategy is OcrPageStrategy.AUTO
        assert config.ocr_scan_confidence == 0.7

    def test_env_selects_scanned_pages(self, tmp_path):
        from lilbee.core.config.enums import OcrPageStrategy

        env = clean_env(tmp_path) | {
            "LILBEE_OCR_STRATEGY": "scanned_pages",
            "LILBEE_OCR_SCAN_CONFIDENCE": "0.5",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            config = Config()
        assert config.ocr_strategy is OcrPageStrategy.SCANNED_PAGES
        assert config.ocr_scan_confidence == 0.5

    def test_rejects_an_unknown_strategy(self):
        with pytest.raises(ValueError):
            Config(ocr_strategy="every_page")

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_rejects_a_confidence_outside_zero_to_one(self, value):
        with pytest.raises(ValueError):
            Config(ocr_scan_confidence=value)


class TestTomlConfigFile:
    def test_toml_values_loaded(self, tmp_path):
        ref = "ollama/my-saved-model:latest"
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(f'chat_model = "{ref}"\n')
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.chat_model == ref

    def test_env_var_overrides_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('chat_model = "ollama/toml-model:latest"\n')
        env_ref = "ollama/env-model:latest"
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_CHAT_MODEL"] = env_ref
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.chat_model == env_ref

    def test_no_toml_uses_defaults(self, tmp_path):
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            assert c.chat_model == ""

    def test_corrupt_toml_uses_defaults(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("this is not valid TOML [[[")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.chat_model == ""

    def test_embedding_model_from_toml(self, tmp_path):
        ref = "ollama/my-embed:latest"
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(f'embedding_model = "{ref}"\n')
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.embedding_model == ref

    def test_temperature_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("temperature = 0.5\n")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.temperature == 0.5

    def test_env_var_overrides_toml_for_temperature(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("temperature = 0.5\n")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_TEMPERATURE"] = "0.9"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.temperature == 0.9

    def test_rag_system_prompt_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('rag_system_prompt = "You are a pirate."\n')
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.rag_system_prompt == "You are a pirate."

    def test_env_var_overrides_toml_for_rag_system_prompt(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('rag_system_prompt = "Be verbose."\n')
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_RAG_SYSTEM_PROMPT"] = "Be brief."
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.rag_system_prompt == "Be brief."

    def test_ocr_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('ocr = "off"\n', encoding="utf-8")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.ocr is OcrMode.OFF

    def test_list_field_from_toml_stays_a_list(self, tmp_path):
        """A TOML array maps to a native list, not its stringified repr."""
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            'cors_origins = ["https://a.example", "https://b.example"]\n'
            'crawl_exclude_patterns = [".*/private/.*"]\n'
            'crawl_browser_extra_args = ["--disable-gpu", "--no-sandbox"]\n'
        )
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.cors_origins == ["https://a.example", "https://b.example"]
            assert c.crawl_exclude_patterns == [".*/private/.*"]
            # No before-validator splitter, so the old str(v) coercion hard-failed here.
            assert c.crawl_browser_extra_args == ["--disable-gpu", "--no-sandbox"]

    def test_empty_string_scalar_in_toml_falls_back_to_default(self, tmp_path):
        """Legacy '' sentinel (set_setting wrote it for None) is dropped, not coerced."""
        toml_path = tmp_path / "config.toml"
        toml_path.write_text('chat_model = ""\n')
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.chat_model == ""

    def test_top_p_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("top_p = 0.9\n")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.top_p == 0.9

    def test_top_k_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("top_k = 20\n")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.top_k == 20

    def test_top_k_sampling_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("top_k_sampling = 40\n")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.top_k_sampling == 40

    def test_repeat_penalty_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("repeat_penalty = 1.2\n")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.repeat_penalty == 1.2

    def test_repeat_penalty_defaults_to_one_point_one(self, tmp_path):
        """Fresh Config defaults repeat_penalty to 1.1 so chat doesn't loop."""
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.repeat_penalty == 1.1

    def test_theme_defaults_to_rose_pine(self, tmp_path):
        """Fresh Config opens in rose-pine; the muted palette is the agreed default.

        Regression guard: the TUI fallback in app.py uses rose-pine, but
        cfg.theme is always populated, so the fallback never fires. The
        config-side default has to match or fresh installs see gruvbox.
        """
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.theme == "rose-pine"

    def test_theme_default_matches_app_fallback(self, tmp_path):
        """The Config default and the TUI fallback constant must agree.

        If they drift, the visible default and the documented default
        diverge, which is how bb-akqw-style regressions sneak in.
        """
        from lilbee.cli.tui.app import _DEFAULT_THEME

        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.theme == _DEFAULT_THEME

    def test_num_ctx_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("num_ctx = 4096\n")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.num_ctx == 4096

    def test_seed_from_toml(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("seed = 123\n")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.seed == 123


class TestOcrModeConfig:
    def test_default_is_auto(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            assert Config().ocr is OcrMode.AUTO

    @pytest.mark.parametrize("raw", ["auto", "all", "off"])
    def test_from_env(self, tmp_path, raw) -> None:
        with mock.patch.dict(os.environ, {**clean_env(tmp_path), "LILBEE_OCR": raw}, clear=True):
            assert Config().ocr is OcrMode(raw)

    def test_unknown_mode_is_refused(self, tmp_path) -> None:
        with (
            mock.patch.dict(os.environ, clean_env(tmp_path), clear=True),
            pytest.raises(ValidationError, match="'auto', 'all' or 'off'"),
        ):
            Config(ocr="some")

    @pytest.mark.parametrize("env_var", ["LILBEE_ENABLE_OCR", "LILBEE_OCR_FORCE"])
    def test_retired_env_vars_are_not_read(self, tmp_path, env_var) -> None:
        """No shim: the retired variables set nothing, only LILBEE_OCR does."""
        with mock.patch.dict(os.environ, {**clean_env(tmp_path), env_var: "0"}, clear=True):
            assert Config().ocr is OcrMode.AUTO
        with mock.patch.dict(os.environ, {**clean_env(tmp_path), env_var: "1"}, clear=True):
            assert Config().ocr is OcrMode.AUTO


class TestRetiredOcrKeysMigrate:
    """A config.toml written before the ocr setting loads as the mode it meant."""

    @staticmethod
    def _load(tmp_path, toml: str, **env: str) -> Config:
        (tmp_path / "config.toml").write_text(toml, encoding="utf-8")
        with mock.patch.dict(os.environ, {**clean_env(tmp_path), **env}, clear=True):
            return Config()

    @pytest.mark.parametrize(
        ("toml", "expected"),
        [
            ("enable_ocr = false\n", OcrMode.OFF),
            (f'enable_ocr = false\nvision_model = "{_SAMPLE_VISION_REF}"\n', OcrMode.AUTO),
            ("enable_ocr = true\n", OcrMode.AUTO),
            ('enable_ocr = "auto"\n', OcrMode.AUTO),
            ('enable_ocr = "none"\n', OcrMode.AUTO),
            ('enable_ocr = "off"\n', OcrMode.OFF),
            ('enable_ocr = "maybe"\n', OcrMode.AUTO),
            ("force_ocr = true\n", OcrMode.ALL),
            ("enable_ocr = true\nforce_ocr = true\n", OcrMode.ALL),
            ("enable_ocr = false\nforce_ocr = true\n", OcrMode.OFF),
            (
                f'enable_ocr = false\nforce_ocr = true\nvision_model = "{_SAMPLE_VISION_REF}"\n',
                OcrMode.ALL,
            ),
            ("force_ocr = false\n", OcrMode.AUTO),
        ],
    )
    def test_stored_values_become_the_mode_they_meant(self, tmp_path, toml, expected):
        assert self._load(tmp_path, toml).ocr is expected

    def test_a_vision_model_from_the_env_counts(self, tmp_path):
        """The migration sees the merged settings, so an env vision model keeps OCR on."""
        loaded = self._load(
            tmp_path, "enable_ocr = false\n", LILBEE_VISION_MODEL=_SAMPLE_VISION_REF
        )
        assert loaded.vision_model == _SAMPLE_VISION_REF
        assert loaded.ocr is OcrMode.AUTO

    def test_an_explicit_ocr_wins_over_a_retired_key(self, tmp_path):
        loaded = self._load(tmp_path, 'enable_ocr = false\nocr = "all"\n')
        assert loaded.ocr is OcrMode.ALL

    def test_an_invalid_ocr_keeps_off_from_enable_ocr_through_a_write(self, tmp_path, caplog):
        """Load, write another key, reload: OCR stays off and the warning names its source."""
        import tomllib

        from lilbee.core import settings

        with caplog.at_level("WARNING"):
            loaded = self._load(tmp_path, 'enable_ocr = false\nocr = "OFF"\ntop_k = 7\n')
        assert loaded.ocr is OcrMode.OFF
        assert (
            "config.toml: ocr = 'OFF' is not one of auto, all, off; ocr comes from enable_ocr"
            in caplog.text
        )
        assert "ocr uses its default" not in caplog.text

        settings.set_value(tmp_path, "top_k", 9)
        stored = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
        assert stored == {"ocr": "off", "top_k": 9}
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            reloaded = Config()
        assert (reloaded.ocr, reloaded.top_k) == (OcrMode.OFF, 9)

    def test_a_write_keeps_a_valid_ocr_over_a_retired_key(self, tmp_path):
        import tomllib

        from lilbee.core import settings

        (tmp_path / "config.toml").write_text('enable_ocr = false\nocr = "all"\n', encoding="utf-8")
        settings.set_value(tmp_path, "top_k", 9)
        stored = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
        assert stored == {"ocr": "all", "top_k": 9}

    def test_an_invalid_ocr_without_a_retired_key_uses_the_default(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            loaded = self._load(tmp_path, 'ocr = "OFF"\n')
        assert loaded.ocr is OcrMode.AUTO
        assert "config.toml: ocr = 'OFF' is not one of auto, all, off; ocr uses its default" in (
            caplog.text
        )

    @pytest.mark.parametrize(
        ("toml", "warning"),
        [
            ("enable_ocr = false\n", "config.toml: enable_ocr is replaced by ocr;"),
            (
                "enable_ocr = false\nforce_ocr = true\n",
                "config.toml: enable_ocr and force_ocr are replaced by ocr;",
            ),
        ],
    )
    def test_the_migration_warns_with_the_replacement(self, tmp_path, caplog, toml, warning):
        with caplog.at_level("WARNING", logger="lilbee.core.config.parsing"):
            self._load(tmp_path, toml)
        assert warning in caplog.text


class TestMigrationMatchesTheOldEngineChoice:
    """Differential: a migrated enable_ocr picks the engine the old setting picked."""

    @staticmethod
    def _old_backend(stored: object, vision_model: str) -> OcrBackendUsed:
        """The engine choice before the ocr setting: vision first, then enable_ocr."""
        from lilbee.core.config.parsing import parse_bool

        if vision_model:
            return OcrBackendUsed.VISION
        if isinstance(stored, bool):
            enabled: bool | None = stored
        elif isinstance(stored, str):
            auto = stored.strip().lower() in ("", "auto", "none")
            try:
                enabled = None if auto else parse_bool(stored)
            except ValueError:
                enabled = None
        else:
            enabled = bool(stored)
        return OcrBackendUsed.NONE if enabled is False else OcrBackendUsed.TESSERACT

    @pytest.mark.parametrize("vision_model", ["", _SAMPLE_VISION_REF])
    @pytest.mark.parametrize(
        "stored", [True, False, "true", "false", "", "auto", "none", "off", "on", "maybe", 0, 1, 2]
    )
    def test_the_engine_is_unchanged(self, stored, vision_model):
        from lilbee.core.config.parsing import migrate_ocr_keys

        migrated = OcrMode(migrate_ocr_keys({"enable_ocr": stored}, vision_model)["ocr"])
        assert OcrBackendUsed.chosen(migrated, vision_model) is self._old_backend(
            stored, vision_model
        )


class TestFlashAttentionConfig:
    def test_default_is_none(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            assert c.flash_attention is None

    def test_bool_true_passes_through(self) -> None:
        """A bool already in the right type bypasses string parsing."""
        with mock.patch.dict(os.environ, {"LILBEE_FLASH_ATTENTION": "true"}):
            c = Config()
            assert c.flash_attention is True

    def test_invalid_string_falls_back_to_none(self, tmp_path) -> None:
        """Garbage values fall back to auto rather than crashing the load."""
        env = {**clean_env(tmp_path), "LILBEE_FLASH_ATTENTION": "maybe?"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.flash_attention is None

    def test_assignment_with_bool(self, tmp_path) -> None:
        """Validator on assignment accepts bool inputs verbatim."""
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            c.flash_attention = True
            assert c.flash_attention is True

    def test_assignment_with_int_coerces_to_bool(self, tmp_path) -> None:
        """Non-string non-bool values fall through to bool(v)."""
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            c.flash_attention = 1  # type: ignore[assignment]
            assert c.flash_attention is True


class TestNGpuLayersConfig:
    def test_default_is_none(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            assert c.n_gpu_layers is None

    def test_cpu_alias_means_zero(self, tmp_path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_N_GPU_LAYERS": "cpu"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.n_gpu_layers == 0

    def test_explicit_int_string(self, tmp_path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_N_GPU_LAYERS": "12"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.n_gpu_layers == 12

    def test_invalid_string_falls_back_to_none(self, tmp_path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_N_GPU_LAYERS": "not-a-number"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.n_gpu_layers is None

    def test_assignment_with_int(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            c.n_gpu_layers = 4
            assert c.n_gpu_layers == 4

    def test_assignment_with_float_coerces(self, tmp_path) -> None:
        """Non-string non-None values fall through to int(v)."""
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            c.n_gpu_layers = 3.0  # type: ignore[assignment]
            assert c.n_gpu_layers == 3


class TestMainGpuConfig:
    def test_default_is_none(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            assert c.main_gpu is None

    def test_explicit_int_from_env(self, tmp_path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_MAIN_GPU": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.main_gpu == 1

    def test_auto_string_means_none(self, tmp_path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_MAIN_GPU": "auto"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.main_gpu is None

    def test_invalid_string_falls_back_to_none(self, tmp_path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_MAIN_GPU": "garbage"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.main_gpu is None

    def test_non_string_input_coerces_to_int(self, tmp_path) -> None:
        """Direct assignment with a non-string value falls through to int(v)."""
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            c.main_gpu = 2.0  # type: ignore[assignment]
            assert c.main_gpu == 2


class TestGpuDevicesConfig:
    def test_default_is_none(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            assert c.gpu_devices is None

    def test_single_index_from_env(self, tmp_path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_GPU_DEVICES": "0"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.gpu_devices == "0"

    def test_multi_index_from_env(self, tmp_path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_GPU_DEVICES": " 0, 1 "}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.gpu_devices == "0,1"

    def test_all_alias_means_none(self, tmp_path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_GPU_DEVICES": "all"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.gpu_devices is None

    def test_non_numeric_falls_back_to_none(self, tmp_path) -> None:
        env = {**clean_env(tmp_path), "LILBEE_GPU_DEVICES": "rtx-4060"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.gpu_devices is None

    def test_only_separators_falls_back_to_none(self, tmp_path) -> None:
        """A string that splits into zero parts ('  ,  ,') normalizes to None."""
        env = {**clean_env(tmp_path), "LILBEE_GPU_DEVICES": " , ,"}
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.gpu_devices is None

    def test_non_string_input_coerces_to_str(self, tmp_path) -> None:
        """Direct assignment with a non-string value falls through to str(v)."""
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            c.gpu_devices = 0  # type: ignore[assignment]
            assert c.gpu_devices == "0"


class TestSemanticChunkingConfig:
    def test_default_is_false(self, tmp_path) -> None:
        """Semantic chunking is opt-in: default False, enabled via env/config."""
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
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
            caplog.at_level(logging.WARNING, logger="lilbee.core.config"),
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
        from lilbee.core.config import Config

        parse = Config._parse_semantic_chunking
        assert parse(1) is True
        assert parse(0) is False
        assert parse([1]) is True
        assert parse([]) is False


class TestResolveDefaultsValidator:
    def test_non_dict_input_passes_through(self) -> None:
        """The before-validator hands non-dict input straight back; pydantic
        then rejects it as not a valid model."""
        from pydantic import ValidationError

        from lilbee.core.config import Config

        with pytest.raises(ValidationError, match="valid dict"):
            Config.model_validate(["not", "a", "dict"])

    def test_from_toml(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("semantic_chunking = false\n")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            assert Config().semantic_chunking is False

    def test_env_overrides_toml(self, tmp_path) -> None:
        toml_path = tmp_path / "config.toml"
        toml_path.write_text("semantic_chunking = false\n")
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        env["LILBEE_SEMANTIC_CHUNKING"] = "true"
        with mock.patch.dict(os.environ, env, clear=True):
            assert Config().semantic_chunking is True

    def test_data_root_env_var_is_coerced_to_path(self, tmp_path) -> None:
        """LILBEE_DATA_ROOT sets the data_root field directly as a string;
        deriving the child paths from it must not raise on str / str."""
        env = clean_env()
        env["LILBEE_DATA_ROOT"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.data_root == tmp_path
            assert c.documents_dir == tmp_path / "documents"
            assert c.data_dir == tmp_path / "data"


class TestTopicThresholdConfig:
    def test_default_is_0_75(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
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
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            assert Config().topic_threshold == pytest.approx(0.42)


class TestParseBool:
    def test_truthy_values(self) -> None:
        from lilbee.core.config.parsing import parse_bool

        for truthy in ("true", "TRUE", "1", "yes", "  YES  "):
            assert parse_bool(truthy) is True

    def test_falsy_values(self) -> None:
        from lilbee.core.config.parsing import parse_bool

        for falsy in ("false", "FALSE", "0", "no", "  NO  "):
            assert parse_bool(falsy) is False

    def test_invalid_raises(self) -> None:
        from lilbee.core.config.parsing import parse_bool

        with pytest.raises(ValueError, match="Invalid boolean"):
            parse_bool("maybe")


class TestOcrTimeoutConfig:
    def test_default_is_300(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            assert c.ocr_timeout == 300.0

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
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            assert c.cors_origins == []

    def test_cors_origins_list_passthrough(self) -> None:
        """List values pass through the validator unchanged."""
        cfg.cors_origins = ["https://a.com", "https://b.com"]
        assert cfg.cors_origins == ["https://a.com", "https://b.com"]


class TestCorsOriginRegexConfig:
    def test_cors_origin_regex_default_matches_obsidian_desktop(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert pat.fullmatch("app://obsidian.md")

    def test_cors_origin_regex_default_matches_capacitor_localhost(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert pat.fullmatch("capacitor://localhost")

    def test_cors_origin_regex_default_matches_http_localhost_any_port(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert pat.fullmatch("http://localhost")
            assert pat.fullmatch("http://localhost:3000")
            assert pat.fullmatch("http://localhost:7433")
            assert pat.fullmatch("https://localhost:8443")

    def test_cors_origin_regex_default_matches_loopback_ipv4(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert pat.fullmatch("http://127.0.0.1:7433")
            assert pat.fullmatch("https://127.0.0.1")

    def test_cors_origin_regex_default_matches_loopback_ipv6(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert pat.fullmatch("http://[::1]:7433")
            assert pat.fullmatch("https://[::1]")

    def test_cors_origin_regex_default_rejects_random_remote(self, tmp_path) -> None:

        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            pat = re.compile(c.cors_origin_regex)
            assert not pat.fullmatch("https://evil.example.com")
            assert not pat.fullmatch("http://not-localhost.example")
            assert not pat.fullmatch("app://some-other-app.md")

    def test_cors_origin_regex_from_env_overrides_default(self, tmp_path) -> None:
        env = clean_env(tmp_path)
        env["LILBEE_CORS_ORIGIN_REGEX"] = r"^https://only-this\.example$"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.cors_origin_regex == r"^https://only-this\.example$"

    def test_cors_origin_regex_from_env_match_nothing_disables_default(self, tmp_path) -> None:
        # Empty env vars are ignored by _PlainEnvSource, so the documented opt-out is
        # to set a regex that matches nothing: e.g. ^$.
        env = clean_env(tmp_path)
        env["LILBEE_CORS_ORIGIN_REGEX"] = "^$"
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.cors_origin_regex == "^$"

    def test_cors_origin_regex_default_compiles(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            re.compile(c.cors_origin_regex)

    def test_cors_origin_regex_default_equals_constant(self, tmp_path) -> None:
        with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
            c = Config()
            assert c.cors_origin_regex == DEFAULT_CORS_ORIGIN_REGEX


class TestLocalDotLilbee:
    def test_local_lilbee_overrides_default(self, tmp_path):
        local = tmp_path / ".lilbee"
        local.mkdir()
        env = clean_env()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.core.system.find_local_root", return_value=local),
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
            mock.patch("lilbee.core.system.find_local_root", return_value=local),
        ):
            c = Config()
            assert c.data_root == explicit

    def test_no_local_uses_platform_default(self):
        env = clean_env()
        env["LILBEE_SKIP_TOML_CONFIG"] = "1"
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("lilbee.core.system.find_local_root", return_value=None),
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

    def test_max_tokens_reaches_the_providers_as_num_predict(self):
        from lilbee.providers.base import filter_options, normalize_generation_options

        c = Config()
        opts = c.generation_options()
        assert opts["num_predict"] == 4096
        assert "max_tokens" not in opts
        assert filter_options(opts)["num_predict"] == 4096
        assert normalize_generation_options(opts)["max_tokens"] == 4096

    def test_model_default_cap_uses_the_same_name(self):
        from lilbee.providers.model_defaults import ModelDefaults

        c = Config()
        c.apply_model_defaults(ModelDefaults(max_tokens=512))
        assert c.generation_options()["num_predict"] == 4096
        c.max_tokens = None
        assert c.generation_options()["num_predict"] == 512

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
        env = clean_env()
        env["LILBEE_SKIP_TOML_CONFIG"] = "1"
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
        for label in ("PERSON", "ORG", "GPE", "PRODUCT", "FAC", "EVENT"):
            assert label in c.concept_allowed_ent_types

    def test_default_excludes_quantitative_types(self):
        c = Config()
        for label in ("QUANTITY", "CARDINAL", "DATE", "TIME", "MONEY", "PERCENT"):
            assert label not in c.concept_allowed_ent_types

    def test_default_excludes_norp(self):
        """NORP surfaces are adjectival (Saturnian, American) and make poor
        page subjects; a corpus that wants them opts back in via the env
        override."""
        assert "NORP" not in Config().concept_allowed_ent_types

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
        env = clean_env()
        env["LILBEE_SKIP_TOML_CONFIG"] = "1"
        env["LILBEE_CONCEPT_ALLOWED_ENT_TYPES"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert "PERSON" in c.concept_allowed_ent_types

    def test_non_string_non_collection_input_falls_back_to_default(self):
        """A programmatic override with an unsupported type keeps defaults.

        The validator accepts str/set/frozenset/list. Anything else (None,
        int, bytes) hits the trailing fallback branch instead of raising.
        """
        c = Config(concept_allowed_ent_types=None)  # type: ignore[arg-type]
        assert "PERSON" in c.concept_allowed_ent_types


class TestEmptyStringValidation:
    def test_empty_model_roles_accepted(self, tmp_path):
        """Empty means "not configured" for both model roles, like vision_model."""
        c = Config(
            data_root=tmp_path,
            documents_dir=tmp_path / "docs",
            data_dir=tmp_path / "data",
            lancedb_dir=tmp_path / "data" / "lancedb",
            models_dir=tmp_path / "models",
            chat_model="",
            embedding_model="",
            embedding_dim=768,
            chunk_size=512,
            chunk_overlap=100,
            max_embed_chars=2000,
            top_k=10,
            max_distance=0.7,
            rag_system_prompt="You are helpful.",
            ignore_dirs=frozenset(),
        )
        assert c.chat_model == ""
        assert c.embedding_model == ""

    def test_empty_rag_system_prompt_rejected(self, tmp_path):
        with pytest.raises(Exception, match="at least 1 character"):
            Config(
                data_root=tmp_path,
                documents_dir=tmp_path / "docs",
                data_dir=tmp_path / "data",
                lancedb_dir=tmp_path / "data" / "lancedb",
                models_dir=tmp_path / "models",
                chat_model=_SAMPLE_CHAT_REF,
                embedding_model=_SAMPLE_EMBED_REF,
                embedding_dim=768,
                chunk_size=512,
                chunk_overlap=100,
                max_embed_chars=2000,
                top_k=10,
                max_distance=0.7,
                rag_system_prompt="",
                ignore_dirs=frozenset(),
            )


class TestEmptyStringToNone:
    def test_empty_temperature_falls_back_to_default(self, tmp_path):
        env = clean_env(tmp_path)
        env["LILBEE_TEMPERATURE"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.temperature == 0.1

    def test_whitespace_seed_becomes_none(self, tmp_path):
        env = clean_env(tmp_path)
        env["LILBEE_SEED"] = "   "
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.seed is None


class TestIgnoreDirsFallback:
    def test_non_string_non_collection_returns_defaults(self, tmp_path):
        env = clean_env(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config(ignore_dirs=42)  # type: ignore[arg-type]
        assert c.ignore_dirs == DEFAULT_IGNORE_DIRS


class TestDefaultCrawlExcludePatterns:
    """The out-of-the-box default exclude list blocks common noise without
    accidentally rejecting real content URLs."""

    def _matches_any(self, url: str) -> bool:
        from lilbee.core.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS

        return any(re.search(p, url) for p in DEFAULT_CRAWL_EXCLUDE_PATTERNS)

    def test_every_pattern_compiles(self):
        """Every shipped default must be valid Python regex."""
        from lilbee.core.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS

        for pattern in DEFAULT_CRAWL_EXCLUDE_PATTERNS:
            re.compile(pattern)

    def test_every_category_contributes(self):
        """Each per-category tuple appears in the master default list."""
        from lilbee.core.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS
        from lilbee.core.config.defaults import (
            _ARCHIVE_EXCLUDE,
            _ATTACHMENT_EXCLUDE,
            _AUTH_EXCLUDE,
            _DUPLICATE_VIEW_EXCLUDE,
            _ECOMMERCE_EXCLUDE,
            _FEED_EXCLUDE,
            _META_EXCLUDE,
            _TRACKING_EXCLUDE,
            _WP_EXCLUDE,
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
        from lilbee.core.config import Config

        result = Config._split_crawl_exclude_patterns("/page/\\d+\n/tag/\n/category/")
        assert result == ["/page/\\d+", "/tag/", "/category/"]

    def test_list_passes_through_unchanged(self):
        """TOML lists and programmatic lists pass through the validator."""
        from lilbee.core.config import Config

        result = Config._split_crawl_exclude_patterns(["/page/", "/tag/"])
        assert result == ["/page/", "/tag/"]

    def test_empty_string_yields_empty_list(self):
        """Empty env var collapses to an empty list, disabling the filter."""
        from lilbee.core.config import Config

        assert Config._split_crawl_exclude_patterns("") == []
        assert Config._split_crawl_exclude_patterns("\n\n  \n") == []


class TestCrawlBrowserExtraArgsValidator:
    def test_newline_separated_string_splits(self):
        """The persist path joins list values with newlines; reload must split them."""
        from lilbee.core.config import Config

        result = Config._split_crawl_browser_extra_args("--flag-a\n--flag-b")
        assert result == ["--flag-a", "--flag-b"]

    def test_list_passes_through_unchanged(self):
        from lilbee.core.config import Config

        assert Config._split_crawl_browser_extra_args(["--a", "--b"]) == ["--a", "--b"]

    def test_persisted_newline_string_round_trips(self, tmp_path):
        """A value persisted as a newline-joined string must not crash the whole
        config load (which would silently discard every other setting)."""
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            'crawl_browser_extra_args = "--flag-a\\n--flag-b"\nchat_model = "ollama/keep:latest"\n'
        )
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
            assert c.crawl_browser_extra_args == ["--flag-a", "--flag-b"]
            assert c.chat_model == "ollama/keep:latest"  # other settings survive


class TestPlainEnvSourceSkipsEmpty:
    def test_empty_chat_model_uses_default(self, tmp_path):
        env = clean_env(tmp_path)
        env["LILBEE_CHAT_MODEL"] = ""
        with mock.patch.dict(os.environ, env, clear=True):
            c = Config()
        assert c.chat_model == ""  # default, not empty


_TOML_VISION = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
_TOML_RERANKER = "org/Test-Rerank-GGUF/test-rerank-Q4_K_M.gguf"
_TOML_CHAT = "ollama/toml-chat:latest"


class TestEmptyValueClearsModelRole:
    """An empty env or config.toml value clears a model role that can be off, and nothing else."""

    @staticmethod
    def _config(tmp_path: Path, **env_values: str) -> Config:
        (tmp_path / "config.toml").write_text(
            f'vision_model = "{_TOML_VISION}"\n'
            f'reranker_model = "{_TOML_RERANKER}"\n'
            f'chat_model = "{_TOML_CHAT}"\n'
            "chunk_size = 777\n",
            encoding="utf-8",
        )
        env = clean_env(tmp_path)
        env.update(env_values)
        with mock.patch.dict(os.environ, env, clear=True):
            return Config()

    def test_unset_env_keeps_the_config_toml_vision_model(self, tmp_path):
        assert self._config(tmp_path).vision_model == _TOML_VISION

    def test_empty_env_clears_the_vision_model_and_keeps_other_toml_fields(self, tmp_path):
        c = self._config(tmp_path, LILBEE_VISION_MODEL="")
        assert c.vision_model == ""
        assert c.reranker_model == _TOML_RERANKER
        assert c.chunk_size == 777

    def test_whitespace_env_clears_the_vision_model(self, tmp_path):
        assert self._config(tmp_path, LILBEE_VISION_MODEL="   ").vision_model == ""

    def test_a_valid_env_ref_wins_over_config_toml(self, tmp_path):
        ref = "org/Other-Vision-GGUF/other-Q4_K_M.gguf"
        assert self._config(tmp_path, LILBEE_VISION_MODEL=ref).vision_model == ref

    def test_empty_env_clears_the_reranker_model(self, tmp_path):
        c = self._config(tmp_path, LILBEE_RERANKER_MODEL="")
        assert c.reranker_model == ""
        assert c.vision_model == _TOML_VISION

    def test_empty_env_on_a_required_role_or_number_still_counts_as_unset(self, tmp_path):
        c = self._config(tmp_path, LILBEE_CHAT_MODEL="", LILBEE_CHUNK_SIZE="")
        assert c.chat_model == _TOML_CHAT
        assert c.chunk_size == 777

    @pytest.mark.parametrize(
        ("field", "raw", "expected"),
        [
            ("vision_model", None, False),
            ("vision_model", "", True),
            ("reranker_model", "", True),
            ("chat_model", "", False),
            ("chunk_size", "", False),
            ("chunk_size", "5", True),
            ("chunk_size", 0, True),
            ("ocr", "off", True),
        ],
    )
    def test_value_is_set(self, field, raw, expected):
        assert value_is_set(field, raw) is expected

    def test_config_toml_keeps_an_empty_clearable_role_and_drops_other_empties(self, tmp_path):
        toml_path = tmp_path / "config.toml"
        toml_path.write_text(
            'vision_model = ""\nreranker_model = ""\nchat_model = ""\nchunk_size = ""\ntop_k = 9\n',
            encoding="utf-8",
        )
        assert _TomlSource(Config, toml_path)() == {
            "vision_model": "",
            "reranker_model": "",
            "top_k": 9,
        }


@pytest.fixture()
def _task_validation_enabled():
    """Unset the conftest-level bypass so validate_model_task_assignment fires."""
    prev = os.environ.pop("LILBEE_SKIP_MODEL_TASK_VALIDATION", None)
    try:
        yield
    finally:
        if prev is not None:
            os.environ["LILBEE_SKIP_MODEL_TASK_VALIDATION"] = prev


class TestValidateModelTaskAssignment:
    """The single write-boundary check for role-slot assignment."""

    def test_chat_slot_accepts_chat_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        ref = f"{PICKS_CHAT[0].hf_repo}/tiny-1b-Q4_K_M.gguf"
        assert validate_model_task_assignment("chat_model", ref) == ref

    def test_chat_slot_rejects_vision_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with pytest.raises(ValueError, match="vision"):
            validate_model_task_assignment("chat_model", PICKS_VISION[0].hf_repo)

    def test_chat_slot_rejects_reranker_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with pytest.raises(ValueError, match="rerank"):
            validate_model_task_assignment("chat_model", PICKS_RERANK[0].hf_repo)

    def test_embedding_slot_rejects_chat_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with pytest.raises(ValueError, match="chat"):
            validate_model_task_assignment("embedding_model", PICKS_CHAT[0].hf_repo)

    def test_vision_slot_rejects_chat_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with pytest.raises(ValueError, match="chat"):
            validate_model_task_assignment("vision_model", PICKS_CHAT[0].hf_repo)

    def test_reranker_slot_rejects_vision_model(self, _task_validation_enabled):
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with pytest.raises(ValueError, match="vision"):
            validate_model_task_assignment("reranker_model", PICKS_VISION[0].hf_repo)

    def test_empty_string_passes_through(self, _task_validation_enabled):
        """Empty or whitespace refs bypass validation (role unset)."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        assert validate_model_task_assignment("vision_model", "") == ""
        assert validate_model_task_assignment("reranker_model", "   ") == "   "

    def test_provider_prefix_bypasses_catalog(self, _task_validation_enabled):
        """Provider-prefixed refs (ollama/, openai/, ...) bypass the featured
        catalog check entirely; routing handles task taxonomy at the wire.
        """
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        ref = "ollama/qwen3:0.6b"
        assert validate_model_task_assignment("chat_model", ref) == ref

    def test_bare_hf_repo_canonicalizes_to_the_picks_ref(self, _task_validation_enabled):
        """A bare ``hf_repo`` that is a current pick resolves to that pick's ref."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        repo = PICKS_RERANK[0].hf_repo
        assert validate_model_task_assignment("reranker_model", repo) == repo

    def test_out_of_catalog_rejected(self, _task_validation_enabled):
        """Refs that are neither featured nor installed are rejected as not installed."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with pytest.raises(ValueError, match="not installed"):
            validate_model_task_assignment("chat_model", "totally-unknown-model:99b")

    def test_skip_env_var_disables_check(self, tmp_path):
        """LILBEE_SKIP_MODEL_TASK_VALIDATION bypasses the role check when pytest is imported."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with mock.patch.dict(os.environ, {"LILBEE_SKIP_MODEL_TASK_VALIDATION": "1"}):
            # Bypass: returns input unchanged, does not raise.
            result = validate_model_task_assignment("chat_model", "totally-unknown-model:99b")
        assert result == "totally-unknown-model:99b"

    def test_skip_env_var_alone_does_not_bypass_in_production(self, tmp_path):
        """Shell-level env var without the pytest sentinel must not bypass validation."""
        import sys

        from lilbee.modelhub.role_validator import validate_model_task_assignment

        saved_pytest = sys.modules.pop("pytest", None)
        try:
            with (
                mock.patch.dict(os.environ, {"LILBEE_SKIP_MODEL_TASK_VALIDATION": "1"}),
                pytest.raises(ValueError, match="not installed"),
            ):
                validate_model_task_assignment("chat_model", "totally-unknown-model:99b")
        finally:
            if saved_pytest is not None:
                sys.modules["pytest"] = saved_pytest

    def test_task_mismatch_carries_structured_fields(self, _task_validation_enabled):
        """TaskMismatchError carries the structured fields each surface needs to format messages."""
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.role_validator import TaskMismatchError, validate_model_task_assignment

        vision = PICKS_VISION[0].hf_repo
        with pytest.raises(TaskMismatchError) as exc_info:
            validate_model_task_assignment("chat_model", vision)

        err = exc_info.value
        assert err.ref == vision
        assert err.entry_task == ModelTask.VISION
        assert err.expected_task == ModelTask.CHAT

    def test_installed_non_featured_chat_model_accepted(self, _task_validation_enabled):
        """A non-featured chat model installed locally is a valid chat_model assignment."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment
        from tests.conftest import install_fake_model

        ref = install_fake_model(
            "MaziyarPanahi/Qwen3-1.7B-GGUF", "Qwen3-1.7B.Q4_K_M.gguf", task="chat"
        )
        assert validate_model_task_assignment("chat_model", ref) == ref

    def test_bare_non_featured_repo_canonicalizes_to_installed_quant(
        self, _task_validation_enabled
    ):
        """A bare non-featured repo persists as the installed quant's full ref."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment
        from tests.conftest import install_fake_model

        ref = install_fake_model(
            "bartowski/SmolLM2-360M-Instruct-GGUF", "SmolLM2-360M-Q4_K_M.gguf", task="chat"
        )
        result = validate_model_task_assignment(
            "chat_model", "bartowski/SmolLM2-360M-Instruct-GGUF"
        )
        assert result == ref

    def test_bare_non_featured_repo_without_install_rejected(self, _task_validation_enabled):
        """A bare non-featured repo with no installed quant is rejected as not installed."""
        from lilbee.modelhub.role_validator import validate_model_task_assignment

        with pytest.raises(ValueError, match="not installed"):
            validate_model_task_assignment("chat_model", "org/Never-Pulled-GGUF")

    def test_installed_non_featured_wrong_role_rejected(self, _task_validation_enabled):
        """An installed non-featured chat model in the reranker slot raises TaskMismatchError."""
        from lilbee.catalog.types import ModelTask
        from lilbee.modelhub.role_validator import TaskMismatchError, validate_model_task_assignment
        from tests.conftest import install_fake_model

        ref = install_fake_model(
            "MaziyarPanahi/Qwen3-1.7B-GGUF", "Qwen3-1.7B.Q4_K_M.gguf", task="chat"
        )
        with pytest.raises(TaskMismatchError) as exc_info:
            validate_model_task_assignment("reranker_model", ref)
        assert exc_info.value.entry_task == ModelTask.CHAT
        assert exc_info.value.expected_task == ModelTask.RERANK


class TestBuildCfgFallback:
    """The cfg-construction fallback recovers from a stale persisted config.toml."""

    def test_falls_back_to_defaults_on_validation_error(self, tmp_path):
        """A toml carrying an invalid model ref triggers the fallback path."""
        from lilbee.core.config.model import _build_cfg

        toml_path = tmp_path / "config.toml"
        # Bare ``name:tag`` is rejected by the new validator.
        toml_path.write_text('chat_model = "qwen3:0.6b"\n')
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            built_cfg, error = _build_cfg()
        assert error is not None
        assert "must be a HuggingFace ref" in str(error)
        # Falls back to defaults: the role comes back unconfigured.
        assert built_cfg.chat_model == ""

    def test_returns_none_error_on_clean_load(self, tmp_path):
        from lilbee.core.config.model import _build_cfg

        env = clean_env(tmp_path)
        env["LILBEE_SKIP_TOML_CONFIG"] = "1"
        with mock.patch.dict(os.environ, env, clear=True):
            _, error = _build_cfg()
        assert error is None

    def test_fresh_import_honors_toml_model_fields(self, tmp_path):
        """A config.toml with model refs must survive first package import.

        The model-ref validator imports the catalog package, which once imported
        cfg back at module level: on a fresh interpreter the cycle rejected every
        config.toml carrying a model field, silently falling back to defaults.
        Only a subprocess exercises the fresh-import path, so this test shells out.
        """
        import json
        import subprocess
        import sys

        pinned = "unsloth/MiniMax-M2-GGUF/Q4_K_M/MiniMax-M2-Q4_K_M-00001-of-00003.gguf"
        (tmp_path / "config.toml").write_text(
            f'chat_model = "{pinned}"\nchat_n_ctx_target = 131072\n'
        )
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        env["PATH"] = os.environ["PATH"]
        probe = (
            "import json\n"
            "from lilbee.core.config import cfg, config_load_error\n"
            "print(json.dumps({'error': str(config_load_error), "
            "'chat_model': str(cfg.chat_model)}))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
        )
        payload = json.loads(result.stdout)
        assert payload["error"] == "None"
        assert payload["chat_model"] == pinned

    def test_empty_string_persisted_nullable_uses_default(self, tmp_path):
        """Legacy bug: set_setting wrote None as ""; pydantic can't coerce.

        Empty-string TOML values must be treated as missing so a stale
        config from before that fix doesn't crash the whole Config load.
        """
        from lilbee.core.config.model import _build_cfg

        toml_path = tmp_path / "config.toml"
        toml_path.write_text('max_tokens = ""\n')
        env = clean_env()
        env["LILBEE_DATA"] = str(tmp_path)
        with mock.patch.dict(os.environ, env, clear=True):
            built_cfg, error = _build_cfg()
        assert error is None
        assert built_cfg.max_tokens == 4096


class TestABadEnumInConfigTomlKeepsTheRest:
    """One invalid enum value in config.toml drops that key alone, with a warning."""

    @staticmethod
    def _build(tmp_path, toml: str, **env: str):
        from lilbee.core.config.model import _build_cfg

        (tmp_path / "config.toml").write_text(toml, encoding="utf-8")
        with mock.patch.dict(os.environ, {**clean_env(tmp_path), **env}, clear=True):
            return _build_cfg()

    @pytest.mark.parametrize(
        ("key", "bad", "default"),
        [
            ("ocr", '"OFF"', OcrMode.AUTO),
            ("ocr", "true", OcrMode.AUTO),
            ("kv_cache_type", '"q9"', KvCacheType.Q8_0),
            ("fts_language", '"Klingon"', FtsLanguage.ENGLISH),
            ("chat_mode", '"banter"', ChatMode.SEARCH),
        ],
    )
    def test_the_bad_key_takes_its_default_and_every_other_key_loads(
        self, tmp_path, caplog, key, bad, default
    ):
        toml = (
            f"{key} = {bad}\ntop_k = 7\n"
            f'vision_model = "{_SAMPLE_VISION_REF}"\ngemini_api_key = "sk-kept"\n'
        )
        with caplog.at_level("WARNING", logger="lilbee.core.config.model"):
            built, error = self._build(tmp_path, toml)
        assert error is None
        assert getattr(built, key) == default
        assert built.top_k == 7
        assert built.vision_model == _SAMPLE_VISION_REF
        assert built.gemini_api_key == "sk-kept"
        assert f"config.toml: {key} = " in caplog.text

    def test_the_warning_names_the_key_the_value_and_the_allowed_values(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="lilbee.core.config.model"):
            self._build(tmp_path, 'ocr = "OFF"\n')
        assert (
            "config.toml: ocr = 'OFF' is not one of auto, all, off; ocr uses its default"
            in caplog.text
        )

    @pytest.mark.parametrize(
        ("key", "stored", "loaded"),
        [("fts_language", "german", FtsLanguage.GERMAN), ("chat_mode", "CHAT", ChatMode.CHAT)],
    )
    def test_a_value_the_field_normalizes_is_kept(self, tmp_path, caplog, key, stored, loaded):
        with caplog.at_level("WARNING", logger="lilbee.core.config.model"):
            built, error = self._build(tmp_path, f'{key} = "{stored}"\n')
        assert error is None
        assert getattr(built, key) is loaded
        assert "config.toml:" not in caplog.text

    def test_a_valid_env_value_still_wins_over_the_bad_toml_value(self, tmp_path):
        built, error = self._build(tmp_path, 'ocr = "OFF"\ntop_k = 7\n', LILBEE_OCR="all")
        assert error is None
        assert built.ocr is OcrMode.ALL
        assert built.top_k == 7

    def test_a_bad_value_on_a_field_that_is_not_an_enum_still_falls_back(self, tmp_path):
        built, error = self._build(tmp_path, 'top_k = "many"\nocr = "off"\n')
        assert error is not None
        assert built.ocr is OcrMode.AUTO


class TestABadEnumInTheEnvironmentKeepsTheRest:
    """One invalid enum value in a LILBEE_* variable takes its default alone, with a warning."""

    @staticmethod
    def _build(tmp_path, toml: str, **env: str):
        from lilbee.core.config.model import _build_cfg

        (tmp_path / "config.toml").write_text(toml, encoding="utf-8")
        with mock.patch.dict(os.environ, {**clean_env(tmp_path), **env}, clear=True):
            return _build_cfg()

    @staticmethod
    def _run(tmp_path, variable: str, *args: str) -> subprocess.CompletedProcess[str]:
        """Run a fresh interpreter with *variable* set to a value its setting refuses."""
        env = {
            **clean_env(tmp_path),
            variable: "bogus",
            "LILBEE_NO_SPLASH": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        return subprocess.run(
            [sys.executable, *args],
            env=env,
            capture_output=True,
            encoding="utf-8",
            timeout=120,
        )

    @pytest.mark.parametrize(
        ("variable", "key", "stored", "default", "allowed"),
        [
            ("LILBEE_OCR", "ocr", "off", "auto", "auto, all, off"),
            ("LILBEE_RERANKER_TYPE", "reranker_type", "llm", "auto", "auto, cross_encoder, llm"),
        ],
    )
    def test_help_exits_0_and_the_setting_takes_its_default(
        self, tmp_path, variable, key, stored, default, allowed
    ):
        """The losing fields: config.toml sets the same key, and one more that still loads."""
        (tmp_path / "config.toml").write_text(f'{key} = "{stored}"\ntop_k = 7\n', encoding="utf-8")
        warning = f"{variable} = 'bogus' is not one of {allowed}; {key} uses its default"
        probe = (
            "import json\n"
            "from lilbee.core.config import cfg, config_load_error\n"
            f"print(json.dumps([str(config_load_error), cfg.{key}.value, cfg.top_k]))\n"
        )

        loaded = self._run(tmp_path, variable, "-c", probe)
        assert loaded.returncode == 0, loaded.stderr
        assert json.loads(loaded.stdout) == ["None", default, 7]
        assert warning in loaded.stderr

        shown = self._run(tmp_path, variable, "-m", "lilbee", "--help")
        assert shown.returncode == 0, shown.stderr
        assert "Usage:" in shown.stdout
        assert warning in shown.stderr
        assert "Traceback" not in shown.stderr

    @pytest.mark.parametrize(
        ("key", "bad", "default"),
        [
            ("ocr", "OFF", OcrMode.AUTO),
            ("kv_cache_type", "q9", KvCacheType.Q8_0),
            ("fts_language", "Klingon", FtsLanguage.ENGLISH),
            ("chat_mode", "banter", ChatMode.SEARCH),
        ],
    )
    def test_the_bad_variable_takes_its_default_and_every_other_value_loads(
        self, tmp_path, caplog, key, bad, default
    ):
        with caplog.at_level("WARNING", logger="lilbee.core.config.model"):
            built, error = self._build(
                tmp_path, "top_k = 7\n", **{f"LILBEE_{key.upper()}": bad}, LILBEE_CHUNK_SIZE="321"
            )
        assert error is None
        assert getattr(built, key) == default
        assert (built.top_k, built.chunk_size) == (7, 321)
        assert f"LILBEE_{key.upper()} = {bad!r} is not one of " in caplog.text
        assert f"; {key} uses its default" in caplog.text

    def test_the_default_wins_over_a_retired_key_in_config_toml(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="lilbee.core.config.model"):
            built, error = self._build(tmp_path, "enable_ocr = false\n", LILBEE_OCR="bogus")
        assert error is None
        assert built.ocr is OcrMode.AUTO
        assert "LILBEE_OCR = 'bogus' is not one of auto, all, off; ocr uses its default" in (
            caplog.text
        )

    @pytest.mark.parametrize(
        ("key", "raw", "loaded"),
        [("fts_language", "german", FtsLanguage.GERMAN), ("chat_mode", "CHAT", ChatMode.CHAT)],
    )
    def test_a_value_the_field_normalizes_is_kept(self, tmp_path, caplog, key, raw, loaded):
        with caplog.at_level("WARNING", logger="lilbee.core.config.model"):
            built, error = self._build(tmp_path, "top_k = 7\n", **{f"LILBEE_{key.upper()}": raw})
        assert error is None
        assert getattr(built, key) is loaded
        assert built.top_k == 7
        assert "is not one of" not in caplog.text

    def test_a_bad_value_on_a_variable_that_is_not_an_enum_still_raises(self, tmp_path):
        from lilbee.core.config.model import _build_cfg

        env = {**clean_env(tmp_path), "LILBEE_TOP_K": "many"}
        with mock.patch.dict(os.environ, env, clear=True), pytest.raises(ValidationError):
            _build_cfg()


class TestChatCtxTargetDefault:
    def test_explicit_env_var_wins_over_scaling(self, tmp_path):
        env = clean_env(tmp_path)
        env["LILBEE_CHAT_N_CTX_TARGET"] = "32768"
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch(
                "lilbee.core.system._read_total_memory_bytes",
                return_value=4 * 1024**3,  # tiny host
            ),
        ):
            c = Config()
        assert c.chat_n_ctx_target == 32768

    def test_default_scales_with_host_ram(self, tmp_path):
        env = clean_env(tmp_path)
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch(
                "lilbee.core.system._read_total_memory_bytes",
                return_value=48 * 1024**3,  # 32-64 GiB tier
            ),
        ):
            c = Config()
        assert c.chat_n_ctx_target == 16384

    def test_default_floors_on_small_host(self, tmp_path):
        env = clean_env(tmp_path)
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch(
                "lilbee.core.system._read_total_memory_bytes",
                return_value=8 * 1024**3,
            ),
        ):
            c = Config()
        assert c.chat_n_ctx_target == 8192


class TestEngineKnobValidators:
    """The tri-state engine knobs accept string aliases from env/config."""

    def test_flash_attention_auto_is_none(self):
        with mock.patch.dict(os.environ, {"LILBEE_FLASH_ATTENTION": "auto"}):
            assert Config().flash_attention is None

    def test_n_gpu_layers_cpu_alias_is_zero(self):
        # Call the before-validator directly: the env source coerces int-typed
        # fields before the validator runs, so "cpu" must be exercised here.
        assert Config._parse_n_gpu_layers("cpu") == 0

    def test_n_gpu_layers_auto_alias_is_none(self):
        assert Config._parse_n_gpu_layers("auto") is None


class TestActiveConfigScope:
    """``config_scope`` binds a Config for the block; ``active_config`` reads it."""

    def test_active_config_defaults_to_global(self):
        from lilbee.core.config import active_config, cfg

        assert active_config() is cfg

    def test_config_scope_binds_and_restores(self, tmp_path):
        from lilbee.core.config import active_config, cfg, config_scope

        scoped = cfg.model_copy(update={"data_root": tmp_path})
        with config_scope(scoped):
            assert active_config() is scoped
        assert active_config() is cfg


class TestValidateOcrTimeout:
    """validate_ocr_timeout mirrors the ocr_timeout field's own ge=0.0 bound,
    the rule the CLI already applies through a direct cfg.ocr_timeout assignment."""

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="ocr_timeout"):
            validate_ocr_timeout(-5.0)

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="ocr_timeout"):
            validate_ocr_timeout("abc")

    def test_zero_is_valid(self):
        validate_ocr_timeout(0.0)

    def test_huge_is_valid(self):
        validate_ocr_timeout(1e18)

    def test_none_is_valid(self):
        validate_ocr_timeout(None)

    def test_does_not_mutate_the_active_config(self):
        scoped = cfg.model_copy(update={"ocr_timeout": 42.0})
        with config_scope(scoped):
            with pytest.raises(ValueError):
                validate_ocr_timeout(-5.0)
            assert scoped.ocr_timeout == 42.0


class TestBoolVocabularyMatchesPydantic:
    """parse_bool must accept what pydantic accepts for the same settings object.

    Every other bool field on Config is coerced by pydantic, which takes
    on/off/y/n/t/f as well. The narrower hand-rolled vocabulary made the same
    env spelling mean different things on different fields, and on enable_ocr
    it inverted: the ValueError fell through to bool("off"), which is True.
    """

    @pytest.mark.parametrize("raw", ["on", "y", "t", "true", "1", "yes", "ON", " on "])
    def test_truthy_spellings(self, raw):
        from lilbee.core.config.parsing import parse_bool

        assert parse_bool(raw) is True

    @pytest.mark.parametrize("raw", ["off", "n", "f", "false", "0", "no", "OFF", " off "])
    def test_falsy_spellings(self, raw):
        from lilbee.core.config.parsing import parse_bool

        assert parse_bool(raw) is False

    def test_unknown_spelling_still_raises(self):
        from lilbee.core.config.parsing import parse_bool

        with pytest.raises(ValueError, match="Invalid boolean"):
            parse_bool("maybe")

    @pytest.mark.parametrize(
        ("raw", "expected"), [("off", OcrMode.OFF), ("on", OcrMode.AUTO), ("n", OcrMode.OFF)]
    )
    def test_a_stored_enable_ocr_reads_with_the_same_vocabulary(self, raw, expected):
        """A retired enable_ocr string migrates through parse_bool's vocabulary."""
        from lilbee.core.config.parsing import migrate_ocr_keys

        assert migrate_ocr_keys({"enable_ocr": raw}, "")["ocr"] == expected


class TestCrawlExclusionsMatchWholeSegments:
    """The exclusion patterns are regexes against the whole URL, not globs.

    Bare path prefixes therefore matched longer words and silently dropped
    legitimate content pages from a crawl.
    """

    @staticmethod
    def _excluded(url: str) -> bool:
        import re

        from lilbee.core.config.defaults import _AUTH_EXCLUDE, _ECOMMERCE_EXCLUDE

        return any(re.search(p, url) for p in _AUTH_EXCLUDE + _ECOMMERCE_EXCLUDE)

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/cartography/maps",
            "https://example.com/accounting/gaap",
            "https://example.com/professional-services",
            "https://example.com/registers-of-companies",
        ],
    )
    def test_content_pages_are_not_excluded(self, url):
        assert not self._excluded(url)

    @pytest.mark.parametrize(
        ("url", "excluded"),
        [
            ("https://x.dev/docs?ref=sidebar", False),
            ("https://x.dev/p?share=twitter", False),
            ("https://x.dev/p?utm_source=newsletter", True),
            ("https://x.dev/p?fbclid=abc", True),
            ("https://x.dev/p?replytocom=5", True),
        ],
    )
    def test_only_campaign_tokens_are_treated_as_tracking(self, url, excluded):
        """?ref= and ?share= are ordinary content links on docs and forum
        platforms; dropping one can drop the only URL that reaches a page."""
        import re

        from lilbee.core.config.defaults import _TRACKING_EXCLUDE

        assert any(re.search(p, url) for p in _TRACKING_EXCLUDE) is excluded

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/cart",
            "https://example.com/cart/",
            "https://example.com/cart?step=1",
            "https://example.com/checkout/step1",
            "https://example.com/login",
            "https://example.com/my-account/orders",
        ],
    )
    def test_transactional_and_auth_urls_are_still_excluded(self, url):
        assert self._excluded(url)


class TestFtsLanguage:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("German", "German"),
            ("german", "German"),
            ("GERMAN", "German"),
            ("gErMaN", "German"),
            ("  german  ", "German"),
            ("\tenglish\n", "English"),
            ("tamil", "Tamil"),
        ],
    )
    def test_normalizes_case_and_padding(self, given, expected):
        """Every casing and padding the field accepted before is still accepted."""
        assert Config(fts_language=given).fts_language == expected

    @pytest.mark.parametrize("given", ["Klingon", "", "   ", "en", "Englsh", None, 123])
    def test_rejects_anything_outside_the_value_set(self, given):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Config(fts_language=given)

    def test_rejects_unsupported_language(self):
        # A bad name would otherwise fail FTS index creation quietly and
        # hybrid search would silently degrade to vector-only.
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="fts_language must be one of"):
            Config(fts_language="Klingon")
