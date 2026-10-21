"""A crawl exclude pattern crawlberg refuses is refused when the setting is saved."""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest
from litestar.testing import TestClient
from pydantic import ValidationError

from lilbee.core.config import DEFAULT_CRAWL_EXCLUDE_PATTERNS, Config, cfg
from lilbee.core.config.model import _build_cfg
from lilbee.core.crawl_patterns import refused_exclude_patterns
from lilbee.mcp_server import settings_set
from tests._crawlberg_stub import StubCrawlberg, refusing_patterns
from tests._sys_modules import inject_modules
from tests.conftest import clean_env

# Python's re module compiles it; crawlberg's regex rules refuse it.
PYTHON_ONLY = r"\Z"
# crawlberg compiles it; Python's re module refuses it.
CRAWLBERG_ONLY = r"/wiki/\p{L}+:"
REASON = "unrecognized escape sequence"


@pytest.fixture()
def crawlberg():
    with StubCrawlberg(refuse=refusing_patterns(PYTHON_ONLY)).installed() as stub:
        yield stub


class TestTheCheck:
    def test_each_refused_pattern_is_named_with_its_index_and_crawlbergs_reason(self, crawlberg):
        assert refused_exclude_patterns(["/ok/", PYTHON_ONLY, "/fine"]) == [(1, REASON)]

    def test_a_pattern_python_refuses_and_crawlberg_compiles_passes(self, crawlberg):
        assert refused_exclude_patterns([CRAWLBERG_ONLY]) == []

    def test_every_refused_pattern_is_reported(self):
        both = refusing_patterns(PYTHON_ONLY, "bad(")
        with StubCrawlberg(refuse=both).installed():
            assert refused_exclude_patterns([PYTHON_ONLY, "ok", "bad("]) == [
                (0, REASON),
                (2, REASON),
            ]

    def test_without_the_crawler_extra_nothing_is_refused(self):
        with inject_modules({"crawlberg": None}):
            assert refused_exclude_patterns([PYTHON_ONLY]) == []

    def test_a_refusal_with_no_error_line_is_reported_whole(self):
        def refuse(config):
            return "invalid_config: invalid exclude_path regex 'x': too big"

        with StubCrawlberg(refuse=refuse).installed():
            assert refused_exclude_patterns(["x"]) == [
                (0, "invalid_config: invalid exclude_path regex 'x': too big")
            ]


class TestTheLogOfExcludedLinks:
    def test_a_pattern_only_crawlberg_compiles_is_left_out_of_the_log_and_the_rest_stay(self):
        from lilbee.crawler.backends.crawlberg.fetcher import _python_patterns

        kept = _python_patterns(["/ok/", CRAWLBERG_ONLY, "/fine"])
        assert [p.pattern for p in kept] == ["/ok/", "/fine"]


class TestStartup:
    def test_the_shipped_list_loads_without_asking_crawlberg(self):
        """Every command builds the config; none of them loads the crawler for its default list."""
        asked = StubCrawlberg(refuse=refusing_patterns("never"))
        with asked.installed():
            Config(crawl_exclude_patterns=list(DEFAULT_CRAWL_EXCLUDE_PATTERNS))
        assert asked.configs == []


class TestEveryEntryPoint:
    """The setting is one field of Config, so each way to save it meets the same check."""

    def test_a_patch_of_the_api_is_refused_with_the_pattern_and_the_reason(self, crawlberg):
        import lilbee.server.auth as auth_mod
        from lilbee.server.app import create_app

        auth_mod.session_manager.disable()
        try:
            client = TestClient(create_app())
            before = list(cfg.crawl_exclude_patterns)
            response = client.patch("/api/config", json={"crawl_exclude_patterns": [PYTHON_ONLY]})
        finally:
            auth_mod.session_manager.cleanup()
        assert response.status_code == 400
        assert REASON in response.text
        assert cfg.crawl_exclude_patterns == before

    def test_a_patch_of_the_api_takes_a_pattern_only_crawlberg_compiles(self, crawlberg):
        import lilbee.server.auth as auth_mod
        from lilbee.server.app import create_app

        auth_mod.session_manager.disable()
        try:
            client = TestClient(create_app())
            response = client.patch(
                "/api/config", json={"crawl_exclude_patterns": [CRAWLBERG_ONLY]}
            )
        finally:
            auth_mod.session_manager.cleanup()
        assert response.status_code == 200
        assert cfg.crawl_exclude_patterns == [CRAWLBERG_ONLY]

    def test_config_toml_drops_the_refused_list_and_names_the_reason(self, tmp_path, crawlberg):
        built, warnings = _build_toml(
            tmp_path, f"crawl_exclude_patterns = [{json.dumps(PYTHON_ONLY)}]\n"
        )
        assert (
            built.crawl_exclude_patterns
            == Config.model_fields["crawl_exclude_patterns"].default_factory()
        )
        assert len(warnings) == 1
        assert "crawl_exclude_patterns" in warnings[0]
        assert REASON in warnings[0]

    def test_config_toml_keeps_a_pattern_only_crawlberg_compiles(self, tmp_path, crawlberg):
        built, warnings = _build_toml(tmp_path, f"crawl_exclude_patterns = ['{CRAWLBERG_ONLY}']\n")
        assert built.crawl_exclude_patterns == [CRAWLBERG_ONLY]
        assert warnings == ()

    def test_the_mcp_settings_tool_returns_the_pattern_and_the_reason(self, tmp_path, crawlberg):
        cfg.data_root = tmp_path
        before = list(cfg.crawl_exclude_patterns)
        result = settings_set({"crawl_exclude_patterns": [PYTHON_ONLY]})
        assert REASON in result["error"]
        assert cfg.crawl_exclude_patterns == before

    def test_the_cli_variable_stops_the_command_with_the_reason(self, tmp_path, crawlberg):
        env = {**clean_env(tmp_path), "LILBEE_CRAWL_EXCLUDE_PATTERNS": PYTHON_ONLY}
        with mock.patch.dict(os.environ, env, clear=True), pytest.raises(Exception, match=REASON):
            Config()

    def test_an_assignment_to_the_setting_is_refused(self, crawlberg):
        probe = cfg.model_copy()
        with pytest.raises(ValidationError, match=REASON):
            probe.crawl_exclude_patterns = [PYTHON_ONLY]


def _build_toml(tmp_path, toml: str):
    (tmp_path / "config.toml").write_text(toml, encoding="utf-8")
    with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
        return _build_cfg()
