"""A crawl exclude pattern crawlberg refuses is refused when the setting is saved.

Loading the config never asks crawlberg: the check loads its native library.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest import mock

import pytest
from litestar.testing import TestClient

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
        from lilbee.crawler.backends.crawlberg.links import _python_patterns

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

    def test_config_toml_loads_a_refused_list_without_asking_crawlberg(self, tmp_path):
        asked = StubCrawlberg(refuse=refusing_patterns(PYTHON_ONLY))
        with asked.installed():
            built, warnings = _build_toml(
                tmp_path, f"crawl_exclude_patterns = [{json.dumps(PYTHON_ONLY)}]\n"
            )
        assert built.crawl_exclude_patterns == [PYTHON_ONLY]
        assert warnings == ()
        assert asked.configs == []

    def test_config_toml_keeps_a_pattern_only_crawlberg_compiles(self, tmp_path):
        built, warnings = _build_toml(tmp_path, f"crawl_exclude_patterns = ['{CRAWLBERG_ONLY}']\n")
        assert built.crawl_exclude_patterns == [CRAWLBERG_ONLY]
        assert warnings == ()

    def test_the_mcp_settings_tool_returns_the_pattern_and_the_reason(self, tmp_path, crawlberg):
        cfg.data_root = tmp_path
        before = list(cfg.crawl_exclude_patterns)
        result = settings_set({"crawl_exclude_patterns": [PYTHON_ONLY]})
        assert REASON in result["error"]
        assert cfg.crawl_exclude_patterns == before

    def test_the_variable_loads_a_refused_list_without_asking_crawlberg(self, tmp_path):
        env = {**clean_env(tmp_path), "LILBEE_CRAWL_EXCLUDE_PATTERNS": PYTHON_ONLY}
        asked = StubCrawlberg(refuse=refusing_patterns(PYTHON_ONLY))
        with asked.installed(), mock.patch.dict(os.environ, env, clear=True):
            assert Config().crawl_exclude_patterns == [PYTHON_ONLY]
        assert asked.configs == []

    def test_a_write_of_one_pattern_per_line_names_the_refused_line(self, tmp_path, crawlberg):
        cfg.data_root = tmp_path
        result = settings_set({"crawl_exclude_patterns": f"/ok/\n{PYTHON_ONLY}"})
        assert f"[1] {PYTHON_ONLY!r}: {REASON}" in result["error"]

    def test_a_write_of_the_shipped_list_does_not_ask_crawlberg(self, tmp_path):
        cfg.data_root = tmp_path
        asked = StubCrawlberg(refuse=refusing_patterns(PYTHON_ONLY))
        with asked.installed():
            result = settings_set({"crawl_exclude_patterns": list(DEFAULT_CRAWL_EXCLUDE_PATTERNS)})
        assert "error" not in result
        assert asked.configs == []

    def test_a_refused_write_stores_nothing(self, tmp_path, crawlberg):
        cfg.data_root = tmp_path
        settings_set({"crawl_exclude_patterns": [PYTHON_ONLY]})
        assert not (tmp_path / "config.toml").exists()


class TestImportTime:
    def test_loading_a_config_with_custom_patterns_leaves_crawlberg_unimported(self, tmp_path):
        """A fresh process builds cfg from a config.toml that holds patterns; none imports it."""
        (tmp_path / "config.toml").write_text(
            "crawl_exclude_patterns = ['/custom/', '/other']\n", encoding="utf-8"
        )
        script = tmp_path.parent / f"{tmp_path.name}-probe.py"
        script.write_text(_PROBE, encoding="utf-8")
        env = {**clean_env(tmp_path), "PYTHONPATH": os.pathsep.join(sys.path)}
        done = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "['/custom/', '/other'] []"

    def test_the_check_module_imports_first_in_a_fresh_process(self, tmp_path):
        env = {**clean_env(tmp_path), "PYTHONPATH": os.pathsep.join(sys.path)}
        done = subprocess.run(
            [sys.executable, "-c", "import lilbee.core.crawl_patterns"],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert done.returncode == 0, done.stderr


_PROBE = """
import sys

attempts = []


class Spy:
    def find_spec(self, name, path=None, target=None):
        if name == "crawlberg" or name.startswith("crawlberg."):
            attempts.append(name)
        return None


sys.meta_path.insert(0, Spy())
from lilbee.core.config import cfg

print(cfg.crawl_exclude_patterns, attempts)
"""


def _build_toml(tmp_path, toml: str):
    (tmp_path / "config.toml").write_text(toml, encoding="utf-8")
    with mock.patch.dict(os.environ, clean_env(tmp_path), clear=True):
        return _build_cfg()
