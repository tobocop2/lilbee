"""The integration rerun covers the tests that download a model, and nothing else."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from lilbee.catalog import download_model
from tests.integration import conftest as integration_conftest
from tests.integration.conftest import EMBED_ENTRY

# Literals, not the conftest constants: reading those back would assert nothing.
EXPECTED_RERUNS = {"reruns": 2, "reruns_delay": 10}
EXPECTED_TIMEOUT = 180

REPO_ROOT = Path(__file__).resolve().parent.parent
INTEGRATION_DIR = "tests/integration"

# Every module the real collection scopes, as measured. A module that starts
# downloading, or stops, moves this set and the run says so.
RERUN_MODULES = frozenset(
    {
        "tests/integration/test_download_progress_integration.py",
        "tests/integration/test_interactive_integration.py",
        "tests/integration/test_pdf_integration.py",
        "tests/integration/test_pipeline_integration.py",
        "tests/integration/test_rag_integration.py",
        "tests/integration/test_tui_integration.py",
        "tests/integration/test_xet_download_e2e.py",
    }
)

# One test per module that downloads, named in full: a rename lands the id in
# neither list, so the assertion cannot pass by matching nothing.
RERUN_ITEMS = (
    "tests/integration/test_download_progress_integration.py"
    "::TestRealDownloadProgress::test_progress_fires_on_download",
    "tests/integration/test_interactive_integration.py::TestSearch::test_search_finds_exact_keyword",
    "tests/integration/test_pdf_integration.py::TestPdfExtraction::test_scanned_pdf_indexed",
    "tests/integration/test_pipeline_integration.py::TestEmbedder::test_embed_returns_float_vector",
    "tests/integration/test_rag_integration.py"
    "::TestDownloadProgressCallbacks::test_download_fires_progress_callbacks",
    "tests/integration/test_xet_download_e2e.py::test_vision_model_brings_its_projector",
)

# Tests that reach no download: a failure there is the test's own and is
# reported on the first attempt.
NO_RERUN_ITEMS = (
    "tests/integration/test_catalog_integration.py::test_every_pick_has_a_gguf",
    "tests/integration/test_child_guard_binding.py"
    "::test_pdeathsig_reaps_the_child_when_the_parent_dies",
    "tests/integration/test_fleet_integration.py::test_client_embeds_over_real_http",
    "tests/integration/test_pdf_integration.py"
    "::TestTesseractOcrFallback::test_tesseract_extracts_text",
)


def _install_the_embedding_model() -> None:
    """Stand in for the call a new download helper would make."""
    download_model(EMBED_ENTRY)


def _a_new_download_fixture() -> None:
    """Stand in for a new fixture that reaches the download through a helper."""
    _install_the_embedding_model()


def _a_test_that_searches() -> None:
    """Stand in for an integration test that downloads nothing itself."""


def _a_fixture_that_downloads_nothing() -> None:
    """Stand in for a fixture that only prepares local state."""


class _FixtureDef:
    """A resolved fixture, reduced to what the scoping predicate reads."""

    def __init__(self, func: Callable[[], None]) -> None:
        self.func = func


class _FixtureInfo:
    """A resolved fixture closure, reduced to what the scoping predicate reads."""

    def __init__(self, funcs: tuple[Callable[[], None], ...]) -> None:
        self.name2fixturedefs = {func.__name__: (_FixtureDef(func),) for func in funcs}


class _CollectedItem:
    """A collected test item, reduced to what the collection hook reads."""

    def __init__(self, function: Callable[[], None], *fixtures: Callable[[], None]) -> None:
        self.function = function
        self._fixtureinfo = _FixtureInfo(fixtures)
        self.markers: list[pytest.MarkDecorator] = []

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        return None

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.markers.append(marker)

    def marker_names(self) -> list[str]:
        return [marker.name for marker in self.markers]

    def marker(self, name: str) -> pytest.MarkDecorator:
        return next(marker for marker in self.markers if marker.name == name)


def _collect(marker_expression: str) -> list[str]:
    """The node ids the real integration collection puts on one side of a marker."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            INTEGRATION_DIR,
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            marker_expression,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [line.strip() for line in result.stdout.splitlines() if "::" in line]


@pytest.fixture(scope="module")
def rerun_items() -> list[str]:
    """The node ids the real collection scopes for rerun."""
    return _collect("flaky")


@pytest.fixture(scope="module")
def plain_items() -> list[str]:
    """The node ids the real collection leaves on their first attempt."""
    return _collect("not flaky")


def test_a_fixture_that_downloads_makes_its_tests_rerun() -> None:
    item = _CollectedItem(_a_test_that_searches, _a_new_download_fixture)

    integration_conftest.pytest_collection_modifyitems([item])

    assert "flaky" in item.marker_names()
    assert item.marker("flaky").kwargs == EXPECTED_RERUNS


def test_a_test_that_downloads_is_rerun_without_a_fixture() -> None:
    """The download is what scopes the rerun, not the way the test reaches it."""
    item = _CollectedItem(_install_the_embedding_model)

    integration_conftest.pytest_collection_modifyitems([item])

    assert "flaky" in item.marker_names()


def test_a_test_that_downloads_nothing_is_not_rerun() -> None:
    """A genuine failure is reported on its first attempt, not retried."""
    item = _CollectedItem(_a_test_that_searches, _a_fixture_that_downloads_nothing)

    integration_conftest.pytest_collection_modifyitems([item])

    assert "flaky" not in item.marker_names()


def test_every_test_still_gets_the_integration_timeout() -> None:
    """The rerun marker is added beside the timeout default, not in place of it."""
    items = [
        _CollectedItem(_a_test_that_searches, _a_new_download_fixture),
        _CollectedItem(_a_test_that_searches),
    ]

    integration_conftest.pytest_collection_modifyitems(items)

    assert [item.marker("timeout").args for item in items] == [
        (EXPECTED_TIMEOUT,),
        (EXPECTED_TIMEOUT,),
    ]


def test_the_rerun_covers_every_module_that_downloads(rerun_items: list[str]) -> None:
    assert {node_id.split("::")[0] for node_id in rerun_items} == RERUN_MODULES


def test_every_download_test_is_rerun(rerun_items: list[str]) -> None:
    assert [node_id for node_id in RERUN_ITEMS if node_id not in rerun_items] == []


def test_no_test_without_a_download_is_rerun(plain_items: list[str]) -> None:
    assert [node_id for node_id in NO_RERUN_ITEMS if node_id not in plain_items] == []
