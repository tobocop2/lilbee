"""A source whose path is a symlink loop gets no link, on every Python version."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from lilbee.app.search import resolve_vault_path
from lilbee.core import settings
from lilbee.core.config import cfg, config_scope
from lilbee.data.ingest.discovery import resolve_source_path_checked
from lilbee.retrieval.query.formatting import SourceRef, source_ref_link


@pytest.fixture()
def loop_world(tmp_path):
    """A documents directory that holds a real symlink loop and one good file."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "good.md").write_text("x", encoding="utf-8")
    try:
        os.symlink(docs / "loop-b", docs / "loop-a")
        os.symlink(docs / "loop-a", docs / "loop-b")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    (tmp_path / "data").mkdir()
    config = cfg.model_copy(
        update={
            "data_root": tmp_path / "data",
            "documents_dir": docs,
            "vault_base": tmp_path,
            "linked_roots": {},
        }
    )
    with config_scope(config):
        settings.set_value(config.data_root, "linked_roots", {})
        yield docs


def _ref(source: str) -> SourceRef:
    return SourceRef(source, source, 1)


def _raising_loop(self: Path, strict: bool = False) -> Path:
    raise RuntimeError(f"Symlink loop from {str(self)!r}")


class TestSymlinkLoopSource:
    def test_the_checked_resolver_returns_none_for_a_real_loop(self, loop_world):
        assert resolve_source_path_checked("loop-a") is None
        assert resolve_source_path_checked("good.md") == (loop_world / "good.md").resolve()

    def test_a_missing_file_still_resolves_and_one_under_a_loop_does_not(self, loop_world):
        assert resolve_source_path_checked("missing.md") == (loop_world / "missing.md").resolve()
        assert resolve_source_path_checked("loop-a/missing.md") is None

    def test_the_sources_link_for_a_real_loop_is_plain_text(self, loop_world):
        assert "file:" not in source_ref_link(_ref("loop-a"))
        assert "file:" in source_ref_link(_ref("good.md"))

    def test_the_vault_path_of_a_real_loop_is_none(self, loop_world):
        assert resolve_vault_path("loop-a") is None
        assert resolve_vault_path("good.md") == "docs/good.md"

    def test_a_resolve_that_raises_on_a_loop_gives_no_checked_path(self, loop_world):
        with mock.patch.object(Path, "resolve", _raising_loop):
            assert resolve_source_path_checked("good.md") is None

    def test_a_resolve_that_leaves_the_looped_link_in_the_path_gives_no_checked_path(
        self, loop_world
    ):
        # Python 3.13 returns the unresolved path for a loop; stand in for it on every version.
        with mock.patch.object(Path, "resolve", lambda self, strict=False: self):
            assert resolve_source_path_checked("loop-a") is None
            assert resolve_source_path_checked("good.md") == loop_world / "good.md"

    def test_a_resolve_that_raises_on_a_loop_gives_no_link(self, loop_world):
        with mock.patch.object(Path, "resolve", _raising_loop):
            assert "file:" not in source_ref_link(_ref("good.md"))

    def test_a_resolve_that_raises_on_a_loop_gives_no_vault_path(self, loop_world):
        assert resolve_vault_path("good.md") == "docs/good.md"
        with mock.patch.object(Path, "resolve", _raising_loop):
            assert resolve_vault_path("good.md") is None
