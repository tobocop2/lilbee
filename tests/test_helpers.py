"""Tests for CLI helper functions."""

from unittest import mock

import pytest
from rich.console import Console

from lilbee.app.ingest import link_files
from lilbee.cli.helpers import link_paths
from lilbee.core.config import cfg


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all helper tests."""
    snapshot = cfg.model_copy()

    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"

    yield tmp_path

    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


class TestLinkFiles:
    def test_link_single_file(self, tmp_path):
        src = tmp_path / "hello.txt"
        src.write_text("hello")

        result = link_files([src])

        assert result.linked == ["hello.txt"]
        assert result.skipped == []
        dest = cfg.documents_dir / "hello.txt"
        assert dest.is_symlink()  # linked, not copied
        assert dest.resolve() == src.resolve()
        assert dest.read_text() == "hello"

    def test_skip_existing_no_force(self, tmp_path):
        (cfg.documents_dir / "hello.txt").write_text("old")
        src = tmp_path / "hello.txt"
        src.write_text("new")

        result = link_files([src])

        assert result.linked == []
        assert result.skipped == ["hello.txt"]
        assert (cfg.documents_dir / "hello.txt").read_text() == "old"

    def test_relink_same_target_is_idempotent(self, tmp_path):
        src = tmp_path / "hello.txt"
        src.write_text("hello")
        link_files([src])

        result = link_files([src])

        assert result.linked == []
        assert result.skipped == ["hello.txt"]

    def test_overwrite_existing_with_force(self, tmp_path):
        (cfg.documents_dir / "hello.txt").write_text("old")
        src = tmp_path / "hello.txt"
        src.write_text("new")

        result = link_files([src], force=True)

        assert result.linked == ["hello.txt"]
        assert result.skipped == []
        dest = cfg.documents_dir / "hello.txt"
        assert dest.is_symlink()
        assert dest.read_text() == "new"

    def test_link_directory(self, tmp_path):
        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("a")
        (src_dir / "b.txt").write_text("b")

        result = link_files([src_dir])

        assert result.linked == ["mydir"]
        link = cfg.documents_dir / "mydir"
        assert link.is_symlink()  # one link for the whole tree, no per-file copy
        assert (link / "a.txt").read_text() == "a"
        assert (link / "b.txt").read_text() == "b"

    def test_source_already_inside_documents_dir_is_skipped(self, tmp_path):
        inside = cfg.documents_dir / "already.txt"
        inside.write_text("here")

        result = link_files([inside])

        assert result.linked == []
        assert result.skipped == ["already.txt"]
        assert not (cfg.documents_dir / "already.txt").is_symlink()

    def test_empty_paths(self):
        result = link_files([])

        assert result.linked == []
        assert result.skipped == []

    def test_creates_documents_dir(self, tmp_path):
        import shutil

        shutil.rmtree(cfg.documents_dir)

        src = tmp_path / "file.txt"
        src.write_text("content")

        result = link_files([src])

        assert result.linked == ["file.txt"]
        assert cfg.documents_dir.exists()


class TestLinkPaths:
    def test_returns_linked_names(self, tmp_path):
        src = tmp_path / "doc.txt"
        src.write_text("content")
        con = Console()

        linked = link_paths([src], con)

        assert linked == ["doc.txt"]

    def test_prints_warning_for_skipped(self, tmp_path):
        (cfg.documents_dir / "doc.txt").write_text("old")
        src = tmp_path / "doc.txt"
        src.write_text("new")
        con = Console(quiet=True)

        with mock.patch.object(con, "print") as mock_print:
            linked = link_paths([src], con)

        assert linked == []
        mock_print.assert_called_once()
        assert "already exists" in str(mock_print.call_args)
