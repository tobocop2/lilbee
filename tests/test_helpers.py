"""Tests for CLI helper functions."""

from dataclasses import fields, replace
from unittest import mock

import pytest
from rich.console import Console

from lilbee.cli.helpers import copy_files, copy_paths
from lilbee.config import cfg


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths for all helper tests."""
    snapshot = replace(cfg)

    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"

    yield tmp_path

    for f in fields(cfg):
        setattr(cfg, f.name, getattr(snapshot, f.name))


class TestCopyFiles:
    def test_copy_single_file(self, tmp_path):
        src = tmp_path / "hello.txt"
        src.write_text("hello")

        result = copy_files([src])

        assert result.copied == ["hello.txt"]
        assert result.skipped == []
        assert (cfg.documents_dir / "hello.txt").read_text() == "hello"

    def test_skip_existing_no_force(self, tmp_path):
        (cfg.documents_dir / "hello.txt").write_text("old")
        src = tmp_path / "hello.txt"
        src.write_text("new")

        result = copy_files([src])

        assert result.copied == []
        assert result.skipped == ["hello.txt"]
        assert (cfg.documents_dir / "hello.txt").read_text() == "old"

    def test_overwrite_existing_with_force(self, tmp_path):
        (cfg.documents_dir / "hello.txt").write_text("old")
        src = tmp_path / "hello.txt"
        src.write_text("new")

        result = copy_files([src], force=True)

        assert result.copied == ["hello.txt"]
        assert result.skipped == []
        assert (cfg.documents_dir / "hello.txt").read_text() == "new"

    def test_copy_directory(self, tmp_path):
        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("a")
        (src_dir / "b.txt").write_text("b")

        result = copy_files([src_dir])

        assert result.copied == ["mydir"]
        assert (cfg.documents_dir / "mydir" / "a.txt").read_text() == "a"
        assert (cfg.documents_dir / "mydir" / "b.txt").read_text() == "b"

    def test_empty_paths(self):
        result = copy_files([])

        assert result.copied == []
        assert result.skipped == []

    def test_creates_documents_dir(self, tmp_path):
        import shutil

        shutil.rmtree(cfg.documents_dir)

        src = tmp_path / "file.txt"
        src.write_text("content")

        result = copy_files([src])

        assert result.copied == ["file.txt"]
        assert cfg.documents_dir.exists()


class TestCopyPaths:
    def test_returns_copied_names(self, tmp_path):
        src = tmp_path / "doc.txt"
        src.write_text("content")
        con = Console()

        copied = copy_paths([src], con)

        assert copied == ["doc.txt"]

    def test_prints_warning_for_skipped(self, tmp_path):
        (cfg.documents_dir / "doc.txt").write_text("old")
        src = tmp_path / "doc.txt"
        src.write_text("new")
        con = Console(quiet=True)

        with mock.patch.object(con, "print") as mock_print:
            copied = copy_paths([src], con)

        assert copied == []
        mock_print.assert_called_once()
        assert "already exists" in str(mock_print.call_args)
