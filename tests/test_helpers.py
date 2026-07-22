"""Tests for CLI helper functions."""

import sys
from unittest import mock

import pytest
from rich.console import Console

from lilbee.app.ingest import link_files, symlinks_supported
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
        if symlinks_supported():
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
        if symlinks_supported():
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
        if symlinks_supported():
            assert link.is_symlink()  # one link for the whole tree, no per-file copy
        assert (link / "a.txt").read_text() == "a"
        assert (link / "b.txt").read_text() == "b"

    @pytest.mark.skipif(not symlinks_supported(), reason="symlinks unavailable on this host")
    def test_dangling_link_auto_relocates_without_force(self, tmp_path):
        import shutil

        # A prior add's link whose target moved away is dangling; re-adding the
        # source at its new path must relink silently, not demand --force.
        a = tmp_path / "a" / "corpus"
        a.mkdir(parents=True)
        (a / "x.txt").write_text("content")
        link_files([a])
        assert (cfg.documents_dir / "corpus").resolve() == a.resolve()

        b = tmp_path / "b" / "corpus"
        b.parent.mkdir()
        shutil.move(str(a), str(b))  # move breaks the link, keeping the basename
        assert (cfg.documents_dir / "corpus").is_symlink()
        assert not (cfg.documents_dir / "corpus").exists()  # dangling

        result = link_files([b], force=False)

        assert result.linked == ["corpus"]
        assert result.skipped == []
        assert (cfg.documents_dir / "corpus").resolve() == b.resolve()

    @pytest.mark.skipif(not symlinks_supported(), reason="symlinks unavailable on this host")
    def test_force_does_not_clobber_a_real_directory(self, tmp_path):
        # A real directory occupying the name is never replaced by a link on force.
        (cfg.documents_dir / "corpus").mkdir()
        (cfg.documents_dir / "corpus" / "keep.txt").write_text("keep")
        src = tmp_path / "corpus"
        src.mkdir()
        (src / "x.txt").write_text("x")

        result = link_files([src], force=True)

        assert result.skipped == ["corpus"]
        assert (cfg.documents_dir / "corpus" / "keep.txt").read_text() == "keep"

    def test_source_already_inside_documents_dir_is_skipped(self, tmp_path):
        inside = cfg.documents_dir / "already.txt"
        inside.write_text("here")

        result = link_files([inside])

        assert result.linked == []
        assert result.skipped == ["already.txt"]
        assert not (cfg.documents_dir / "already.txt").is_symlink()

    def test_falls_back_to_copy_when_symlinks_unsupported(self, tmp_path, monkeypatch):
        # Unprivileged Windows can't symlink; add must still work by copying.
        from lilbee.app import ingest

        monkeypatch.setattr(ingest, "symlinks_supported", lambda: False)
        src = tmp_path / "hello.txt"
        src.write_text("hello")

        result = link_files([src])

        assert result.linked == ["hello.txt"]
        dest = cfg.documents_dir / "hello.txt"
        assert not dest.is_symlink()  # copied, not linked
        assert dest.is_file()
        assert dest.read_text() == "hello"

    def test_copies_a_file_when_no_link_can_be_made(self, tmp_path, monkeypatch):
        # Last-resort fallback: neither symlink nor hard link available -> copy.
        from lilbee.app import ingest

        def _no_link(*_a, **_k):
            raise OSError("cross-device link")

        monkeypatch.setattr(ingest, "symlinks_supported", lambda: False)
        monkeypatch.setattr(ingest.os, "link", _no_link)  # hard link fails -> copy
        src = tmp_path / "f.txt"
        src.write_text("data")

        result = link_files([src])

        assert result.linked == ["f.txt"]
        dest = cfg.documents_dir / "f.txt"
        assert not ingest.is_link(dest)
        assert dest.read_text() == "data"

    @pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
    def test_directory_junction_on_unprivileged_windows(self, tmp_path, monkeypatch):
        # No symlink privilege on Windows: a directory links via a junction (no
        # copy), which discovery follows just like a symlink.
        from lilbee.app import ingest
        from lilbee.core.system import is_link
        from lilbee.data.ingest import discover_files

        monkeypatch.setattr(ingest, "symlinks_supported", lambda: False)
        src = tmp_path / "corpus"
        src.mkdir()
        (src / "a.txt").write_text("a")

        result = link_files([src])

        assert result.linked == ["corpus"]
        dest = cfg.documents_dir / "corpus"
        assert is_link(dest)  # a junction reads as a link...
        assert not dest.is_symlink()  # ...but is not a symlink
        assert (dest / "a.txt").read_text() == "a"  # junction redirects transparently
        assert "corpus/a.txt" in discover_files()  # discovery follows it

    def test_copy_fallback_for_directory_filters_ignored_dirs(self, tmp_path, monkeypatch):
        from lilbee.app import ingest

        monkeypatch.setattr(ingest, "symlinks_supported", lambda: False)
        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("a")
        (src_dir / ".git").mkdir()
        (src_dir / ".git" / "cfg").write_text("x")

        result = link_files([src_dir])

        assert result.linked == ["mydir"]
        dest = cfg.documents_dir / "mydir"
        assert not dest.is_symlink()
        assert (dest / "a.txt").read_text() == "a"
        assert not (dest / ".git").exists()  # ignored dirs filtered at copy time

    @pytest.mark.skipif(sys.platform == "win32", reason="backslash is a separator on Windows")
    def test_rejects_a_name_containing_a_separator(self, tmp_path):
        # A basename with a path separator is skipped, never placed (traversal guard).
        weird = tmp_path / "a\\b"  # backslash is a valid POSIX filename character
        weird.write_text("x")

        result = link_files([weird])

        assert result.skipped == ["a\\b"]
        assert not (cfg.documents_dir / "a\\b").exists()

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
