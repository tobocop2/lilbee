"""Nested .lilbeeignore matching, discovery pruning, and removal of newly-ignored sources."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config import cfg


@pytest.fixture
def isolated_env(tmp_path):
    """Redirect the corpus paths at a tmp tree and restore cfg afterwards."""
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.linked_roots = {}
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _write(path: Path, text: str = "content") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestIgnoreRules:
    def test_root_file_excludes_matching_file(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules

        base = isolated_env / "repo"
        _write(base / IGNORE_FILENAME, "*.min.js\n")
        target = _write(base / "app.min.js")

        rules = IgnoreRules()
        assert rules.excludes_path(target, base=base)
        assert not rules.excludes_path(_write(base / "app.js"), base=base)

    def test_nested_file_scopes_to_its_own_subtree(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules

        base = isolated_env / "repo"
        _write(base / "vendor" / IGNORE_FILENAME, "*.js\n")
        inside = _write(base / "vendor" / "lib.js")
        outside = _write(base / "src" / "lib.js")

        rules = IgnoreRules()
        assert rules.excludes_path(inside, base=base)
        assert not rules.excludes_path(outside, base=base)

    def test_deeper_file_overrides_shallower(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules

        base = isolated_env / "repo"
        _write(base / IGNORE_FILENAME, "*.md\n")
        _write(base / "docs" / IGNORE_FILENAME, "!keep.md\n")
        kept = _write(base / "docs" / "keep.md")
        dropped = _write(base / "docs" / "other.md")

        rules = IgnoreRules()
        assert not rules.excludes_path(kept, base=base)
        assert rules.excludes_path(dropped, base=base)

    def test_directory_pattern_excludes_everything_beneath_it(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules

        base = isolated_env / "repo"
        _write(base / IGNORE_FILENAME, "testdata/\n")
        buried = _write(base / "testdata" / "deep" / "fixture.md")

        rules = IgnoreRules()
        # The query agrees with the walk for a file the walk never enumerated,
        # because its pruned parent directory is tested on the way down.
        assert rules.excludes_path(buried, base=base)
        assert rules.excludes_entry(base / "testdata", base=base, is_dir=True)

    def test_negation_cannot_resurrect_a_file_under_a_pruned_directory(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules

        base = isolated_env / "repo"
        _write(base / IGNORE_FILENAME, "testdata/\n!testdata/keep.md\n")
        buried = _write(base / "testdata" / "keep.md")

        rules = IgnoreRules()
        assert rules.excludes_path(buried, base=base)

    def test_corpus_layer_applies_to_every_root(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules

        _write(isolated_env / IGNORE_FILENAME, "*.min.js\n")
        base = isolated_env / "repo"
        target = _write(base / "app.min.js")

        rules = IgnoreRules.for_corpus()
        assert rules.excludes_path(target, base=base)

    def test_nested_layer_overrides_the_corpus_layer(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules

        _write(isolated_env / IGNORE_FILENAME, "*.min.js\n")
        base = isolated_env / "repo"
        _write(base / IGNORE_FILENAME, "!app.min.js\n")
        target = _write(base / "app.min.js")

        rules = IgnoreRules.for_corpus()
        assert not rules.excludes_path(target, base=base)

    def test_blank_lines_and_comments_are_ignored(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules

        base = isolated_env / "repo"
        _write(base / IGNORE_FILENAME, "# a comment\n\n*.log\n")
        assert IgnoreRules().excludes_path(_write(base / "run.log"), base=base)

    def test_a_comment_only_file_compiles_to_no_layer(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules, _load_spec

        # The file lilbee init scaffolds is all comments. pathspec still builds a
        # pattern per line, so keeping the spec would charge every walked file a
        # lookup against something that can never match.
        base = isolated_env / "repo"
        _write(base / IGNORE_FILENAME, "# just a comment\n\n   \n")
        assert _load_spec(base / IGNORE_FILENAME) is None
        assert not IgnoreRules().excludes_path(_write(base / "note.md"), base=base)

    def test_missing_files_exclude_nothing(self, isolated_env):
        from lilbee.data.ingest.ignore import IgnoreRules

        base = isolated_env / "repo"
        assert not IgnoreRules().excludes_path(_write(base / "app.js"), base=base)

    def test_unreadable_ignore_file_is_skipped(self, isolated_env, monkeypatch):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules

        base = isolated_env / "repo"
        _write(base / IGNORE_FILENAME, "*.md\n")
        target = _write(base / "note.md")

        def _boom(*args, **kwargs):
            raise OSError("unreadable")

        monkeypatch.setattr(Path, "read_text", _boom)
        assert not IgnoreRules().excludes_path(target, base=base)


class TestDiscoveryHonoursIgnoreFiles:
    def test_owned_tree_drops_ignored_files(self, isolated_env):
        from lilbee.data.ingest import discover_files
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        _write(cfg.documents_dir / IGNORE_FILENAME, "drafts/\n")
        _write(cfg.documents_dir / "drafts" / "wip.md")
        _write(cfg.documents_dir / "final.md")

        found = discover_files()
        assert set(found) == {"final.md"}

    def test_registered_root_keys_survive_filtering(self, isolated_env):
        from lilbee.data.ingest import discover_files
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        root = isolated_env / "repo"
        _write(root / IGNORE_FILENAME, "*.min.js\n")
        _write(root / "app.min.js")
        _write(root / "src" / "app.js")
        cfg.linked_roots = {"repo": str(root)}

        found = discover_files()
        assert set(found) == {"repo/src/app.js"}

    def test_ignore_file_itself_never_ingests(self, isolated_env):
        from lilbee.data.ingest import discover_files
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        _write(cfg.documents_dir / IGNORE_FILENAME, "*.log\n")
        _write(cfg.documents_dir / "note.md")

        assert set(discover_files()) == {"note.md"}

    def test_single_file_root_is_not_filtered(self, isolated_env):
        from lilbee.data.ingest import discover_files
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        _write(isolated_env / IGNORE_FILENAME, "*.md\n")
        target = _write(isolated_env / "loose" / "note.md")
        cfg.linked_roots = {"note.md": str(target)}

        assert set(discover_files()) == {"note.md"}

    def test_a_single_file_root_has_no_walked_base(self, isolated_env):
        from lilbee.data.ingest.discovery import resolve_source_root

        cfg.linked_roots = {"note.md": str(isolated_env / "note.md")}
        assert resolve_source_root("note.md") is None
        base, path = resolve_source_root("loose.md")
        assert base == cfg.documents_dir
        assert path == cfg.documents_dir / "loose.md"

    def test_pruned_directory_is_never_walked(self, isolated_env, monkeypatch):
        from lilbee.data.ingest import discover_files
        from lilbee.data.ingest.ignore import IGNORE_FILENAME

        _write(cfg.documents_dir / IGNORE_FILENAME, "heavy/\n")
        _write(cfg.documents_dir / "heavy" / "a.md")
        _write(cfg.documents_dir / "keep.md")

        seen: list[str] = []
        real_walk = __import__("os").walk

        def _spy(top, *args, **kwargs):
            for root, dirs, files in real_walk(top, *args, **kwargs):
                seen.append(str(root))
                yield root, dirs, files

        monkeypatch.setattr("lilbee.data.ingest.discovery.os.walk", _spy)
        assert set(discover_files()) == {"keep.md"}
        assert not any("heavy" in entry for entry in seen)


class TestReconcilesIndexAgainstPatterns:
    """A pattern added after ingest must drop what it now excludes, and only that."""

    def _src(self, filename: str, source_type=None):
        from lilbee.data.store import SourceType

        return {
            "filename": filename,
            "file_hash": "",
            "ingested_at": "",
            "chunk_count": 1,
            "source_type": source_type or SourceType.DOCUMENT,
        }

    def test_newly_ignored_source_is_selected_and_others_are_not(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules
        from lilbee.data.ingest.pipeline import _ignored_sources

        _write(cfg.documents_dir / IGNORE_FILENAME, "*.min.js\n")
        _write(cfg.documents_dir / "app.min.js")

        sources = [self._src("app.min.js"), self._src("keep.md"), self._src("deleted.md")]
        assert _ignored_sources(sources, IgnoreRules.for_corpus()) == ["app.min.js"]

    def test_imported_source_is_never_selected(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules
        from lilbee.data.ingest.pipeline import _ignored_sources
        from lilbee.data.store import SourceType

        # An import has no file on disk, so a pattern has nothing to match it
        # against; resolving its key would point at a path it does not own.
        _write(cfg.documents_dir / IGNORE_FILENAME, "*.pdf\n")
        sources = [self._src("shared.pdf", SourceType.IMPORTED)]
        assert _ignored_sources(sources, IgnoreRules.for_corpus()) == []

    def test_source_under_a_pruned_directory_is_selected(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules
        from lilbee.data.ingest.pipeline import _ignored_sources

        _write(cfg.documents_dir / IGNORE_FILENAME, "testdata/\n")
        _write(cfg.documents_dir / "testdata" / "deep" / "fixture.md")

        sources = [self._src("testdata/deep/fixture.md")]
        assert _ignored_sources(sources, IgnoreRules.for_corpus()) == ["testdata/deep/fixture.md"]

    def test_single_file_root_is_never_selected(self, isolated_env):
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules
        from lilbee.data.ingest.pipeline import _ignored_sources

        _write(isolated_env / IGNORE_FILENAME, "*.md\n")
        target = _write(isolated_env / "loose" / "note.md")
        cfg.linked_roots = {"note.md": str(target)}

        assert _ignored_sources([self._src("note.md")], IgnoreRules.for_corpus()) == []

    def test_removal_writes_no_skip_marker(self, isolated_env, monkeypatch):
        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.ignore import IGNORE_FILENAME, IgnoreRules
        from lilbee.data.ingest.skip_marker import load_skip_markers

        _write(cfg.documents_dir / IGNORE_FILENAME, "*.min.js\n")
        _write(cfg.documents_dir / "app.min.js")

        class _Store:
            def remove_documents(self, names):
                from lilbee.data.store.types import RemoveResult

                return RemoveResult(removed=list(names), not_found=[])

        monkeypatch.setattr(pipeline, "get_services", lambda: type("S", (), {"store": _Store()})())
        monkeypatch.setattr(
            "lilbee.app.ingest.forget_removed_from_wiki_index", lambda removed: None
        )

        removed = pipeline._forget_ignored([self._src("app.min.js")], IgnoreRules.for_corpus())
        assert removed == ["app.min.js"]
        # A marker would outlive the pattern and hold the file out after the
        # pattern was deleted; the ignore file is the only durable statement.
        assert load_skip_markers(cfg.data_root) == {}

    def test_nothing_ignored_touches_no_store(self, isolated_env, monkeypatch):
        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.ignore import IgnoreRules

        def _explode():
            raise AssertionError("the store must not be read when nothing is excluded")

        monkeypatch.setattr(pipeline, "get_services", _explode)
        assert pipeline._forget_ignored([self._src("keep.md")], IgnoreRules()) == []
