"""Tests for wiki shared utilities."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.core.text import clean_label_for_display, is_valid_label, make_slug
from lilbee.wiki.shared import (
    SUBDIR_TO_TYPE,
    WikiPageType,
    WikiSubdir,
    parse_frontmatter,
)


class TestSubdirToType:
    def test_all_expected_keys(self):
        assert set(SUBDIR_TO_TYPE) == {
            WikiSubdir.SUMMARIES,
            WikiSubdir.SYNTHESIS,
            WikiSubdir.CONCEPTS,
            WikiSubdir.ENTITIES,
            WikiSubdir.DRAFTS,
            WikiSubdir.ARCHIVE,
        }

    def test_values(self):
        assert SUBDIR_TO_TYPE[WikiSubdir.SUMMARIES] is WikiPageType.SUMMARY
        assert SUBDIR_TO_TYPE[WikiSubdir.SYNTHESIS] is WikiPageType.SYNTHESIS
        assert SUBDIR_TO_TYPE[WikiSubdir.CONCEPTS] is WikiPageType.CONCEPT
        assert SUBDIR_TO_TYPE[WikiSubdir.ENTITIES] is WikiPageType.ENTITY
        assert SUBDIR_TO_TYPE[WikiSubdir.DRAFTS] is WikiPageType.DRAFT
        assert SUBDIR_TO_TYPE[WikiSubdir.ARCHIVE] is WikiPageType.ARCHIVE


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        text = "---\ntitle: Hello\ngenerated_at: '2026-01-01'\n---\nBody"
        result = parse_frontmatter(text)
        assert result["title"] == "Hello"
        assert result["generated_at"] == "2026-01-01"

    def test_no_frontmatter(self):
        assert parse_frontmatter("Just text") == {}

    def test_unclosed_frontmatter(self):
        assert parse_frontmatter("---\ntitle: Hello\nNo close") == {}

    def test_multiple_sources(self):
        text = "---\nsources: [a.txt, b.txt, c.txt]\n---\n"
        result = parse_frontmatter(text)
        assert result["sources"] == ["a.txt", "b.txt", "c.txt"]

    def test_empty_string(self):
        assert parse_frontmatter("") == {}

    def test_invalid_yaml_returns_empty(self):
        text = "---\n: [unbalanced\n---\nBody"
        assert parse_frontmatter(text) == {}

    def test_bare_date_parsed_as_date_object(self):
        text = "---\ngenerated_at: 2026-01-01\n---\n"
        result = parse_frontmatter(text)
        import datetime

        assert isinstance(result["generated_at"], datetime.date)

    def test_triple_dash_inside_yaml_not_confused_for_delimiter(self):
        text = "---\ntitle: Hello\ndesc: 'has --- inside'\n---\nBody"
        result = parse_frontmatter(text)
        assert result["title"] == "Hello"
        assert "---" in result["desc"]


class TestMakeSlug:
    def test_spaces_to_dashes(self):
        assert make_slug("gradual typing") == "gradual-typing"

    def test_slashes_to_double_dashes(self):
        assert make_slug("path/to/concept") == "path--to--concept"

    def test_lowercase(self):
        assert make_slug("Python Types") == "python-types"

    def test_strips_special_characters(self):
        assert make_slug("hello! world?") == "hello-world"

    def test_preserves_hyphens(self):
        assert make_slug("well-known") == "well-known"

    def test_empty_string(self):
        assert make_slug("") == ""

    def test_strips_leading_and_trailing_hyphens(self):
        assert make_slug("-well-known-") == "well-known"

    def test_table_delimited_label_reduces_to_body(self):
        # Even if the sanity gate let this through, the slug should not
        # contain the leading double hyphen that bit bb-8b7s.
        assert make_slug("| | Body") == "body"

    def test_preserves_internal_double_hyphen_path_encoding(self):
        # ``/`` encodes as ``--`` so path-like labels round-trip.
        # Trimming only targets leading/trailing runs.
        assert make_slug("path/to/concept") == "path--to--concept"

    def test_punctuation_only_returns_empty(self):
        assert make_slug("!!!") == ""

    def test_hyphens_only_returns_empty(self):
        # Separate branch from general punctuation: hyphens survive the
        # slug-clean regex and are removed only by the leading/trailing
        # strip. ``"---"`` must still collapse to ``""`` so the
        # ``if not slug: return None`` guard in the extractor fires.
        assert make_slug("---") == ""


class TestIsValidLabel:
    def test_accepts_ordinary_label(self):
        assert is_valid_label("Chevrolet Caprice") is True

    def test_accepts_short_but_nonempty_acronym_boundary(self):
        # "C++" length 3, alnum ratio 1/3 < 0.5 -> reject.
        # Document the boundary so future edits know this is intentional.
        assert is_valid_label("C++") is False

    def test_rejects_under_min_length(self):
        assert is_valid_label("ab") is False

    def test_rejects_structural_pipe(self):
        assert is_valid_label("| | Body") is False

    def test_rejects_structural_hash(self):
        assert is_valid_label("#heading") is False

    def test_rejects_structural_angle(self):
        assert is_valid_label(">>>>") is False

    def test_rejects_structural_char_mid_label(self):
        # Leading char is alpha so the first-char gate passes; the
        # structural-char membership check (line 161) is what rejects
        # ``foo|bar``. Guards against a label that sneaks past the
        # first-char filter but still carries markdown-table noise.
        assert is_valid_label("foo|bar") is False

    def test_rejects_leading_digit(self):
        # The bb-8b7s "158 vehicle" pattern: page numbers prepended to entities.
        assert is_valid_label("158 vehicle") is False

    def test_rejects_paren_prefix(self):
        # bb-8b7s: "(7.0 l)" slipped past the original digit-only first-char
        # guard because "(" is not a digit; slug-cleanup then stripped the
        # paren and wrote "70-l.md" on disk.
        assert is_valid_label("(7.0 l)") is False

    def test_rejects_hyphen_prefix(self):
        # bb-8b7s: "-answers" from markdown bracket-link extraction residue.
        assert is_valid_label("-answers") is False

    def test_rejects_low_alnum_ratio(self):
        assert is_valid_label("---!!") is False

    def test_accepts_hyphenated_label(self):
        assert is_valid_label("E-mail") is True

    @pytest.mark.parametrize(
        ("label", "valid"),
        [
            ("Model\nController", False),
            ("Model\rController", False),
            ("Chevrolet\u00a0Caprice", True),
            ("Chevrolet\u2009Caprice", True),
        ],
        ids=["newline", "carriage-return", "non-breaking-space", "thin-space"],
    )
    def test_only_a_line_break_disqualifies_a_label(self, label: str, valid: bool):
        """A label crossing a line break truncates the single-line marker comment
        it gets interpolated into, leaving a file no reader classifies as a
        placeholder. Exotic spaces do not split a line, and PDF text is full of
        them, so rejecting those would drop real entities."""
        assert is_valid_label(label) is valid

    def test_strips_whitespace_before_checking(self):
        assert is_valid_label("  ab  ") is False
        assert is_valid_label("  Chevrolet  ") is True


class TestCleanLabelForDisplay:
    def test_strips_pipes(self):
        assert clean_label_for_display("| | designer") == "designer"

    def test_collapses_whitespace(self):
        assert clean_label_for_display("Chevrolet   Caprice") == "Chevrolet Caprice"

    def test_preserves_proper_noun_case(self):
        assert clean_label_for_display("iPhone") == "iPhone"

    def test_returns_empty_for_all_structural(self):
        assert clean_label_for_display("|#>") == ""

    def test_leaves_ordinary_label_unchanged(self):
        assert clean_label_for_display("brake pads") == "brake pads"


class TestSlugWhitespaceHandling:
    """The docstring promises whitespace maps to single hyphens."""

    def test_a_double_space_does_not_collide_with_a_slash(self):
        """`--` is the reserved encoding for `/`, so a run of spaces producing
        it made two different entities share one wiki page, and whichever was
        written second silently overwrote the first."""
        from lilbee.core.text import make_slug

        assert make_slug("Chevrolet  Caprice") != make_slug("Chevrolet/Caprice")
        assert make_slug("Chevrolet  Caprice") == "chevrolet-caprice"
        assert make_slug("Chevrolet/Caprice") == "chevrolet--caprice"

    @pytest.mark.parametrize("label", ["Chevrolet\tCaprice", "Chevrolet\nCaprice"])
    def test_other_whitespace_becomes_a_hyphen_not_nothing(self, label):
        """Tabs and newlines were deleted, welding two words into one token."""
        from lilbee.core.text import make_slug

        assert make_slug(label) == "chevrolet-caprice"

    def test_leading_and_trailing_whitespace_is_trimmed(self):
        from lilbee.core.text import make_slug

        assert make_slug("  Caprice  ") == "caprice"


class TestAtomicWriteText:
    def test_replaces_the_previous_file(self, tmp_path):
        from lilbee.wiki.shared import atomic_write_text

        path = tmp_path / "pages" / "brakes.md"
        atomic_write_text(path, "first")
        atomic_write_text(path, "second")
        assert path.read_text() == "second"
        assert list(path.parent.iterdir()) == [path]

    def test_failed_replace_keeps_the_old_file_and_leaves_no_temp(self, tmp_path, monkeypatch):
        import os

        from lilbee.wiki.shared import atomic_write_text

        def _disk_full(*_args, **_kwargs):
            raise OSError("disk full")

        path = tmp_path / "brakes.md"
        atomic_write_text(path, "original")
        monkeypatch.setattr(os, "replace", _disk_full)

        with pytest.raises(OSError, match="disk full"):
            atomic_write_text(path, "replacement")

        assert path.read_text() == "original"
        assert list(tmp_path.iterdir()) == [path]


def _wiki_lock_held() -> bool:
    """Whether the wiki build mutex is held, probed from another thread.

    The mutex is re-entrant, so it reads as free to the thread holding it and
    carries no ``locked()`` on this Python; a non-blocking acquire off-thread
    answers for both lock kinds.
    """
    from lilbee.wiki.shared import WIKI_BUILD_LOCK

    free: list[bool] = []

    def probe() -> None:
        acquired = WIKI_BUILD_LOCK.acquire(blocking=False)
        if acquired:
            WIKI_BUILD_LOCK.release()
        free.append(acquired)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return not free[0]


class TestPendingMarkerPredicate:
    """One definition of a PENDING marker for every reader.

    persistence used a strict prefix test and drafts a whitespace-tolerant
    regex, so the two disagreed about the same file.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("<!-- PENDING: batch parse failed -->", True),
            ("<!--  PENDING:  batch  parse  failed -->", True),
            ("<!-- PENDING: concept slug collision with x -->", True),
            ("<!-- DRIFT: 50% content changed -->", False),
            ("Wrote <!-- PENDING: batch parse failed --> then retried", False),
            ("<!-- PENDING: batch parse failed for source q3>q2.pdf -->", True),
            ("<!-- PENDING: batch parse failed", False),
            ("---\ntitle: Brakes\n---\n", False),
            ("", False),
        ],
        ids=[
            "parse",
            "spaced",
            "collision",
            "drift",
            "quoted-mid-line",
            "source-name-with-angle-bracket",
            "truncated",
            "content",
            "empty",
        ],
    )
    def test_classifies_a_leading_marker_line(self, text: str, expected: bool):
        from lilbee.wiki.shared import is_pending_marker_text

        assert is_pending_marker_text(text) is expected

    def test_a_marker_further_down_is_not_a_marker(self):
        """Markers are written as the first line, so a body quoting one is
        review content, not a placeholder."""
        from lilbee.wiki.shared import is_pending_marker_text

        assert is_pending_marker_text("# Page\n\n<!-- PENDING: batch parse failed -->") is False


class TestFrontmatterWithMarkers:
    """A draft carries marker comments above its frontmatter.

    Every reader that parses the raw file (browse.build_page_info, and so the
    drafts listing) would otherwise see no frontmatter at all: no title, no
    source count, no generated_at.
    """

    _PAGE = "---\ntitle: Brakes\nsources: [a.md]\n---\n# Brakes\n"

    @pytest.mark.parametrize(
        "prefix",
        [
            "",
            "<!-- origin: concepts -->\n\n",
            "<!-- DRIFT: 50% content changed; origin: concepts -->\n\n",
            "<!-- PENDING: concept slug collision with source wiki/drafts/b.md -->\n\n"
            "<!-- DRIFT: 50% content changed; origin: concepts; source: 11dd481a -->\n\n",
        ],
        ids=["plain", "origin", "drift", "stacked"],
    )
    def test_frontmatter_survives_a_leading_marker_run(self, prefix: str):
        assert parse_frontmatter(prefix + self._PAGE)["title"] == "Brakes"

    def test_a_body_with_no_frontmatter_still_parses_as_empty(self):
        assert parse_frontmatter("<!-- origin: concepts -->\n\n# Brakes\n") == {}

    def test_blank_lines_alone_do_not_open_the_frontmatter_scan(self):
        """Blank lines are only consumed after a marker, so an unmarked page
        still has to carry its delimiter on line zero."""
        assert parse_frontmatter("\n\n---\ntitle: Brakes\n---\n") == {}


class TestWikiBuildMutex:
    """Every mutating wiki entry point holds one process-wide lock while it
    writes, so an MCP build, an HTTP prune, a CLI synthesize and a TUI accept
    cannot interleave over the same pages, index.md, and log.md."""

    def test_the_mutex_is_re_entrant(self):
        """A writer holding it calls helpers that take it again (a lint that
        records a log entry); a plain lock would deadlock the whole process."""
        from lilbee.wiki.shared import WIKI_BUILD_LOCK

        with WIKI_BUILD_LOCK:
            assert WIKI_BUILD_LOCK.acquire(blocking=False)
            WIKI_BUILD_LOCK.release()

    def test_lint_holds_the_lock_while_appending_to_the_log(self, monkeypatch, tmp_path):
        """`wiki lint` from CLI, HTTP and MCP all reach log.md unlocked, racing a
        build's own append over a file whose first writer creates it."""
        from lilbee.wiki import lint as lint_mod

        held: list[bool] = []
        monkeypatch.setattr(
            lint_mod, "append_wiki_log", lambda *a, **kw: held.append(_wiki_lock_held())
        )
        config = cfg.model_copy(update={"data_root": tmp_path, "wiki_dir": "wiki"})
        (tmp_path / "wiki").mkdir()

        lint_mod.lint_all(MagicMock(), config)
        assert held == [True]
        assert not _wiki_lock_held()

    def test_run_full_build_holds_the_lock_while_generating(self, monkeypatch):
        from lilbee.wiki.generation import run_full_build

        held: list[bool] = []

        def fake_build_wiki(
            entities, provider, store, config, *, extract_concepts, on_progress, stats, cancel
        ):
            held.append(_wiki_lock_held())
            return []

        def fake_extractor(*args, **kwargs):
            extractor = MagicMock()
            extractor.extract.return_value = []
            return extractor

        services = MagicMock()
        services.store.get_sources.return_value = []
        monkeypatch.setattr("lilbee.wiki.generation.get_services", lambda: services)
        monkeypatch.setattr("lilbee.wiki.generation.build_wiki", fake_build_wiki)
        monkeypatch.setattr("lilbee.wiki.generation.update_wiki_index", lambda *a, **kw: None)
        monkeypatch.setattr("lilbee.wiki.generation.append_wiki_log", lambda *a, **kw: None)
        monkeypatch.setattr("lilbee.wiki.generation.get_entity_extractor", fake_extractor)

        run_full_build(cfg)
        assert held == [True]
        assert not _wiki_lock_held()

    def test_run_full_synthesize_holds_the_lock_while_generating(self, monkeypatch):
        from lilbee.wiki.generation import run_full_synthesize

        held: list[bool] = []

        def fake_generate(provider, store, clusterer, config, on_progress, stats, cancel):
            held.append(_wiki_lock_held())
            return []

        monkeypatch.setattr("lilbee.wiki.generation.get_services", MagicMock())
        monkeypatch.setattr("lilbee.wiki.generation.generate_synthesis_pages", fake_generate)

        run_full_synthesize(cfg)
        assert held == [True]
        assert not _wiki_lock_held()

    def test_prune_holds_the_lock_while_reconciling(self, monkeypatch, tmp_path):
        from lilbee.wiki.prune import prune_wiki

        held: list[bool] = []

        def fake_reconcile(store, wiki_root, config):
            held.append(_wiki_lock_held())
            return []

        monkeypatch.setattr("lilbee.wiki.prune._reconcile_orphan_rows", fake_reconcile)
        config = cfg.model_copy(update={"data_root": tmp_path, "wiki_dir": "wiki"})

        prune_wiki(MagicMock(), config)
        assert held == [True]
        assert not _wiki_lock_held()

    def test_accept_draft_holds_the_lock_while_publishing(self, monkeypatch, tmp_path):
        from lilbee.wiki.drafts import accept_draft

        wiki_root = tmp_path / "wiki"
        (wiki_root / "drafts").mkdir(parents=True)
        (wiki_root / "drafts" / "x.md").write_text("---\nsources: [a.txt]\n---\n\nbody\n")
        held: list[bool] = []

        def fake_index(content, wiki_source, store, config, chunks=None):
            held.append(_wiki_lock_held())
            return 1

        monkeypatch.setattr("lilbee.wiki.drafts.index_wiki_page", fake_index)

        accept_draft("x", wiki_root, MagicMock())
        assert held == [True]
        assert not _wiki_lock_held()

    def test_reject_draft_holds_the_lock_while_unlinking(self, monkeypatch, tmp_path):
        """Reject races accept over the same file: unlinking outside the lock can
        pull the draft out from under a publish that already landed the page."""
        from lilbee.wiki import drafts

        wiki_root = tmp_path / "wiki"
        (wiki_root / "drafts").mkdir(parents=True)
        (wiki_root / "drafts" / "x.md").write_text("---\nsources: [a.txt]\n---\n\nbody\n")
        held: list[bool] = []
        real_draft_path = drafts._draft_path

        def spy(root, slug):
            held.append(_wiki_lock_held())
            return real_draft_path(root, slug)

        monkeypatch.setattr("lilbee.wiki.drafts._draft_path", spy)

        drafts.reject_draft("x", wiki_root)
        assert held == [True]
        assert not _wiki_lock_held()
        assert not (wiki_root / "drafts" / "x.md").exists()

    def test_the_ingest_hook_takes_the_lock_in_its_worker_thread(self, monkeypatch):
        """The hook runs on the event loop, so it must acquire off it: a blocking
        acquire on the loop would stall every other request for a whole build."""
        import asyncio

        from lilbee.wiki.ingest import incremental_update

        entity = MagicMock()
        entity.chunk_refs = [MagicMock(source="a.txt")]
        extractor = MagicMock()
        extractor.extract.return_value = [entity]
        services = MagicMock()
        services.store.get_sources.return_value = []
        observed: list[tuple[bool, int]] = []

        def fake_build_wiki(entities, provider, store, config, *, extract_concepts, stats):
            observed.append((_wiki_lock_held(), threading.get_ident()))
            return []

        monkeypatch.setattr("lilbee.wiki.ingest.get_services", lambda: services)
        monkeypatch.setattr("lilbee.wiki.build_wiki", fake_build_wiki)
        monkeypatch.setattr("lilbee.wiki.update_wiki_index", lambda *a, **kw: None)
        monkeypatch.setattr("lilbee.wiki.append_wiki_log", lambda *a, **kw: None)
        monkeypatch.setattr(
            "lilbee.wiki.entity_extractor.get_entity_extractor", lambda *a, **kw: extractor
        )
        monkeypatch.setattr(cfg, "wiki", True)

        asyncio.run(incremental_update({"a.txt"}))
        assert len(observed) == 1
        locked, tid = observed[0]
        assert locked
        assert tid != threading.get_ident()
        assert not _wiki_lock_held()

    def test_the_ingest_cap_skip_logs_under_the_lock_off_the_loop(self, monkeypatch):
        """Above the cap the hook writes one log.md entry and stops; that append
        is a shared-file write like any other, and it must not block the loop."""
        import asyncio

        from lilbee.wiki.ingest import incremental_update

        entity = MagicMock()
        entity.chunk_refs = [MagicMock(source="a.txt")]
        extractor = MagicMock()
        extractor.extract.return_value = [entity, entity]
        services = MagicMock()
        services.store.get_sources.return_value = []
        observed: list[tuple[bool, int]] = []

        monkeypatch.setattr("lilbee.wiki.ingest.get_services", lambda: services)
        monkeypatch.setattr(
            "lilbee.wiki.append_wiki_log",
            lambda *a, **kw: observed.append((_wiki_lock_held(), threading.get_ident())),
        )
        monkeypatch.setattr(
            "lilbee.wiki.entity_extractor.get_entity_extractor", lambda *a, **kw: extractor
        )
        monkeypatch.setattr(cfg, "wiki", True)
        monkeypatch.setattr(cfg, "wiki_ingest_update_cap", 1)

        asyncio.run(incremental_update({"a.txt"}))
        assert len(observed) == 1
        locked, tid = observed[0]
        assert locked
        assert tid != threading.get_ident()
        assert not _wiki_lock_held()
