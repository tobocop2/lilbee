"""Tests for wiki index and log generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import write_wiki_page
from lilbee.config import cfg
from lilbee.wiki.index import (
    _parse_title,
    append_wiki_log,
    parse_source_count,
    update_wiki_index,
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.wiki = True
    cfg.wiki_dir = "wiki"
    cfg.chat_model = "test-model"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


_SUMMARY_PAGE = (
    "---\ntitle: My Document\nsources: [doc.md]\n---\n\n# My Document\n\nSome content.\n"
)

_CONCEPT_PAGE = (
    "---\n"
    "title: Type Safety\n"
    "sources: [a.md, b.md, c.md]\n"
    "---\n\n"
    "# Type Safety\n\nCross-source synthesis.\n"
)


class TestParseTitle:
    def test_from_frontmatter(self):
        assert _parse_title("---\ntitle: Hello World\n---\n# Other") == "Hello World"

    def test_from_heading(self):
        assert _parse_title("# First Heading\nSome text.") == "First Heading"

    def test_frontmatter_takes_precedence(self):
        assert _parse_title("---\ntitle: FM Title\n---\n# Heading") == "FM Title"

    def test_no_title(self):
        assert _parse_title("Just text, no heading.") == ""

    def test_empty_string(self):
        assert _parse_title("") == ""


class TestParseSourceCount:
    def test_single_source(self):
        assert parse_source_count("---\nsources: [doc.md]\n---\n") == 1

    def test_multiple_sources(self):
        assert parse_source_count("---\nsources: [a.md, b.md, c.md]\n---\n") == 3

    def test_no_sources_field(self):
        assert parse_source_count("---\ntitle: Hello\n---\n") == 0

    def test_empty_sources(self):
        assert parse_source_count("---\nsources: []\n---\n") == 0

    def test_no_frontmatter(self):
        assert parse_source_count("Just text, no frontmatter") == 0

    def test_string_sources_comma_separated(self):
        assert parse_source_count('---\nsources: "a.md, b.md"\n---\n') == 2


class TestUpdateWikiIndex:
    def test_empty_wiki(self, isolated_env: Path):
        path = update_wiki_index()
        assert path == isolated_env / "wiki" / "index.md"
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# Wiki Index")
        assert "- [" not in content

    def test_summary_pages_listed(self, isolated_env: Path):
        write_wiki_page(isolated_env, "summaries", "my-doc", _SUMMARY_PAGE)
        path = update_wiki_index()
        content = path.read_text(encoding="utf-8")
        assert "[My Document](summaries/my-doc.md)" in content
        assert "summary" in content
        assert "1 sources" in content

    def test_concept_pages_listed(self, isolated_env: Path):
        write_wiki_page(isolated_env, "concepts", "type-safety", _CONCEPT_PAGE)
        path = update_wiki_index()
        content = path.read_text(encoding="utf-8")
        assert "[Type Safety](concepts/type-safety.md)" in content
        assert "concept" in content
        assert "3 sources" in content

    def test_both_subdirs(self, isolated_env: Path):
        write_wiki_page(isolated_env, "summaries", "doc-a", _SUMMARY_PAGE)
        write_wiki_page(isolated_env, "concepts", "type-safety", _CONCEPT_PAGE)
        path = update_wiki_index()
        content = path.read_text(encoding="utf-8")
        assert "summaries/doc-a.md" in content
        assert "concepts/type-safety.md" in content

    def test_sorted_within_subdir(self, isolated_env: Path):
        write_wiki_page(isolated_env, "summaries", "z-doc", _SUMMARY_PAGE)
        write_wiki_page(isolated_env, "summaries", "a-doc", _SUMMARY_PAGE)
        path = update_wiki_index()
        content = path.read_text(encoding="utf-8")
        a_pos = content.index("a-doc")
        z_pos = content.index("z-doc")
        assert a_pos < z_pos

    def test_fallback_title_from_stem(self, isolated_env: Path):
        write_wiki_page(isolated_env, "summaries", "no-title", "Just text, no heading.")
        path = update_wiki_index()
        content = path.read_text(encoding="utf-8")
        assert "[No Title]" in content

    def test_creates_wiki_dir_if_missing(self, isolated_env: Path):
        wiki_dir = isolated_env / "wiki"
        assert not wiki_dir.exists()
        path = update_wiki_index()
        assert path.exists()

    def test_overwrites_existing_index(self, isolated_env: Path):
        write_wiki_page(isolated_env, "summaries", "doc-a", _SUMMARY_PAGE)
        update_wiki_index()
        write_wiki_page(isolated_env, "summaries", "doc-b", _SUMMARY_PAGE)
        path = update_wiki_index()
        content = path.read_text(encoding="utf-8")
        assert "doc-a" in content
        assert "doc-b" in content

    def test_accepts_explicit_config(self, isolated_env: Path):
        write_wiki_page(isolated_env, "summaries", "doc", _SUMMARY_PAGE)
        path = update_wiki_index(config=cfg)
        assert path.exists()


class TestAppendWikiLog:
    def test_creates_log_file(self, isolated_env: Path):
        path = append_wiki_log("generated", "summary page for doc.md")
        assert path == isolated_env / "wiki" / "log.md"
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# Wiki Log")
        assert "generated | summary page for doc.md" in content

    def test_appends_to_existing(self, isolated_env: Path):
        append_wiki_log("generated", "first event")
        path = append_wiki_log("pruned (archived)", "second event")
        content = path.read_text(encoding="utf-8")
        assert "first event" in content
        assert "second event" in content
        assert content.count("## [") == 2

    def test_timestamp_format(self, isolated_env: Path):
        path = append_wiki_log("test", "details")
        content = path.read_text(encoding="utf-8")
        # Should contain a date like [2026-04-04]
        import re

        assert re.search(r"## \[\d{4}-\d{2}-\d{2}\]", content)

    def test_preserves_header(self, isolated_env: Path):
        append_wiki_log("first", "a")
        append_wiki_log("second", "b")
        content = (isolated_env / "wiki" / "log.md").read_text(encoding="utf-8")
        assert content.count("# Wiki Log") == 1

    def test_creates_wiki_dir_if_missing(self, isolated_env: Path):
        wiki_dir = isolated_env / "wiki"
        assert not wiki_dir.exists()
        path = append_wiki_log("test", "details")
        assert path.exists()

    def test_accepts_explicit_config(self, isolated_env: Path):
        path = append_wiki_log("test", "details", config=cfg)
        assert path.exists()
