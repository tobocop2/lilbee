"""Tests for per-source batched wiki generation.

Covers the per-source batch path in ``lilbee.wiki.generation``:
section splitting, concept curation toggling, parse-failure and
slug-collision drafts, the incremental ``extract_concepts=False``
kwarg, and PENDING marker replacement on successful regen.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import SearchChunk
from lilbee.wiki.batch import match_label
from lilbee.wiki.entity_extractor import (
    ChunkRef,
    EntityKind,
    ExtractedEntity,
)
from lilbee.wiki.generation import _all_sources_in_scope, build_wiki
from lilbee.wiki.page import WIKI_DEFAULT_SEED
from lilbee.wiki.persistence import delete_pending_marker_if_present, divert_to_drafts
from lilbee.wiki.shared import (
    PENDING_MARKER_KEYWORD_COLLISION,
    PENDING_MARKER_KEYWORD_PARSE,
    WikiSubdir,
)
from lilbee.wiki.stats import BuildStats
from lilbee.wiki.synthesis import (
    _MAX_REUSE_CONCEPT_LABELS,
    _existing_concept_labels,
    _parse_declared_concepts,
    _prefix_heading,
    _split_batched_output,
    generate_source_batch,
)


def _chunk(source: str, idx: int, text: str) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text/plain",
        chunk_type="raw",
        page_start=0,
        page_end=0,
        line_start=0,
        line_end=0,
        chunk=text,
        chunk_index=idx,
        vector=[0.1] * cfg.embedding_dim,
    )


def _entity(slug: str, label: str, sources: list[str]) -> ExtractedEntity:
    return ExtractedEntity(
        slug=slug,
        kind=EntityKind.ENTITY,
        label=label,
        type_hint="PERSON",
        chunk_refs=tuple(ChunkRef(source=s, chunk_index=0) for s in sources),
    )


@pytest.fixture(autouse=True)
def isolated(wiki_isolated_env: Path):
    cfg.wiki = True
    cfg.wiki_batch_min_chunks = 1
    yield wiki_isolated_env


@pytest.fixture
def stub_embedder(monkeypatch):
    svc = MagicMock()
    svc.embedder.embed_batch.side_effect = lambda texts, **kw: [
        [0.1] * cfg.embedding_dim for _ in texts
    ]
    monkeypatch.setattr("lilbee.wiki.page.get_services", lambda: svc)
    monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
    return svc


def _mock_batch_provider(text: str, *, truncated: bool = False) -> MagicMock:
    from lilbee.providers.base import ChatResult, FinishReason

    provider = MagicMock()
    provider.chat.return_value = ChatResult(
        text=text,
        tool_calls=(),
        finish_reason=FinishReason.LENGTH if truncated else FinishReason.STOP,
    )
    provider.get_capabilities.return_value = []
    return provider


def _mock_store_for_source(source: str, chunks: list[SearchChunk]) -> MagicMock:
    store = MagicMock()
    store.get_chunks_by_source.side_effect = lambda name: chunks if name == source else []
    store.get_sources.return_value = [{"filename": source, "chunk_count": len(chunks)}]
    return store


_EXCERPT = "Henry Ford founded Ford Motor."


def _valid_citation_block(source: str = "s.txt") -> str:
    return (
        "\n\n---\n"
        "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
        f'[^src1]: {source}, excerpt: "{_EXCERPT}"\n'
    )


def _section(name: str, body: str | None = None) -> str:
    body = body or f"> {_EXCERPT}[^src1]\n"
    return f"## {name}\n\n{body}"


def _declare(*concepts: str) -> str:
    """Render the ``CONCEPTS:`` declaration line the batched prompt requires."""
    return "CONCEPTS: " + "; ".join(concepts) + "\n\n"


class TestSplitBatchedOutput:
    def test_matches_known_entities_case_insensitive(self):
        text = f"{_section('Henry Ford')}{_section('Ford Motor')}"
        parsed = _split_batched_output(text, {"Henry Ford", "Ford Motor"})
        assert set(parsed.keys()) == {"Henry Ford", "Ford Motor"}
        assert all(kind is EntityKind.ENTITY for kind, _ in parsed.values())

    def test_parse_fallback_to_h1(self):
        """A ``# Name`` section should still parse when the model used H1."""
        text = "# Henry Ford\n\n> fact body.[^src1]\n"
        parsed = _split_batched_output(text, {"Henry Ford"})
        assert "Henry Ford" in parsed

    def test_bold_line_does_not_split_a_section(self):
        """A mid-body ``**emphasis**`` line is body text, not a section boundary."""
        text = "## Henry Ford\n\n**Founder**\n\n> fact body.[^src1]\n"
        parsed = _split_batched_output(text, {"Henry Ford"})
        assert set(parsed) == {"Henry Ford"}
        assert "**Founder**" in parsed["Henry Ford"][1]

    def test_unmatched_entity_header_dropped(self):
        text = _section("Unknown Widget")
        parsed = _split_batched_output(text, {"Henry Ford"})
        assert parsed == {}

    def test_undeclared_concept_header_dropped(self):
        """A header matching no declared concept is noise, not a concept page."""
        text = _section("Key Takeaways")
        parsed = _split_batched_output(text, set(), expected_concept_labels={"Brake System"})
        assert parsed == {}

    def test_declared_concept_header_is_tagged_concept(self):
        text = _section("Brake System")
        parsed = _split_batched_output(text, set(), expected_concept_labels={"Brake System"})
        assert parsed["Brake System"][0] is EntityKind.CONCEPT

    def test_heading_is_rewritten_to_the_matched_label(self):
        """A header that matched only as a substring gets the full label as its H1."""
        text = "## Ford\n\n> Henry Ford founded Ford Motor.[^src1]\n"
        parsed = _split_batched_output(text, {"Henry Ford"})
        assert parsed["Henry Ford"][1].startswith("# Henry Ford\n\n")

    def test_empty_body_section_is_dropped(self):
        text = "## Henry Ford\n\n## Ford Motor\n\n> body.[^src1]\n"
        parsed = _split_batched_output(text, {"Henry Ford", "Ford Motor"})
        # Only Ford Motor had a body under its heading.
        assert "Ford Motor" in parsed
        assert "Henry Ford" not in parsed

    def test_a_declaration_rendered_as_a_heading_opens_no_section(self):
        """``## CONCEPTS: a; b`` is the declaration line, not a section header:
        its text substring-matches the labels it declares."""
        text = "## CONCEPTS: Assembly Line; Model T\n\nIntro prose about the plant.\n"
        parsed = _split_batched_output(
            text, set(), expected_concept_labels={"Assembly Line", "Model T"}
        )
        assert parsed == {}

    def test_a_real_section_after_the_declaration_still_parses(self):
        text = "## CONCEPTS: Assembly Line\n\nIntro prose.\n\n" + _section("Assembly Line")
        parsed = _split_batched_output(text, set(), expected_concept_labels={"Assembly Line"})
        assert set(parsed) == {"Assembly Line"}
        assert "Intro prose" not in parsed["Assembly Line"][1]

    def test_no_headers_returns_empty_dict(self):
        """A response that contains no H1/H2 headers recovers nothing."""
        text = "just a paragraph with no headings at all.\n\n> some body text\n"
        parsed = _split_batched_output(text, {"Henry Ford"})
        assert parsed == {}


class TestParseDeclaredConcepts:
    def test_reads_semicolon_separated_labels(self):
        text = _declare("Brake System", "Assembly Line") + _section("Brake System")
        assert _parse_declared_concepts(text) == {"Brake System", "Assembly Line"}

    def test_absent_declaration_curates_nothing(self):
        assert _parse_declared_concepts(_section("Brake System")) == set()

    def test_blank_entries_are_dropped(self):
        assert _parse_declared_concepts("CONCEPTS: Brakes; ;  \n") == {"Brakes"}

    @pytest.mark.parametrize(
        "line",
        [
            "**CONCEPTS:** Brake System; Assembly Line",
            "**CONCEPTS: Brake System; Assembly Line**",
            "`CONCEPTS: Brake System; Assembly Line`",
            "## CONCEPTS: Brake System; Assembly Line",
            "> CONCEPTS:  Brake System ; Assembly Line ",
            "concepts: Brake System; Assembly Line",
        ],
    )
    def test_markdown_decoration_around_the_declaration_is_tolerated(self, line: str):
        """Instruct models bold, quote and fence the label they are told to emit."""
        assert _parse_declared_concepts(f"{line}\n\n## Brake System\n\nbody\n") == {
            "Brake System",
            "Assembly Line",
        }


class TestMatchLabel:
    """Overlapping labels must bind the same way on every run."""

    def test_exact_match_wins_over_a_longer_overlap(self):
        matched = match_label("ford", {"Henry Ford", "Ford"}, EntityKind.ENTITY)
        assert matched == (EntityKind.ENTITY, "Ford")

    def test_longest_overlapping_label_wins(self):
        matched = match_label("henry ford jr.", {"Ford", "Henry Ford"}, EntityKind.ENTITY)
        assert matched == (EntityKind.ENTITY, "Henry Ford")

    def test_header_that_is_a_substring_of_a_label_still_matches(self):
        matched = match_label("ford", {"Henry Ford"}, EntityKind.ENTITY)
        assert matched == (EntityKind.ENTITY, "Henry Ford")

    def test_empty_label_never_matches(self):
        assert match_label("ford", {""}, EntityKind.ENTITY) is None

    def test_no_overlap_returns_none(self):
        assert match_label("chevrolet", {"Henry Ford"}, EntityKind.ENTITY) is None


class TestPrefixHeading:
    def test_builds_the_h1_from_the_label(self):
        out = _prefix_heading("Henry Ford", "body text")
        assert out.startswith("# Henry Ford\n\n")
        assert "body text" in out

    def test_structural_characters_are_stripped_from_the_heading(self):
        out = _prefix_heading("| | designer", "body")
        assert out.startswith("# designer\n\n")


class TestDeletePendingMarkerIfPresent:
    def test_returns_false_when_path_missing(self, tmp_path: Path):
        assert delete_pending_marker_if_present(tmp_path, "missing") is False

    def test_returns_false_when_file_is_not_pending_marker(self, tmp_path: Path):
        drafts = tmp_path / "drafts"
        drafts.mkdir()
        path = drafts / "foo.md"
        # No leading PENDING-PARSE / PENDING-COLLISION marker.
        path.write_text("just a plain draft body\n")
        assert delete_pending_marker_if_present(drafts, "foo") is False
        assert path.exists()

    def test_returns_false_on_read_oserror(self, tmp_path: Path, monkeypatch):
        drafts = tmp_path / "drafts"
        drafts.mkdir()
        path = drafts / "foo.md"
        path.write_text("ignored\n")

        def boom(self, *a, **kw):  # type: ignore[no-untyped-def]
            raise OSError("unreadable")

        monkeypatch.setattr(Path, "read_text", boom)
        assert delete_pending_marker_if_present(drafts, "foo") is False
        # File stays on disk since we couldn't even read it.
        assert path.exists()


class TestGenerateSourceBatchEdgeCases:
    """Guard branches in ``generate_source_batch``."""

    def test_empty_chunks_returns_empty_list(self, stub_embedder):
        provider = _mock_batch_provider("unused")
        pages = generate_source_batch(
            source="s.txt",
            entities=[],
            chunks=[],
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert pages == []
        provider.chat.assert_not_called()

    def test_provider_exception_marks_every_entity_pending(self, stub_embedder, caplog):
        chunks = [_chunk("s.txt", 0, "body")]
        provider = MagicMock()
        provider.chat.side_effect = RuntimeError("LLM down")
        provider.get_capabilities.return_value = []
        caplog.set_level("WARNING", logger="lilbee.wiki.synthesis")
        pages = generate_source_batch(
            source="s.txt",
            entities=[_entity("henry", "Henry", ["s.txt"])],
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert pages == []
        assert any("Batched LLM call failed" in r.message for r in caplog.records)
        marker = cfg.data_root / cfg.wiki_dir / WikiSubdir.DRAFTS / "henry.md"
        assert PENDING_MARKER_KEYWORD_PARSE in marker.read_text()

    def test_empty_response_marks_every_entity_pending(self, stub_embedder, caplog):
        chunks = [_chunk("s.txt", 0, "body")]
        # strip_reasoning + .strip() produces an empty string.
        provider = _mock_batch_provider("   \n  \n")
        caplog.set_level("WARNING", logger="lilbee.wiki.synthesis")
        pages = generate_source_batch(
            source="s.txt",
            entities=[_entity("henry", "Henry", ["s.txt"])],
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert pages == []
        assert any("empty response" in r.message for r in caplog.records)
        marker = cfg.data_root / cfg.wiki_dir / WikiSubdir.DRAFTS / "henry.md"
        assert PENDING_MARKER_KEYWORD_PARSE in marker.read_text()

    def test_truncated_citation_block_publishes_nothing(self, stub_embedder):
        """A response cut at max_tokens loses the shared citation block, so every
        section it carried becomes a PENDING marker rather than an uncited page."""
        chunks = [_chunk("s.txt", 0, _EXCERPT)]
        entities = [
            _entity("henry-ford", "Henry Ford", ["s.txt"]),
            _entity("ford-motor", "Ford Motor", ["s.txt"]),
        ]
        text = _section("Henry Ford") + _section("Ford Motor") + "\n\n---\n<!-- citations (auto-"
        provider = _mock_batch_provider(text, truncated=True)
        pages = generate_source_batch(
            source="s.txt",
            entities=entities,
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert pages == []
        wiki_root = cfg.data_root / cfg.wiki_dir
        for slug in ("henry-ford", "ford-motor"):
            marker = wiki_root / WikiSubdir.DRAFTS / f"{slug}.md"
            assert PENDING_MARKER_KEYWORD_PARSE in marker.read_text()
        assert not (wiki_root / WikiSubdir.ENTITIES).exists()

    def test_section_failing_the_citation_gate_leaves_a_pending_marker(self, stub_embedder):
        """A parsed section dropped by a downstream gate is not silently lost."""
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        # Section body cites nothing, so finalize_section drops it.
        text = _section("Henry Ford", "Henry Ford ran the company.\n")
        provider = _mock_batch_provider(text)
        pages = generate_source_batch(
            source="s.txt",
            entities=[_entity("henry-ford", "Henry Ford", ["s.txt"])],
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert pages == []
        marker = cfg.data_root / cfg.wiki_dir / WikiSubdir.DRAFTS / "henry-ford.md"
        assert PENDING_MARKER_KEYWORD_PARSE in marker.read_text()


class TestSectionFaithfulnessScope:
    """Which chunks a batched section is scored against."""

    @staticmethod
    def _record_scored_chunks(monkeypatch) -> list[list[int]]:
        """Capture the chunk indices handed to each faithfulness check."""
        scored: list[list[int]] = []

        def _record(
            chunks: list[SearchChunk], body: str, label: str, config: object = None
        ) -> float:
            scored.append([c.chunk_index for c in chunks])
            return 1.0

        monkeypatch.setattr("lilbee.wiki.batch.check_faithfulness", _record)
        return scored

    def test_entity_section_scores_against_its_own_chunk_refs(self, stub_embedder, monkeypatch):
        """A niche section is compared to its entity's chunks, not the source centroid."""
        chunks = [_chunk("s.txt", i, f"{_EXCERPT} passage {i}") for i in range(3)]
        entity = ExtractedEntity(
            slug="henry-ford",
            kind=EntityKind.ENTITY,
            label="Henry Ford",
            type_hint="PERSON",
            chunk_refs=(ChunkRef(source="s.txt", chunk_index=1),),
        )
        scored = self._record_scored_chunks(monkeypatch)
        provider = _mock_batch_provider(_section("Henry Ford") + _valid_citation_block())

        generate_source_batch(
            source="s.txt",
            entities=[entity],
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert scored == [[1]]

    def test_concept_section_scores_against_the_whole_source(self, stub_embedder, monkeypatch):
        """Concepts have no extraction refs, so they keep the full chunk pool."""
        chunks = [_chunk("s.txt", i, f"{_EXCERPT} passage {i}") for i in range(3)]
        scored = self._record_scored_chunks(monkeypatch)
        text = _declare("Assembly Line") + _section("Assembly Line") + _valid_citation_block()
        provider = _mock_batch_provider(text)

        generate_source_batch(
            source="s.txt",
            entities=[],
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=True,
            written_concept_slugs={},
        )
        assert scored == [[0, 1, 2]]


class TestFinalizeSectionGuards:
    """``finalize_section`` safety rails that run before any write."""

    def test_empty_header_label_produces_empty_slug_and_skips(self, stub_embedder, caplog):
        """A header that slugifies to empty (all-punctuation) is dropped."""
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        # ``---`` splits into a section whose header is ``---``: after
        # make_slug, this yields the empty string and the section is
        # dropped with an INFO log. The downstream faithfulness /
        # citation checks never run.
        text = (
            _declare("---")
            + "## ---\n\n> Henry Ford founded Ford Motor. [^src1]\n"
            + _valid_citation_block()
        )
        provider = _mock_batch_provider(text)
        caplog.set_level("INFO", logger="lilbee.wiki.batch")
        # Concept-curation mode so the unmatched header is tagged as a
        # concept and reaches _finalize_section (entities-only mode
        # would just drop it in _split_batched_output).
        pages = generate_source_batch(
            source="s.txt",
            entities=[],
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=True,
            written_concept_slugs={},
        )
        assert pages == []
        assert any("Empty slug" in r.message for r in caplog.records)

    def test_below_threshold_routes_to_drafts(self, stub_embedder, monkeypatch, caplog):
        """Score below the faithfulness threshold → draft subdir + info log."""
        # Force the faithfulness score below the threshold by making
        # the body vector orthogonal to the chunk vectors.
        svc = MagicMock()
        svc.embedder.embed_batch.return_value = [[0.0] * cfg.embedding_dim]
        # First entry to zero-vec body, then flip one element on
        # subsequent calls (index step), so the chunk vectors used
        # earlier still differ.
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        text = (
            _section("Henry Ford", "> Henry Ford founded Ford Motor. [^src1]\n")
            + _valid_citation_block()
        )
        provider = _mock_batch_provider(text)
        caplog.set_level("INFO", logger="lilbee.wiki.batch")
        pages = generate_source_batch(
            source="s.txt",
            entities=[_entity("henry-ford", "Henry Ford", ["s.txt"])],
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        # The page landed under drafts, not entities.
        assert len(pages) == 1
        assert WikiSubdir.DRAFTS in pages[0].parts
        assert any("sending to drafts" in r.message for r in caplog.records)


class TestBatchSectionCitationRendering:
    """A section publishes only the footnotes that verified."""

    def test_a_dropped_citation_leaves_no_marker_in_the_body(self, stub_embedder):
        text = _section(
            "Henry Ford", f"> {_EXCERPT}[^src1]\n\n> Ford built a monorail.[^src2]\n"
        ) + (
            "\n\n---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            f'[^src1]: s.txt, excerpt: "{_EXCERPT}"\n'
            '[^src2]: s.txt, excerpt: "Ford built a monorail."\n'
        )
        pages = generate_source_batch(
            source="s.txt",
            entities=[_entity("henry-ford", "Henry Ford", ["s.txt"])],
            chunks=[_chunk("s.txt", 0, _EXCERPT)],
            provider=_mock_batch_provider(text),
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        body = pages[0].read_text()
        assert "[^src1]" in body
        assert "[^src2]" not in body


class TestBuildStatsForABatch:
    """A batch run reports what its gates did, so a regression is measurable."""

    def test_published_sections_count_with_their_verified_citations(self, stub_embedder):
        chunks = [_chunk("s.txt", 0, _EXCERPT)]
        entities = [
            _entity("henry-ford", "Henry Ford", ["s.txt"]),
            _entity("ford-motor", "Ford Motor", ["s.txt"]),
        ]
        text = (
            _section("Henry Ford", f"> {_EXCERPT}[^src1]\n")
            + _section("Ford Motor", f"> {_EXCERPT}[^src1]\n")
            + _valid_citation_block()
        )
        stats = BuildStats()
        pages = generate_source_batch(
            source="s.txt",
            entities=entities,
            chunks=chunks,
            provider=_mock_batch_provider(text),
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
            stats=stats,
        )
        assert len(pages) == 2
        assert stats.pages_generated == 2
        assert stats.pages_published == 2
        assert stats.pages_drafted == 0
        assert stats.pending_markers == 0
        assert stats.publish_rate == 1.0
        assert stats.verified_by_page == {
            f"{cfg.wiki_dir}/{WikiSubdir.ENTITIES}/henry-ford.md": 1,
            f"{cfg.wiki_dir}/{WikiSubdir.ENTITIES}/ford-motor.md": 1,
        }
        assert stats.citation_verify_rate == 1.0

    def test_each_section_claims_only_the_citations_it_references(self, stub_embedder):
        """The response's trailing citation block lands in the last section's body.
        Its definitions must not make that page claim every citation in the run."""
        second_excerpt = "The Model T sold widely."
        chunks = [_chunk("s.txt", 0, _EXCERPT), _chunk("s.txt", 1, second_excerpt)]
        text = (
            _section("Henry Ford", f"> {_EXCERPT}[^src1]\n")
            + _section("Model T", f"> {second_excerpt}[^src2]\n")
            + "\n\n---\n"
            "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
            f'[^src1]: s.txt, excerpt: "{_EXCERPT}"\n'
            f'[^src2]: s.txt, excerpt: "{second_excerpt}"\n'
        )
        stats = BuildStats()
        pages = generate_source_batch(
            source="s.txt",
            entities=[
                _entity("henry-ford", "Henry Ford", ["s.txt"]),
                _entity("model-t", "Model T", ["s.txt"]),
            ],
            chunks=chunks,
            provider=_mock_batch_provider(text),
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
            stats=stats,
        )
        last_page = next(page for page in pages if page.stem == "model-t")
        rendered = last_page.read_text()
        assert "[^src2]:" in rendered
        assert "[^src1]:" not in rendered
        assert stats.verified_by_page == {
            f"{cfg.wiki_dir}/{WikiSubdir.ENTITIES}/henry-ford.md": 1,
            f"{cfg.wiki_dir}/{WikiSubdir.ENTITIES}/model-t.md": 1,
        }

    def test_a_section_below_the_faithfulness_threshold_counts_as_drafted(
        self, stub_embedder, monkeypatch
    ):
        svc = MagicMock()
        svc.embedder.embed_batch.return_value = [[0.0] * cfg.embedding_dim]
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
        text = _section("Henry Ford", f"> {_EXCERPT}[^src1]\n") + _valid_citation_block()
        stats = BuildStats()
        generate_source_batch(
            source="s.txt",
            entities=[_entity("henry-ford", "Henry Ford", ["s.txt"])],
            chunks=[_chunk("s.txt", 0, _EXCERPT)],
            provider=_mock_batch_provider(text),
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
            stats=stats,
        )
        assert stats.pages_generated == 1
        assert stats.pages_published == 0
        assert stats.pages_drafted == 1
        assert stats.publish_rate == 0.0
        assert stats.verified_by_page == {}

    def test_an_unverifiable_excerpt_counts_as_a_dropped_citation_and_a_marker(self, stub_embedder):
        """The citation gate rejecting a footnote is visible in the run's numbers."""
        text = _section("Henry Ford", f"> {_EXCERPT}[^src1]\n") + _valid_citation_block()
        stats = BuildStats()
        pages = generate_source_batch(
            source="s.txt",
            entities=[_entity("henry-ford", "Henry Ford", ["s.txt"])],
            chunks=[_chunk("s.txt", 0, "Nothing in this chunk matches the excerpt.")],
            provider=_mock_batch_provider(text),
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
            stats=stats,
        )
        assert pages == []
        assert stats.pages_generated == 0
        assert stats.citations_rendered == 0
        assert stats.citations_dropped_unverified == 1
        assert stats.citation_verify_rate == 0.0
        assert stats.pending_markers == 1

    def test_a_section_lost_to_a_slug_collision_counts_no_citations(self, stub_embedder):
        """The losing section becomes a collision marker, not a page, so its
        citations were never rendered."""
        text = (
            _declare("Brake System")
            + _section("Brake System", f"> {_EXCERPT}[^src1]\n")
            + _valid_citation_block("second.txt")
        )
        stats = BuildStats()
        pages = generate_source_batch(
            source="second.txt",
            entities=[],
            chunks=[_chunk("second.txt", 0, _EXCERPT)],
            provider=_mock_batch_provider(text),
            store=MagicMock(),
            config=cfg,
            extract_concepts=True,
            written_concept_slugs={"brake-system": "first.txt"},
            stats=stats,
        )
        assert pages == []
        assert stats.pending_markers == 1
        assert stats.citations_rendered == 0
        assert stats.citations_dropped_unverified == 0

    def test_a_marker_withheld_by_a_draft_under_review_is_not_counted(self, stub_embedder):
        """The draft holding review content is kept, so no marker landed to count."""
        drafts_dir = cfg.data_root / cfg.wiki_dir / WikiSubdir.DRAFTS
        review_draft = divert_to_drafts(
            "# Henry Ford\n\nReviewed body.\n",
            drafts_dir,
            "henry-ford",
            0.5,
            "diff",
            WikiSubdir.ENTITIES,
            ["s.txt"],
        )
        stats = BuildStats()
        pages = generate_source_batch(
            source="s.txt",
            entities=[_entity("henry-ford", "Henry Ford", ["s.txt"])],
            chunks=[_chunk("s.txt", 0, _EXCERPT)],
            provider=_mock_batch_provider("A response with no section headers at all.\n"),
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
            stats=stats,
        )
        assert pages == []
        assert stats.pending_markers == 0
        assert "Reviewed body." in review_draft.read_text()

    def test_a_failed_llm_call_counts_one_marker_per_expected_entity(self, stub_embedder):
        provider = MagicMock()
        provider.chat.side_effect = RuntimeError("LLM down")
        provider.get_capabilities.return_value = []
        stats = BuildStats()
        generate_source_batch(
            source="s.txt",
            entities=[
                _entity("henry-ford", "Henry Ford", ["s.txt"]),
                _entity("ford-motor", "Ford Motor", ["s.txt"]),
            ],
            chunks=[_chunk("s.txt", 0, "body")],
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
            stats=stats,
        )
        assert stats.pending_markers == 2
        assert stats.pages_generated == 0

    def test_a_concept_slug_collision_counts_a_marker_not_a_published_page(self, stub_embedder):
        def _batch_text(source: str) -> str:
            return (
                _declare("Brake System") + "## Brake System\n\n> Brake system details. [^src1]\n"
                "\n\n---\n"
                "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
                f'[^src1]: {source}, excerpt: "Brake system details."\n'
            )

        written: dict[str, str] = {}
        stats = BuildStats()
        returned: list[list[Path]] = []
        for source in ("s1.txt", "s2.txt"):
            returned.append(
                generate_source_batch(
                    source=source,
                    entities=[],
                    chunks=[_chunk(source, 0, "Brake system details.")],
                    provider=_mock_batch_provider(_batch_text(source)),
                    store=MagicMock(),
                    config=cfg,
                    extract_concepts=True,
                    written_concept_slugs=written,
                    stats=stats,
                )
            )
        assert stats.pages_published == 1
        assert stats.pending_markers == 1
        assert stats.pages_generated == 1
        # The collision marker is a side channel, not a page: counting it would put
        # the reported page count above pages_generated.
        assert [len(pages) for pages in returned] == [1, 0]

    def test_a_batch_without_a_collector_still_generates(self, stub_embedder):
        """The stats argument is optional: callers that do not measure still build."""
        text = _section("Henry Ford", f"> {_EXCERPT}[^src1]\n") + _valid_citation_block()
        pages = generate_source_batch(
            source="s.txt",
            entities=[_entity("henry-ford", "Henry Ford", ["s.txt"])],
            chunks=[_chunk("s.txt", 0, _EXCERPT)],
            provider=_mock_batch_provider(text),
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert len(pages) == 1


class TestAllSourcesInScope:
    def test_extract_concepts_false_returns_grouped_sources_only(self):
        store = MagicMock()
        grouped = {"a.md": []}
        result = _all_sources_in_scope(
            grouped=grouped,
            store=store,
            config=cfg,
            extract_concepts=False,
        )
        assert result == {"a.md"}
        # get_sources must not be called when concepts are disabled.
        store.get_sources.assert_not_called()

    def test_get_sources_exception_falls_back_to_grouped(self, caplog):
        store = MagicMock()
        store.get_sources.side_effect = RuntimeError("backend down")
        grouped = {"a.md": []}
        caplog.set_level("WARNING", logger="lilbee.wiki.generation")
        result = _all_sources_in_scope(
            grouped=grouped,
            store=store,
            config=cfg,
            extract_concepts=True,
        )
        assert result == {"a.md"}
        assert any("get_sources failed" in r.message for r in caplog.records)

    def test_skips_empty_filenames_and_non_dict_records(self):
        store = MagicMock()
        store.get_sources.return_value = [
            {"filename": "", "chunk_count": 10},
            "not a dict",
            {"filename": "b.md", "chunk_count": 5},
        ]
        cfg.wiki_batch_min_chunks = 1
        result = _all_sources_in_scope(
            grouped={},
            store=store,
            config=cfg,
            extract_concepts=True,
        )
        assert result == {"b.md"}

    def test_skips_sources_already_grouped(self):
        store = MagicMock()
        store.get_sources.return_value = [
            {"filename": "a.md", "chunk_count": 100},
        ]
        cfg.wiki_batch_min_chunks = 1
        result = _all_sources_in_scope(
            grouped={"a.md": []},
            store=store,
            config=cfg,
            extract_concepts=True,
        )
        # Still a single-element set: dedup kept the grouped entry.
        assert result == {"a.md"}

    def test_skips_sources_below_min_chunks(self):
        store = MagicMock()
        store.get_sources.return_value = [
            {"filename": "big.md", "chunk_count": 10},
            {"filename": "small.md", "chunk_count": 1},
        ]
        cfg.wiki_batch_min_chunks = 5
        result = _all_sources_in_scope(
            grouped={},
            store=store,
            config=cfg,
            extract_concepts=True,
        )
        assert result == {"big.md"}


class TestBuildWikiSkipLogging:
    def test_build_wiki_logs_skipped_source_when_below_floor(self, stub_embedder, caplog):
        """A source with no entities AND below min_chunks logs a skip."""
        cfg.wiki_batch_min_chunks = 5
        store = MagicMock()
        # Source makes it into the union via _all_sources_in_scope
        # (chunk_count >= min in that path) but store.get_chunks_by_source
        # returns fewer chunks, so source_extract ends up False again.
        store.get_sources.return_value = [
            {"filename": "s.txt", "chunk_count": 5},
        ]
        store.get_chunks_by_source.return_value = [_chunk("s.txt", 0, "x")]
        provider = _mock_batch_provider("unused")
        caplog.set_level("INFO", logger="lilbee.wiki.generation")
        pages = build_wiki([], provider, store, cfg, extract_concepts=True)
        assert pages == []
        assert any("Skipping source" in r.message for r in caplog.records)
        provider.chat.assert_not_called()


class TestBatchGeneration:
    def test_batch_generation_single_source_multi_entity(self, stub_embedder):
        """One LLM call → N entity pages under wiki/entities/."""
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        entities = [
            _entity("henry-ford", "Henry Ford", ["s.txt"]),
            _entity("ford-motor", "Ford Motor", ["s.txt"]),
        ]
        text = (
            _section("Henry Ford", "> Henry Ford founded Ford Motor. [^src1]\n")
            + _section(
                "Ford Motor",
                "> Henry Ford founded Ford Motor. [^src1]\n",
            )
            + _valid_citation_block()
        )
        provider = _mock_batch_provider(text)
        pages = generate_source_batch(
            source="s.txt",
            entities=entities,
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert len(pages) == 2
        assert provider.chat.call_count == 1
        assert all(WikiSubdir.ENTITIES in str(p) for p in pages)

    def test_batch_generation_llm_curates_concepts(self, stub_embedder):
        """extract_concepts=True asks for concepts and for the declaration line."""
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        entities = [_entity("henry-ford", "Henry Ford", ["s.txt"])]
        text = (
            _declare("Assembly Line")
            + _section("Henry Ford", "> Henry Ford founded Ford Motor. [^src1]\n")
            + _section("Assembly Line", "> Assembly Line innovation.[^src1]\n")
            + _valid_citation_block()
        )
        provider = _mock_batch_provider(text)
        pages = generate_source_batch(
            source="s.txt",
            entities=entities,
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=True,
            written_concept_slugs={},
        )
        prompt_arg = provider.chat.call_args[0][0][0]["content"]
        assert "identify 3-5 CONCEPTS" in prompt_arg
        assert "CONCEPTS: first; second; third" in prompt_arg
        assert {p.parent.name for p in pages} == {WikiSubdir.ENTITIES, WikiSubdir.CONCEPTS}

    def test_existing_concept_names_are_offered_for_reuse(self, stub_embedder):
        """Published concept slugs ride the prompt so rebuilds don't rename them."""
        concepts_dir = cfg.data_root / cfg.wiki_dir / WikiSubdir.CONCEPTS
        concepts_dir.mkdir(parents=True, exist_ok=True)
        (concepts_dir / "assembly-line.md").write_text("---\n---\n\n# Assembly Line\n")
        provider = _mock_batch_provider(_declare("Assembly Line") + _section("Assembly Line"))
        generate_source_batch(
            source="s.txt",
            entities=[],
            chunks=[_chunk("s.txt", 0, _EXCERPT)],
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=True,
            written_concept_slugs={},
        )
        prompt_arg = provider.chat.call_args[0][0][0]["content"]
        assert "Reuse these existing concept names verbatim" in prompt_arg
        assert "assembly line" in prompt_arg

    def test_the_reuse_list_is_capped(self, tmp_path: Path):
        """Uncapped, the list grows with the wiki until it crowds chunks out of the
        prompt: the budget is charged for it, so it cannot grow without bound."""
        concepts_dir = tmp_path / WikiSubdir.CONCEPTS
        concepts_dir.mkdir(parents=True)
        for index in range(_MAX_REUSE_CONCEPT_LABELS + 10):
            (concepts_dir / f"concept-{index:03d}.md").write_text("---\n---\n")
        labels = _existing_concept_labels(tmp_path)
        assert len(labels) == _MAX_REUSE_CONCEPT_LABELS
        assert labels == sorted(labels)

    def test_the_rendered_concept_instruction_is_charged_against_the_chunk_budget(
        self, stub_embedder
    ):
        """The instruction and its reuse list are per-call substitutions the raw
        template does not carry, so an unbudgeted one would overflow num_ctx."""
        cfg.num_ctx = 4096
        cfg.wiki_summary_max_tokens = 512
        concepts_dir = cfg.data_root / cfg.wiki_dir / WikiSubdir.CONCEPTS
        concepts_dir.mkdir(parents=True, exist_ok=True)
        for index in range(_MAX_REUSE_CONCEPT_LABELS):
            (concepts_dir / f"a-long-established-concept-name-{index:03d}.md").write_text("x")
        chunks = [_chunk("s.txt", i, "x" * 400) for i in range(40)]
        provider = _mock_batch_provider(_section("Henry Ford"))
        generate_source_batch(
            source="s.txt",
            entities=[_entity("henry-ford", "Henry Ford", ["s.txt"])],
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=True,
            written_concept_slugs={},
        )
        prompt_arg = provider.chat.call_args[0][0][0]["content"]
        assert len(prompt_arg) // 4 + cfg.wiki_summary_max_tokens <= cfg.num_ctx

    def test_generation_is_seeded_so_rebuilds_converge(self, stub_embedder):
        """No user seed means the fixed wiki seed, not sampler luck."""
        provider = _mock_batch_provider(_section("Henry Ford"))
        generate_source_batch(
            source="s.txt",
            entities=[_entity("henry-ford", "Henry Ford", ["s.txt"])],
            chunks=[_chunk("s.txt", 0, _EXCERPT)],
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert provider.chat.call_args.kwargs["options"]["seed"] == WIKI_DEFAULT_SEED

    def test_batch_generation_parse_fallback_to_h1(self, stub_embedder, tmp_path: Path):
        """H1 sections still parse and write pages."""
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        entities = [_entity("henry-ford", "Henry Ford", ["s.txt"])]
        text = (
            "# Henry Ford\n\n> Henry Ford founded Ford Motor. [^src1]\n" + _valid_citation_block()
        )
        provider = _mock_batch_provider(text)
        pages = generate_source_batch(
            source="s.txt",
            entities=entities,
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert len(pages) == 1

    def test_batch_generation_parse_failure_writes_pending_marker(
        self, stub_embedder, tmp_path: Path
    ):
        """Labels not recovered from the response → drafts/<slug>.md marker."""
        chunks = [_chunk("s.txt", 0, "body")]
        entities = [
            _entity("henry-ford", "Henry Ford", ["s.txt"]),
            _entity("ford-motor", "Ford Motor", ["s.txt"]),
        ]
        # Only one section is present; Ford Motor fails to parse.
        text = _section("Henry Ford", "> body.[^src1]\n") + _valid_citation_block()
        provider = _mock_batch_provider(text)
        generate_source_batch(
            source="s.txt",
            entities=entities,
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        marker = cfg.data_root / cfg.wiki_dir / WikiSubdir.DRAFTS / "ford-motor.md"
        assert marker.exists()
        body = marker.read_text()
        assert PENDING_MARKER_KEYWORD_PARSE in body
        assert "ford motor" in body.lower()

    def test_batch_generation_slug_collision_writes_collision_marker(self, stub_embedder):
        """Two sources proposing the same concept slug → collision marker."""
        chunks1 = [_chunk("s1.txt", 0, "Brake system details.")]
        chunks2 = [_chunk("s2.txt", 0, "Brake system details.")]

        def _batch_text(source: str) -> str:
            return (
                _declare("Brake System") + "## Brake System\n\n> Brake system details. [^src1]\n"
                "\n\n---\n"
                "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
                f'[^src1]: {source}, excerpt: "Brake system details."\n'
            )

        written: dict[str, str] = {}
        provider1 = _mock_batch_provider(_batch_text("s1.txt"))
        generate_source_batch(
            source="s1.txt",
            entities=[],
            chunks=chunks1,
            provider=provider1,
            store=MagicMock(),
            config=cfg,
            extract_concepts=True,
            written_concept_slugs=written,
        )
        provider2 = _mock_batch_provider(_batch_text("s2.txt"))
        generate_source_batch(
            source="s2.txt",
            entities=[],
            chunks=chunks2,
            provider=provider2,
            store=MagicMock(),
            config=cfg,
            extract_concepts=True,
            written_concept_slugs=written,
        )
        drafts_dir = cfg.data_root / cfg.wiki_dir / WikiSubdir.DRAFTS
        collision_files = list(drafts_dir.glob("brake-system-collision-*.md"))
        assert len(collision_files) == 1
        assert PENDING_MARKER_KEYWORD_COLLISION in collision_files[0].read_text()

    def test_below_threshold_concept_collision_still_diverts(self, stub_embedder, monkeypatch):
        """Two sources proposing the same concept slug that BOTH score below the
        faithfulness threshold must still produce a collision marker, not silently
        overwrite each other at drafts/<slug>.md (bb-ziks.68)."""
        svc = MagicMock()
        svc.embedder.embed_batch.return_value = [[0.0] * cfg.embedding_dim]
        monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)

        def _batch_text(source: str) -> str:
            return (
                _declare("Brake System") + "## Brake System\n\n> Brake system details. [^src1]\n"
                "\n\n---\n"
                "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
                f'[^src1]: {source}, excerpt: "Brake system details."\n'
            )

        written: dict[str, str] = {}
        for src in ("s1.txt", "s2.txt"):
            generate_source_batch(
                source=src,
                entities=[],
                chunks=[_chunk(src, 0, "Brake system details.")],
                provider=_mock_batch_provider(_batch_text(src)),
                store=MagicMock(),
                config=cfg,
                extract_concepts=True,
                written_concept_slugs=written,
            )
        drafts_dir = cfg.data_root / cfg.wiki_dir / WikiSubdir.DRAFTS
        collision_files = list(drafts_dir.glob("brake-system-collision-*.md"))
        assert len(collision_files) == 1
        assert PENDING_MARKER_KEYWORD_COLLISION in collision_files[0].read_text()

    def test_batch_generation_skips_sources_below_min_chunks(self, stub_embedder):
        """Source with <min_chunks AND no entities → no call at all."""
        cfg.wiki_batch_min_chunks = 3
        store = MagicMock()
        store.get_sources.return_value = [
            {"filename": "s.txt", "chunk_count": 1},
        ]
        store.get_chunks_by_source.return_value = [_chunk("s.txt", 0, "x")]
        provider = _mock_batch_provider("unused")
        with patch("lilbee.wiki.generation.generate_source_batch") as batch:
            build_wiki([], provider, store, cfg)
        batch.assert_not_called()

    def test_build_wiki_extract_concepts_false_omits_concepts(self, stub_embedder):
        """Incremental path: extract_concepts=False drops the concept
        instruction from the prompt, so no concept sections are written.
        """
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        store = MagicMock()
        store.get_sources.return_value = [{"filename": "s.txt", "chunk_count": len(chunks)}]
        store.get_chunks_by_source.return_value = chunks
        entities = [_entity("henry-ford", "Henry Ford", ["s.txt"])]
        text = (
            _section("Henry Ford", "> Henry Ford founded Ford Motor. [^src1]\n")
            + _valid_citation_block()
        )
        provider = _mock_batch_provider(text)
        build_wiki(entities, provider, store, cfg, extract_concepts=False)
        prompt_arg = provider.chat.call_args[0][0][0]["content"]
        assert "identify 3-5 CONCEPTS" not in prompt_arg

    def test_pending_marker_is_replaced_on_successful_regen(self, stub_embedder):
        """Rerun after a PENDING-PARSE marker and success → marker deleted."""
        drafts_dir = cfg.data_root / cfg.wiki_dir / WikiSubdir.DRAFTS
        drafts_dir.mkdir(parents=True, exist_ok=True)
        # Simulate the previous failed build's marker.
        marker = drafts_dir / "henry-ford.md"
        marker.write_text(
            f"<!-- {PENDING_MARKER_KEYWORD_PARSE} for source s.txt, "
            "entity/concept Henry Ford - retry -->\n"
        )
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        entities = [_entity("henry-ford", "Henry Ford", ["s.txt"])]
        text = (
            _section("Henry Ford", "> Henry Ford founded Ford Motor. [^src1]\n")
            + _valid_citation_block()
        )
        provider = _mock_batch_provider(text)
        generate_source_batch(
            source="s.txt",
            entities=entities,
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        assert not marker.exists()
