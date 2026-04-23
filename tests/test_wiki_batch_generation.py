"""Tests for Phase D per-source batched wiki generation.

Covers the new per-source batch path in ``lilbee.wiki.gen``:
section splitting, concept curation toggling, parse-failure and
slug-collision drafts, the incremental ``extract_concepts=False``
kwarg, and PENDING marker replacement on successful regen.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.config import cfg
from lilbee.store import SearchChunk
from lilbee.wiki.entity_extractor import (
    ChunkRef,
    EntityKind,
    ExtractedEntity,
)
from lilbee.wiki.gen import (
    _generate_source_batch,
    _split_batched_output,
    build_wiki,
)
from lilbee.wiki.shared import DRAFTS_SUBDIR, ENTITIES_SUBDIR


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
    monkeypatch.setattr("lilbee.wiki.gen.get_services", lambda: svc)
    return svc


def _mock_batch_provider(text: str) -> MagicMock:
    provider = MagicMock()
    provider.chat.return_value = text
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

    def test_bold_line_header(self):
        text = "**Henry Ford**\n\n> fact body.[^src1]\n"
        parsed = _split_batched_output(text, {"Henry Ford"})
        assert "Henry Ford" in parsed

    def test_unmatched_entity_header_dropped(self):
        text = _section("Unknown Widget")
        parsed = _split_batched_output(text, {"Henry Ford"})
        assert parsed == {}

    def test_concept_tagged_when_concepts_expected(self):
        text = _section("Brake System")
        parsed = _split_batched_output(text, set(), expected_concept_labels=set())
        assert "Brake System" in parsed
        assert parsed["Brake System"][0] is EntityKind.CONCEPT

    def test_empty_body_section_is_dropped(self):
        text = "## Henry Ford\n\n## Ford Motor\n\n> body.[^src1]\n"
        parsed = _split_batched_output(text, {"Henry Ford", "Ford Motor"})
        # Only Ford Motor had a body under its heading.
        assert "Ford Motor" in parsed
        assert "Henry Ford" not in parsed


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
        pages = _generate_source_batch(
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
        assert all(ENTITIES_SUBDIR in str(p) for p in pages)

    def test_batch_generation_llm_curates_concepts(self, stub_embedder):
        """extract_concepts=True includes the concept-curation paragraph."""
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        entities = [_entity("henry-ford", "Henry Ford", ["s.txt"])]
        text = (
            _section("Henry Ford", "> Henry Ford founded Ford Motor. [^src1]\n")
            + _section("Assembly Line", "> Assembly Line innovation.[^src1]\n")
            + _valid_citation_block()
        )
        provider = _mock_batch_provider(text)
        _generate_source_batch(
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

    def test_batch_generation_parse_fallback_to_h1(self, stub_embedder, tmp_path: Path):
        """H1 sections still parse and write pages."""
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        entities = [_entity("henry-ford", "Henry Ford", ["s.txt"])]
        text = (
            "# Henry Ford\n\n> Henry Ford founded Ford Motor. [^src1]\n" + _valid_citation_block()
        )
        provider = _mock_batch_provider(text)
        pages = _generate_source_batch(
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
        _generate_source_batch(
            source="s.txt",
            entities=entities,
            chunks=chunks,
            provider=provider,
            store=MagicMock(),
            config=cfg,
            extract_concepts=False,
            written_concept_slugs={},
        )
        marker = cfg.data_root / cfg.wiki_dir / DRAFTS_SUBDIR / "ford-motor.md"
        assert marker.exists()
        body = marker.read_text()
        assert "PENDING: batch parse failed" in body
        assert "ford motor" in body.lower()

    def test_batch_generation_slug_collision_writes_collision_marker(self, stub_embedder):
        """Two sources proposing the same concept slug → collision marker."""
        chunks1 = [_chunk("s1.txt", 0, "Brake system details.")]
        chunks2 = [_chunk("s2.txt", 0, "Brake system details.")]

        def _batch_text(source: str) -> str:
            return (
                f"## Brake System\n\n> Brake system details. [^src1]\n"
                "\n\n---\n"
                "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
                f'[^src1]: {source}, excerpt: "Brake system details."\n'
            )

        written: dict[str, str] = {}
        provider1 = _mock_batch_provider(_batch_text("s1.txt"))
        _generate_source_batch(
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
        _generate_source_batch(
            source="s2.txt",
            entities=[],
            chunks=chunks2,
            provider=provider2,
            store=MagicMock(),
            config=cfg,
            extract_concepts=True,
            written_concept_slugs=written,
        )
        drafts_dir = cfg.data_root / cfg.wiki_dir / DRAFTS_SUBDIR
        collision_files = list(drafts_dir.glob("brake-system-collision-*.md"))
        assert len(collision_files) == 1
        assert "PENDING: concept slug collision" in collision_files[0].read_text()

    def test_batch_generation_skips_sources_below_min_chunks(self, stub_embedder):
        """Source with <min_chunks AND no entities → no call at all."""
        cfg.wiki_batch_min_chunks = 3
        store = MagicMock()
        store.get_sources.return_value = [
            {"filename": "s.txt", "chunk_count": 1},
        ]
        store.get_chunks_by_source.return_value = [_chunk("s.txt", 0, "x")]
        provider = _mock_batch_provider("unused")
        with patch("lilbee.wiki.gen._generate_source_batch") as batch:
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
        drafts_dir = cfg.data_root / cfg.wiki_dir / DRAFTS_SUBDIR
        drafts_dir.mkdir(parents=True, exist_ok=True)
        # Simulate the previous failed build's marker.
        marker = drafts_dir / "henry-ford.md"
        marker.write_text(
            "<!-- PENDING: batch parse failed for source s.txt, "
            "entity/concept Henry Ford - retry -->\n"
        )
        chunks = [_chunk("s.txt", 0, "Henry Ford founded Ford Motor.")]
        entities = [_entity("henry-ford", "Henry Ford", ["s.txt"])]
        text = (
            _section("Henry Ford", "> Henry Ford founded Ford Motor. [^src1]\n")
            + _valid_citation_block()
        )
        provider = _mock_batch_provider(text)
        _generate_source_batch(
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
