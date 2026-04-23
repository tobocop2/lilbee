"""Tests for concept + entity page generation and the build_wiki orchestrator.

Covers ``_gather_chunks_for_label`` under reranker-on and reranker-off,
the per-source diversity cap, and that ``generate_concept_page``,
``generate_entity_page``, and ``build_wiki`` wire through the expected
subdirs and reuse the shared _generate_page pipeline.
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
    _apply_per_source_cap,
    _augment_surface_map_with_existing_pages,
    _entity_surface_map,
    _gather_chunks_for_label,
    _hash_existing_sources,
    build_wiki,
    generate_concept_page,
    generate_entity_page,
)
from lilbee.wiki.shared import CONCEPTS_SUBDIR, ENTITIES_SUBDIR


def _chunk(source: str, idx: int, text: str = "t") -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text/plain",
        page_start=1,
        page_end=1,
        line_start=1,
        line_end=1,
        chunk=text,
        chunk_index=idx,
        vector=[0.0] * 3,
    )


@pytest.fixture(autouse=True)
def _compact_cfg_defaults() -> None:
    """Shrink retrieval knobs so tiny test fixtures flow through the cap logic.

    The conftest's autouse ``_isolate_cfg`` snapshots and restores the whole
    cfg, so we only override the fields we need for these tests.
    """
    cfg.wiki_concept_max_chunks_per_page = 3
    cfg.candidate_multiplier = 2
    cfg.diversity_max_per_source = 2
    cfg.reranker_model = ""


class TestApplyPerSourceCap:
    def test_zero_cap_keeps_everything(self) -> None:
        chunks = [_chunk("a.txt", 0), _chunk("a.txt", 1), _chunk("b.txt", 0)]
        assert _apply_per_source_cap(chunks, 0) == chunks

    def test_cap_of_one_keeps_first_chunk_per_source(self) -> None:
        chunks = [
            _chunk("a.txt", 0),
            _chunk("a.txt", 1),
            _chunk("b.txt", 0),
            _chunk("b.txt", 1),
        ]
        result = _apply_per_source_cap(chunks, 1)
        assert [(c.source, c.chunk_index) for c in result] == [
            ("a.txt", 0),
            ("b.txt", 0),
        ]

    def test_order_preserved_within_cap(self) -> None:
        chunks = [
            _chunk("a.txt", 3),
            _chunk("a.txt", 1),
            _chunk("a.txt", 2),
        ]
        result = _apply_per_source_cap(chunks, 2)
        assert [c.chunk_index for c in result] == [3, 1]


class TestGatherChunksForLabel:
    def test_empty_label_returns_empty(self) -> None:
        assert _gather_chunks_for_label("  ", MagicMock(), MagicMock(), cfg) == []

    def test_embed_failure_returns_empty(self) -> None:
        provider = MagicMock()
        provider.embed.side_effect = RuntimeError("embed crashed")
        store = MagicMock()
        assert _gather_chunks_for_label("braking", provider, store, cfg) == []
        store.search.assert_not_called()

    def test_empty_embed_result_returns_empty(self) -> None:
        provider = MagicMock()
        provider.embed.return_value = []
        assert _gather_chunks_for_label("braking", provider, MagicMock(), cfg) == []

    def test_no_candidates_returns_empty(self) -> None:
        provider = MagicMock()
        provider.embed.return_value = [[0.1, 0.2, 0.3]]
        store = MagicMock()
        store.search.return_value = []
        assert _gather_chunks_for_label("braking", provider, store, cfg) == []

    def test_reranker_off_respects_hybrid_order_and_cap(self) -> None:
        cfg.reranker_model = ""
        provider = MagicMock()
        provider.embed.return_value = [[0.1, 0.2, 0.3]]
        store = MagicMock()
        store.search.return_value = [
            _chunk("a.txt", 0),
            _chunk("a.txt", 1),
            _chunk("a.txt", 2),
            _chunk("b.txt", 0),
        ]
        result = _gather_chunks_for_label("braking", provider, store, cfg)
        assert [(c.source, c.chunk_index) for c in result] == [
            ("a.txt", 0),
            ("a.txt", 1),
            ("b.txt", 0),
        ]
        provider.rerank.assert_not_called()

    def test_reranker_on_reorders_by_score(self) -> None:
        cfg.reranker_model = "some-rerank-model"
        provider = MagicMock()
        provider.embed.return_value = [[0.1, 0.2, 0.3]]
        provider.supports_rerank.return_value = True
        provider.rerank.return_value = [0.1, 0.9, 0.5]
        store = MagicMock()
        store.search.return_value = [
            _chunk("a.txt", 0, "A"),
            _chunk("b.txt", 0, "B"),
            _chunk("c.txt", 0, "C"),
        ]
        result = _gather_chunks_for_label("braking", provider, store, cfg)
        assert [c.source for c in result] == ["b.txt", "c.txt", "a.txt"]
        provider.rerank.assert_called_once()

    def test_reranker_on_but_unsupported_falls_back(self) -> None:
        cfg.reranker_model = "x"
        provider = MagicMock()
        provider.embed.return_value = [[0.1, 0.2, 0.3]]
        provider.supports_rerank.return_value = False
        store = MagicMock()
        store.search.return_value = [_chunk("a.txt", 0), _chunk("b.txt", 0)]
        result = _gather_chunks_for_label("q", provider, store, cfg)
        assert [c.source for c in result] == ["a.txt", "b.txt"]
        provider.rerank.assert_not_called()

    def test_rerank_exception_falls_back_to_hybrid_order(self) -> None:
        cfg.reranker_model = "x"
        provider = MagicMock()
        provider.embed.return_value = [[0.1, 0.2, 0.3]]
        provider.supports_rerank.return_value = True
        provider.rerank.side_effect = RuntimeError("boom")
        store = MagicMock()
        store.search.return_value = [_chunk("a.txt", 0), _chunk("b.txt", 0)]
        result = _gather_chunks_for_label("q", provider, store, cfg)
        assert [c.source for c in result] == ["a.txt", "b.txt"]

    def test_rerank_returning_wrong_length_is_ignored(self) -> None:
        cfg.reranker_model = "x"
        provider = MagicMock()
        provider.embed.return_value = [[0.1, 0.2, 0.3]]
        provider.supports_rerank.return_value = True
        provider.rerank.return_value = [0.9]  # wrong length
        store = MagicMock()
        store.search.return_value = [_chunk("a.txt", 0), _chunk("b.txt", 0)]
        result = _gather_chunks_for_label("q", provider, store, cfg)
        assert [c.source for c in result] == ["a.txt", "b.txt"]


class TestGeneratePages:
    def test_concept_page_routes_to_concepts_subdir(self, tmp_path: Path) -> None:
        provider = MagicMock()
        store = MagicMock()
        sentinel = tmp_path / "out.md"
        with (
            patch(
                "lilbee.wiki.gen._gather_chunks_for_label",
                return_value=[_chunk("a.txt", 0, "body")],
            ),
            patch("lilbee.wiki.gen._generate_page", return_value=sentinel) as gen,
        ):
            result = generate_concept_page("braking systems", provider, store, cfg)
        assert result is sentinel
        kwargs = gen.call_args.kwargs
        assert kwargs["page_type"] == CONCEPTS_SUBDIR
        assert kwargs["slug"] == "braking-systems"
        assert kwargs["label"] == "braking systems"
        # The rendered prompt must know we're writing a concept, not an entity.
        assert "Kind: concept" in kwargs["prompt"]

    def test_entity_page_routes_to_entities_subdir(self, tmp_path: Path) -> None:
        provider = MagicMock()
        store = MagicMock()
        sentinel = tmp_path / "out.md"
        with (
            patch(
                "lilbee.wiki.gen._gather_chunks_for_label",
                return_value=[_chunk("hist.txt", 0, "body")],
            ),
            patch("lilbee.wiki.gen._generate_page", return_value=sentinel) as gen,
        ):
            result = generate_entity_page("Henry Ford", provider, store, cfg)
        assert result is sentinel
        kwargs = gen.call_args.kwargs
        assert kwargs["page_type"] == ENTITIES_SUBDIR
        assert kwargs["slug"] == "henry-ford"
        assert "Kind: entity" in kwargs["prompt"]

    def test_no_chunks_skips_page(self) -> None:
        with (
            patch("lilbee.wiki.gen._gather_chunks_for_label", return_value=[]),
            patch("lilbee.wiki.gen._generate_page") as gen,
        ):
            assert generate_concept_page("x", MagicMock(), MagicMock(), cfg) is None
        gen.assert_not_called()

    def test_source_names_come_from_chunks_and_are_sorted(self, tmp_path: Path) -> None:
        sentinel = tmp_path / "o.md"
        with (
            patch(
                "lilbee.wiki.gen._gather_chunks_for_label",
                return_value=[
                    _chunk("z.txt", 0, "foo"),
                    _chunk("a.txt", 0, "bar"),
                ],
            ),
            patch("lilbee.wiki.gen._generate_page", return_value=sentinel) as gen,
        ):
            generate_concept_page("topic", MagicMock(), MagicMock(), cfg)
        assert gen.call_args.kwargs["source_names"] == ["a.txt", "z.txt"]


class TestHashExistingSources:
    def test_hashes_only_files_that_exist(self, tmp_path: Path) -> None:
        docs = tmp_path / "documents"
        docs.mkdir()
        (docs / "here.txt").write_text("content")
        result = _hash_existing_sources(["here.txt", "missing.txt"], docs)
        assert "here.txt" in result
        assert result["here.txt"]  # non-empty hash
        assert "missing.txt" not in result

    def test_empty_list_returns_empty(self, tmp_path: Path) -> None:
        assert _hash_existing_sources([], tmp_path) == {}


class TestConceptPageWiresResolver:
    """The resolver passed to _generate_page must curry the multi-source bindings."""

    def test_resolver_binds_sources_and_hashes(self, tmp_path: Path) -> None:
        cfg.documents_dir = tmp_path / "documents"
        cfg.documents_dir.mkdir()
        (cfg.documents_dir / "a.txt").write_text("hello")
        captured: dict = {}

        def fake_generate_page(**kwargs: object) -> None:
            captured.update(kwargs)
            return None

        with (
            patch(
                "lilbee.wiki.gen._gather_chunks_for_label",
                return_value=[_chunk("a.txt", 0, "hello")],
            ),
            patch("lilbee.wiki.gen._generate_page", side_effect=fake_generate_page),
        ):
            generate_concept_page("topic", MagicMock(), MagicMock(), cfg)
        resolver = captured["citation_resolver"]
        # resolver is a functools.partial pre-bound with source_names + source_hashes
        # so calling it with an empty parsed-citation list returns an empty list
        # without raising (no chat or store calls required).
        assert resolver([]) == []


class TestPersistAndFinalize:
    """wiki_prune_raw drops the raw chunks once a page lands successfully."""

    def test_prune_raw_deletes_source_chunks(self, tmp_path: Path) -> None:
        from lilbee.wiki.gen import _persist_and_finalize
        from lilbee.wiki.shared import PageTarget

        cfg.wiki_prune_raw = True
        cfg.data_root = tmp_path
        wiki_root = tmp_path / cfg.wiki_dir
        wiki_root.mkdir(parents=True)
        target = PageTarget(
            wiki_root=wiki_root,
            subdir="concepts",
            slug="braking",
            wiki_source=f"{cfg.wiki_dir}/concepts/braking.md",
            page_type="concept",
            label="braking",
        )
        store = MagicMock()
        _persist_and_finalize(
            "# braking\n\nbody.\n",
            target,
            verified=[],
            source_names=["a.txt", "b.txt"],
            store=store,
            config=cfg,
        )
        store.delete_by_source.assert_any_call("a.txt")
        store.delete_by_source.assert_any_call("b.txt")


class TestGeneratePageProgress:
    """_generate_page forwards progress events to the on_progress callback."""

    def test_progress_callback_receives_generating_stage(self, tmp_path: Path) -> None:
        from lilbee.wiki.gen import _generate_page

        cfg.data_root = tmp_path
        (tmp_path / cfg.wiki_dir).mkdir(parents=True, exist_ok=True)
        events: list[tuple[str, dict]] = []
        provider = MagicMock()
        provider.get_capabilities.return_value = []
        provider.chat.side_effect = RuntimeError("simulated")
        _generate_page(
            label="topic",
            prompt="p",
            chunks=[_chunk("a.txt", 0, "body")],
            chunks_text="body",
            citation_resolver=lambda _: [],
            page_type="concepts",
            slug="topic",
            source_names=["a.txt"],
            provider=provider,
            store=MagicMock(),
            config=cfg,
            on_progress=lambda stage, data: events.append((stage, data)),
        )
        stages = [stage for stage, _ in events]
        assert "preparing" in stages
        assert "generating" in stages


class TestSurfaceMapHelpers:
    def test_entity_surface_map_includes_label_and_spaced_slug(self) -> None:
        entities = [
            ExtractedEntity(
                slug="henry-ford",
                kind=EntityKind.ENTITY,
                label="Henry Ford",
                type_hint="PERSON",
                chunk_refs=(ChunkRef("a.txt", 0),),
            ),
        ]
        result = _entity_surface_map(entities)
        assert result == {"Henry Ford": "henry-ford", "henry ford": "henry-ford"}

    def test_entity_surface_map_single_word_slug_skips_spaced(self) -> None:
        entities = [
            ExtractedEntity(
                slug="ford",
                kind=EntityKind.ENTITY,
                label="ford",
                type_hint="PERSON",
                chunk_refs=(ChunkRef("a.txt", 0),),
            ),
        ]
        assert _entity_surface_map(entities) == {"ford": "ford"}

    def test_augment_adds_spaced_slug_for_each_existing_page(self, tmp_path: Path) -> None:
        wiki_root = tmp_path / "wiki"
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "entities").mkdir()
        (wiki_root / "concepts" / "tire-pressure.md").write_text("x\n")
        (wiki_root / "entities" / "henry-ford.md").write_text("x\n")
        mapping: dict[str, str] = {}
        _augment_surface_map_with_existing_pages(mapping, wiki_root)
        assert mapping == {
            "tire pressure": "tire-pressure",
            "henry ford": "henry-ford",
        }

    def test_augment_preserves_existing_entries(self, tmp_path: Path) -> None:
        """Entity labels from the current build should win over on-disk spaced forms."""
        wiki_root = tmp_path / "wiki"
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "concepts" / "tire-pressure.md").write_text("x\n")
        mapping = {"tire pressure": "custom-slug"}
        _augment_surface_map_with_existing_pages(mapping, wiki_root)
        assert mapping["tire pressure"] == "custom-slug"


class TestBuildWikiRewritesLinks:
    """After generating pages, build_wiki rewrites slug surface forms to [[links]]."""

    def test_rewrites_new_and_existing_pages(self, tmp_path: Path) -> None:
        cfg.data_root = tmp_path
        wiki_root = tmp_path / cfg.wiki_dir
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "summaries").mkdir()
        (wiki_root / "concepts" / "braking.md").write_text(
            "# braking\n\nSee henry ford for more.\n"
        )
        (wiki_root / "summaries" / "manual.md").write_text(
            "# Manual\n\nThis book discusses braking thoroughly.\n"
        )
        entities = [
            ExtractedEntity(
                slug="braking",
                kind=EntityKind.CONCEPT,
                label="braking",
                type_hint="noun_phrase",
                chunk_refs=(ChunkRef("a.txt", 0),),
            ),
            ExtractedEntity(
                slug="henry-ford",
                kind=EntityKind.ENTITY,
                label="Henry Ford",
                type_hint="PERSON",
                chunk_refs=(ChunkRef("a.txt", 1),),
            ),
        ]
        with (
            patch("lilbee.wiki.gen.generate_concept_page", return_value=None),
            patch("lilbee.wiki.gen.generate_entity_page", return_value=None),
        ):
            build_wiki(entities, MagicMock(), MagicMock(), cfg)

        concept_body = (wiki_root / "concepts" / "braking.md").read_text()
        summary_body = (wiki_root / "summaries" / "manual.md").read_text()
        assert "[[henry-ford]]" in concept_body
        assert "[[braking]]" in summary_body

    def test_empty_entity_list_is_noop(self, tmp_path: Path) -> None:
        cfg.data_root = tmp_path
        with patch("lilbee.wiki.gen.generate_concept_page") as gc:
            build_wiki([], MagicMock(), MagicMock(), cfg)
        gc.assert_not_called()

    def test_page_is_not_linked_to_itself(self, tmp_path: Path) -> None:
        """braking.md must not gain a [[braking]] self-link even while the
        rewriter is actively editing it with OTHER slugs.

        Two entities so the owning-slug filter is actually exercised:
        without both, ``if not page_map: continue`` short-circuits
        before the regex ever runs and the assertion is satisfied for
        the wrong reason.
        """
        cfg.data_root = tmp_path
        wiki_root = tmp_path / cfg.wiki_dir
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "concepts" / "braking.md").write_text(
            "# Braking\n\nUsed in every henry ford design, braking systems matter.\n"
        )
        entities = [
            ExtractedEntity(
                slug="braking",
                kind=EntityKind.CONCEPT,
                label="braking",
                type_hint="noun_phrase",
                chunk_refs=(ChunkRef("a.txt", 0),),
            ),
            ExtractedEntity(
                slug="henry-ford",
                kind=EntityKind.ENTITY,
                label="Henry Ford",
                type_hint="PERSON",
                chunk_refs=(ChunkRef("a.txt", 1),),
            ),
        ]
        with (
            patch("lilbee.wiki.gen.generate_concept_page", return_value=None),
            patch("lilbee.wiki.gen.generate_entity_page", return_value=None),
        ):
            build_wiki(entities, MagicMock(), MagicMock(), cfg)
        body = (wiki_root / "concepts" / "braking.md").read_text()
        # Proves the rewriter did run (it emitted at least one other link),
        # and that the owning slug was correctly filtered.
        assert "[[henry-ford]]" in body
        assert "[[braking]]" not in body

    def test_empty_page_map_short_circuits(self, tmp_path: Path) -> None:
        """When the only slug in the surface map is the owning slug, the
        rewriter skips the file entirely rather than rewriting with an
        empty map. Ensures the ``if not page_map: continue`` guard runs.
        """
        cfg.data_root = tmp_path
        wiki_root = tmp_path / cfg.wiki_dir
        (wiki_root / "concepts").mkdir(parents=True)
        original_body = "# Braking\n\nAll about braking systems.\n"
        (wiki_root / "concepts" / "braking.md").write_text(original_body)
        entities = [
            ExtractedEntity(
                slug="braking",
                kind=EntityKind.CONCEPT,
                label="braking",
                type_hint="noun_phrase",
                chunk_refs=(ChunkRef("a.txt", 0),),
            ),
        ]
        with patch("lilbee.wiki.gen.generate_concept_page", return_value=None):
            build_wiki(entities, MagicMock(), MagicMock(), cfg)
        # File content unchanged: rewriter saw an empty page_map after
        # filtering the owning slug and short-circuited before reading.
        assert (wiki_root / "concepts" / "braking.md").read_text() == original_body

    def test_incremental_rebuild_links_to_existing_on_disk_slugs(self, tmp_path: Path) -> None:
        """A touched entity's page links to pre-existing slugs not in the touched set."""
        cfg.data_root = tmp_path
        wiki_root = tmp_path / cfg.wiki_dir
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "entities").mkdir()
        # engine.md already exists on disk from a prior build.
        (wiki_root / "entities" / "henry-ford.md").write_text("# Henry Ford\n\nA person.\n")
        # The freshly regenerated page mentions Henry Ford in its body.
        (wiki_root / "concepts" / "braking.md").write_text(
            "# Braking\n\nInvented during henry ford's era.\n"
        )
        entities = [
            ExtractedEntity(
                slug="braking",
                kind=EntityKind.CONCEPT,
                label="braking",
                type_hint="noun_phrase",
                chunk_refs=(ChunkRef("a.txt", 0),),
            )
        ]
        with patch("lilbee.wiki.gen.generate_concept_page", return_value=None):
            build_wiki(entities, MagicMock(), MagicMock(), cfg)
        braking_body = (wiki_root / "concepts" / "braking.md").read_text()
        assert "[[henry-ford]]" in braking_body


class TestBuildWiki:
    def test_dispatches_concept_and_entity_records(self, tmp_path: Path) -> None:
        concept_rec = ExtractedEntity(
            slug="braking",
            kind=EntityKind.CONCEPT,
            label="braking",
            type_hint="noun_phrase",
            chunk_refs=(ChunkRef("a.txt", 0),),
        )
        entity_rec = ExtractedEntity(
            slug="henry-ford",
            kind=EntityKind.ENTITY,
            label="Henry Ford",
            type_hint="PERSON",
            chunk_refs=(ChunkRef("a.txt", 1),),
        )
        concept_path = tmp_path / "concepts" / "braking.md"
        entity_path = tmp_path / "entities" / "henry-ford.md"
        with (
            patch("lilbee.wiki.gen.generate_concept_page", return_value=concept_path) as gc,
            patch("lilbee.wiki.gen.generate_entity_page", return_value=entity_path) as ge,
        ):
            result = build_wiki([concept_rec, entity_rec], MagicMock(), MagicMock(), cfg)
        assert result == [concept_path, entity_path]
        gc.assert_called_once()
        ge.assert_called_once()
        assert gc.call_args.args[0] == "braking"
        assert ge.call_args.args[0] == "Henry Ford"

    def test_none_return_is_skipped(self) -> None:
        rec = ExtractedEntity(
            slug="x",
            kind=EntityKind.CONCEPT,
            label="x",
            type_hint="noun_phrase",
            chunk_refs=(ChunkRef("a.txt", 0),),
        )
        with patch("lilbee.wiki.gen.generate_concept_page", return_value=None):
            assert build_wiki([rec], MagicMock(), MagicMock(), cfg) == []

    def test_defaults_config_to_cfg_singleton(self) -> None:
        """When config=None, build_wiki must resolve to the cfg singleton
        and pass it through to every downstream call site. An empty
        entity list can't prove the fallback works because the for-loop
        would short-circuit either way; feed one entity so the
        generator receives cfg explicitly.
        """
        rec = ExtractedEntity(
            slug="x",
            kind=EntityKind.CONCEPT,
            label="x",
            type_hint="noun_phrase",
            chunk_refs=(ChunkRef("a.txt", 0),),
        )
        with patch("lilbee.wiki.gen.generate_concept_page", return_value=None) as gc:
            build_wiki([rec], MagicMock(), MagicMock(), None)
        gc.assert_called_once()
        assert gc.call_args.args[-1] is cfg
