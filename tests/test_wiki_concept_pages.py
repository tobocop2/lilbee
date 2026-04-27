"""Tests for the wiki link-rewriter and persist/finalize helpers.

Phase D removed the per-entity and per-concept dispatch that used to
live here (``generate_concept_page`` / ``generate_entity_page`` / the
noun-chunk-driven ``_gather_chunks_for_label`` path). The remaining
coverage targets the still-live helpers: ``_entity_surface_map``,
``_augment_surface_map_with_existing_pages``, ``_persist_and_finalize``,
``_generate_page`` progress events, and that ``build_wiki`` still
rewrites [[links]] across the wiki tree after its per-source batched
calls complete.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.core.config import cfg
from lilbee.store import SearchChunk
from lilbee.wiki.batch import _hash_existing_sources
from lilbee.wiki.entity_extractor import (
    ChunkRef,
    EntityKind,
    ExtractedEntity,
)
from lilbee.wiki.generation import (
    _augment_surface_map_with_existing_pages,
    _entity_surface_map,
    build_wiki,
)


@pytest.fixture(autouse=True)
def _stub_wiki_index_services(monkeypatch):
    """Stub ``get_services`` inside the wiki page + quality modules so tests
    that drive ``_persist_and_finalize`` don't hit the real provider when the
    wiki-body indexer or the embedding faithfulness scorer runs.
    """
    svc = MagicMock()
    svc.embedder.embed_batch.side_effect = lambda texts, **kw: [
        [0.1] * cfg.embedding_dim for _ in texts
    ]
    monkeypatch.setattr("lilbee.wiki.page.get_services", lambda: svc)
    monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)
    return svc


def _chunk(source: str, index: int, text: str) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text",
        chunk_type="raw",
        page_start=0,
        page_end=0,
        line_start=0,
        line_end=0,
        chunk=text,
        chunk_index=index,
        vector=[0.1],
    )


class TestHashExistingSources:
    def test_skips_missing_files(self, tmp_path: Path) -> None:
        (tmp_path / "present.txt").write_text("hello")
        result = _hash_existing_sources(["present.txt", "missing.txt"], tmp_path)
        assert "present.txt" in result
        assert "missing.txt" not in result

    def test_returns_hashes_for_existing(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("alpha")
        (tmp_path / "b.txt").write_text("beta")
        result = _hash_existing_sources(["a.txt", "b.txt"], tmp_path)
        assert set(result) == {"a.txt", "b.txt"}
        assert result["a.txt"] != result["b.txt"]


class TestPersistAndFinalize:
    """wiki_prune_raw drops the raw chunks once a page lands successfully."""

    def test_prune_raw_deletes_source_chunks(self, tmp_path: Path) -> None:
        from lilbee.wiki.persistence import _persist_and_finalize
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
        from lilbee.wiki.page import _generate_page

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


def _write_phase_d_sentinel(tmp_path: Path) -> None:
    """Skip the one-time migration so pre-existing concept fixtures
    stay where the test wrote them."""
    sentinel = cfg.data_dir / ".phase-d-migrated"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("skip-for-tests")


class TestBuildWikiRewritesLinks:
    """After batched generation, build_wiki rewrites slug surface forms to [[links]]."""

    def test_rewrites_existing_pages(self, tmp_path: Path) -> None:
        cfg.data_root = tmp_path
        _write_phase_d_sentinel(tmp_path)
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
                slug="henry-ford",
                kind=EntityKind.ENTITY,
                label="Henry Ford",
                type_hint="PERSON",
                chunk_refs=(ChunkRef("a.txt", 1),),
            ),
        ]
        # Prevent the per-source batch from actually calling the LLM.
        with patch("lilbee.wiki.generation._generate_source_batch", return_value=[]):
            build_wiki(entities, MagicMock(), MagicMock(), cfg)

        concept_body = (wiki_root / "concepts" / "braking.md").read_text()
        summary_body = (wiki_root / "summaries" / "manual.md").read_text()
        assert "[[henry-ford]]" in concept_body
        assert "[[braking]]" in summary_body

    def test_empty_entity_list_does_not_call_batch(self, tmp_path: Path) -> None:
        cfg.data_root = tmp_path
        (tmp_path / cfg.wiki_dir).mkdir(parents=True, exist_ok=True)
        store = MagicMock()
        store.get_sources.return_value = []
        with patch("lilbee.wiki.generation._generate_source_batch") as batch:
            build_wiki([], MagicMock(), store, cfg)
        batch.assert_not_called()

    def test_page_is_not_linked_to_itself(self, tmp_path: Path) -> None:
        """braking.md must not gain a [[braking]] self-link even while the
        rewriter is actively editing it with OTHER slugs.
        """
        cfg.data_root = tmp_path
        _write_phase_d_sentinel(tmp_path)
        wiki_root = tmp_path / cfg.wiki_dir
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "concepts" / "braking.md").write_text(
            "# Braking\n\nUsed in every henry ford design, braking systems matter.\n"
        )
        entities = [
            ExtractedEntity(
                slug="henry-ford",
                kind=EntityKind.ENTITY,
                label="Henry Ford",
                type_hint="PERSON",
                chunk_refs=(ChunkRef("a.txt", 1),),
            ),
        ]
        with patch("lilbee.wiki.generation._generate_source_batch", return_value=[]):
            build_wiki(entities, MagicMock(), MagicMock(), cfg)
        body = (wiki_root / "concepts" / "braking.md").read_text()
        assert "[[henry-ford]]" in body
        assert "[[braking]]" not in body


class TestBuildWikiDefaults:
    def test_config_none_defaults_to_cfg_singleton(self, tmp_path: Path) -> None:
        """When config=None, build_wiki must resolve to cfg."""
        cfg.data_root = tmp_path
        (tmp_path / cfg.wiki_dir).mkdir(parents=True, exist_ok=True)
        rec = ExtractedEntity(
            slug="x",
            kind=EntityKind.ENTITY,
            label="x",
            type_hint="NORP",
            chunk_refs=(ChunkRef("a.txt", 0),),
        )
        store = MagicMock()
        store.get_chunks_by_source.return_value = [_chunk("a.txt", 0, "body")]
        with patch("lilbee.wiki.generation._generate_source_batch", return_value=[]) as batch:
            build_wiki([rec], MagicMock(), store, None)
        batch.assert_called_once()
