"""A wiki page records the id of the document it cites, not the id of a page of the same name."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import SourceType, Store
from lilbee.wiki.entity_extractor import ChunkRef, EntityKind, ExtractedEntity
from lilbee.wiki.persistence import persist_and_finalize, recorded_source_ids
from lilbee.wiki.shared import PageTarget, WikiSubdir
from lilbee.wiki.stubs import load_stub_index, refresh_stub_index
from lilbee.wiki.synthesis import generate_source_batch
from tests.test_source_writes import _SHARED, _batch, _ids, _replace
from tests.test_wiki_batch_generation import (
    _chunk,
    _declare,
    _mock_batch_provider,
)


@pytest.fixture()
def shared(wiki_isolated_env: Path) -> Store:
    """A store with a document under a folder labelled ``wiki`` and a wiki page of its name."""
    cfg.linked_roots = {"wiki": str(wiki_isolated_env / "home" / "wiki")}
    store = Store(cfg)
    _batch(store, _SHARED, SourceType.DOCUMENT)
    _replace(store, _SHARED, SourceType.WIKI)
    return store


@pytest.fixture()
def embedder(monkeypatch):
    """Embeds every text to one vector, so a page is scored without a model."""
    svc = MagicMock()
    svc.embedder.embed_batch.side_effect = lambda texts, **kw: [
        [0.1] * cfg.embedding_dim for _ in texts
    ]
    monkeypatch.setattr("lilbee.wiki.page.get_services", lambda: svc)
    monkeypatch.setattr("lilbee.wiki.quality.get_services", lambda: svc)


def _draft_target(slug: str) -> PageTarget:
    return PageTarget(
        wiki_root=cfg.data_root / cfg.wiki_dir,
        subdir=WikiSubdir.DRAFTS,
        slug=slug,
        wiki_source=f"{cfg.wiki_dir}/drafts/{slug}.md",
        page_type=WikiSubdir.CONCEPTS,
        label=slug,
    )


class TestADraftRecordsTheDocumentItCites:
    def test_a_draft_written_after_the_quality_gate(self, shared):
        document = _ids(shared)[SourceType.DOCUMENT.value]
        page = _ids(shared)[SourceType.WIKI.value]
        assert document != page

        path = persist_and_finalize(
            f'---\nsources: ["{_SHARED}"]\n---\n\n# Brakes\n\nBody that failed the gate.\n',
            _draft_target("brakes"),
            [],
            [_SHARED],
            shared,
            cfg,
        )

        assert recorded_source_ids(path.read_text(encoding="utf-8")) == {_SHARED: document}

    def test_a_collision_marker_of_a_second_source(self, shared, embedder):
        document = _ids(shared)[SourceType.DOCUMENT.value]
        page = _ids(shared)[SourceType.WIKI.value]
        assert document != page
        # The pages now live elsewhere, so a citation of the old name is not wiki-sourced.
        cfg.wiki_dir = "notes"

        def _text(source: str) -> str:
            return (
                _declare("Brake System") + "## Brake System\n\n> Brake system details. [^src1]\n"
                "\n\n---\n"
                "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
                f'[^src1]: {source}, excerpt: "Brake system details."\n'
            )

        written: dict[str, str] = {}
        for source, store in (("first.txt", MagicMock()), (_SHARED, shared)):
            generate_source_batch(
                source=source,
                entities=[],
                chunks=[_chunk(source, 0, "Brake system details.")],
                provider=_mock_batch_provider(_text(source)),
                store=store,
                config=cfg,
                extract_concepts=True,
                written_concept_slugs=written,
            )

        drafts = cfg.data_root / cfg.wiki_dir / WikiSubdir.DRAFTS
        [marker] = list(drafts.glob("brake-system-collision-*.md"))
        assert recorded_source_ids(marker.read_text(encoding="utf-8")) == {_SHARED: document}


class _NamesNothing:
    """An extractor that finds no subject in any chunk."""

    def extract(self, chunks):
        return []

    def available(self, quiet: bool = False) -> bool:
        return True


class _NamesBoeing(_NamesNothing):
    def extract(self, chunks):
        refs = tuple(ChunkRef(source=chunk.source, chunk_index=0) for chunk in chunks)
        return [
            ExtractedEntity(
                slug="boeing",
                kind=EntityKind.ENTITY,
                label="Boeing",
                type_hint="ORG",
                chunk_refs=refs,
            )
        ]


class TestAStubIndexFollowsTheDocumentItLists:
    def test_a_document_that_no_longer_names_a_subject_drops_the_stub(self, shared, monkeypatch):
        cfg.wiki_entity_min_mentions = 1
        monkeypatch.setattr("lilbee.wiki.stubs._available_extractor", lambda config: _NamesBoeing())
        refresh_stub_index(shared, cfg)
        assert list(load_stub_index(cfg)) == ["boeing"]

        monkeypatch.setattr(
            "lilbee.wiki.stubs._available_extractor", lambda config: _NamesNothing()
        )
        refresh_stub_index(shared, cfg, sources={_SHARED})

        assert load_stub_index(cfg) == {}
