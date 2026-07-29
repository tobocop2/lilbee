from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.core.config import cfg
from lilbee.wiki.entity_extractor import ChunkRef, EntityKind, ExtractedEntity
from lilbee.wiki.stubs import load_stub_index, refresh_stub_index, stub_index_path


@pytest.fixture(autouse=True)
def isolated_env(wiki_isolated_env: Path):
    yield wiki_isolated_env


def _entity(slug, refs):
    return ExtractedEntity(
        slug=slug,
        kind=EntityKind.ENTITY,
        label=slug,
        type_hint="ORG",
        chunk_refs=tuple(ChunkRef(source=s, chunk_index=i) for s, i in refs),
    )


def test_a_wiped_index_truncates_to_the_changed_source():
    """No index file at all is the case the guard does not cover: the corpus
    has two documents, one is re-ingested, and the tree keeps only that one."""
    cfg.wiki_entity_min_mentions = 1
    assert not stub_index_path().exists()
    store = MagicMock()
    store.get_sources.return_value = [{"filename": "a.md"}, {"filename": "b.md"}]
    store.get_chunks_by_source.return_value = []
    extractor = MagicMock()
    extractor.extract.return_value = [_entity("ford", [("a.md", 0)])]
    with (
        patch("lilbee.wiki.stubs.get_entity_extractor", return_value=extractor),
        patch("lilbee.wiki.stubs.get_services"),
    ):
        refresh_stub_index(store, cfg, sources={"a.md"})
    assert not store.get_sources.called, "went down the incremental path"
    assert list(load_stub_index()) == ["ford"]
    # b.md's subjects are absent and no later sync will revisit b.md.
