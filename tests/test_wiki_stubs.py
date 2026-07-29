"""Tests for the wiki's LLM-free page index."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.core.config import cfg
from lilbee.wiki.entity_extractor import ChunkRef, EntityKind, ExtractedEntity
from lilbee.wiki.shared import WikiSubdir
from lilbee.wiki.stubs import (
    WikiStub,
    load_stub_index,
    refresh_stub_index,
    save_stub_index,
    stub_from_entity,
    stub_index_path,
    ungenerated_stubs,
)


@pytest.fixture(autouse=True)
def isolated_env(wiki_isolated_env: Path):
    yield wiki_isolated_env


def _entity(slug: str, refs: list[tuple[str, int]], kind: EntityKind = EntityKind.ENTITY):
    return ExtractedEntity(
        slug=slug,
        kind=kind,
        label=slug.replace("-", " ").title(),
        type_hint="ORG",
        chunk_refs=tuple(ChunkRef(source=s, chunk_index=i) for s, i in refs),
    )


def _stub(slug: str, sources: tuple[str, ...] = ("a.md",), kind=EntityKind.ENTITY) -> WikiStub:
    return WikiStub(
        slug=slug,
        label=slug,
        kind=kind,
        type_hint="ORG",
        sources=sources,
        mentions=len(sources),
        chunk_refs=tuple((s, 0) for s in sources),
    )


class TestStubShape:
    @pytest.mark.parametrize(
        ("kind", "subdir"),
        [(EntityKind.ENTITY, WikiSubdir.ENTITIES), (EntityKind.CONCEPT, WikiSubdir.CONCEPTS)],
    )
    def test_kind_decides_the_subdir(self, kind: EntityKind, subdir: WikiSubdir):
        stub = _stub("ford", kind=kind)
        assert stub.subdir is subdir
        assert stub.wiki_slug == f"{subdir}/ford"

    def test_refs_are_capped_weakest_source_first(self):
        """The cap has to drop the least-supported evidence, so refs are ordered
        by how much each source has to say before the cut."""
        entity = _entity("ford", [("thin.md", 0), ("thick.md", 0), ("thick.md", 1)])
        stub = stub_from_entity(entity, cap=2)
        assert stub.chunk_refs == (("thick.md", 0), ("thick.md", 1))
        # Sources still record everything the entity was seen in.
        assert stub.sources == ("thick.md", "thin.md")


class TestRoundTrip:
    def test_index_round_trips(self):
        stubs = {"ford": _stub("ford"), "gm": _stub("gm", ("b.md",))}
        save_stub_index(stubs)
        assert load_stub_index() == stubs

    def test_missing_index_is_empty(self):
        assert load_stub_index() == {}

    @pytest.mark.parametrize(
        "payload",
        ["not json at all", '{"version": 999, "stubs": []}', '["wrong", "shape"]'],
        ids=["unparseable", "future-version", "not-an-object"],
    )
    def test_unusable_index_reads_as_empty(self, payload: str):
        """A browse must not fail because the index is corrupt or from a newer
        build; the next sync rewrites it."""
        path = stub_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        assert load_stub_index() == {}

    def test_a_bad_row_is_dropped_without_losing_the_rest(self):
        path = stub_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "stubs": [
                        {"slug": "broken"},
                        _stub("ford").to_dict(),
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert list(load_stub_index()) == ["ford"]


class TestRefresh:
    """A refresh spends no LLM call and keeps the index honest about sources."""

    @staticmethod
    def _run(entities, store=None, sources=None):
        extractor = MagicMock()
        extractor.extract.return_value = entities
        store = store or MagicMock()
        store.get_sources.return_value = []
        store.get_chunks_by_source.return_value = []
        with (
            patch("lilbee.wiki.stubs.get_entity_extractor", return_value=extractor),
            patch("lilbee.wiki.stubs.get_services"),
        ):
            return refresh_stub_index(store, cfg, sources=sources)

    def test_full_refresh_replaces_the_index(self):
        save_stub_index({"stale": _stub("stale")})
        result = self._run([_entity("ford", [("a.md", 0)])])
        assert list(result) == ["ford"]
        assert list(load_stub_index()) == ["ford"]

    def test_incremental_refresh_keeps_untouched_stubs(self):
        save_stub_index({"gm": _stub("gm", ("other.md",))})
        result = self._run([_entity("ford", [("a.md", 0)])], sources={"a.md"})
        assert sorted(result) == ["ford", "gm"]

    def test_a_source_that_stopped_naming_an_entity_drops_it(self):
        """Re-ingesting a document that no longer mentions an entity must not
        leave the entity's page in the browse tree forever."""
        save_stub_index({"ford": _stub("ford", ("a.md",))})
        result = self._run([], sources={"a.md"})
        assert result == {}

    def test_an_entity_in_several_sources_survives_one_being_reindexed(self):
        save_stub_index({"ford": _stub("ford", ("a.md", "b.md"))})
        result = self._run([], sources={"a.md"})
        assert result["ford"].sources == ("b.md",)

    def test_reindexing_merges_rather_than_duplicating(self):
        save_stub_index({"ford": _stub("ford", ("b.md",))})
        result = self._run([_entity("ford", [("a.md", 0)])], sources={"a.md"})
        assert result["ford"].sources == ("a.md", "b.md")

    def test_refresh_defaults_to_the_global_config(self):
        extractor = MagicMock()
        extractor.extract.return_value = []
        store = MagicMock()
        store.get_sources.return_value = []
        with (
            patch("lilbee.wiki.stubs.get_entity_extractor", return_value=extractor),
            patch("lilbee.wiki.stubs.get_services"),
        ):
            assert refresh_stub_index(store) == {}

    def test_refresh_spends_no_llm_call(self):
        """The whole point: the index costs an extraction pass, never a call."""
        provider = MagicMock()
        extractor = MagicMock()
        extractor.extract.return_value = [_entity("ford", [("a.md", 0)])]
        store = MagicMock()
        store.get_sources.return_value = []
        with (
            patch("lilbee.wiki.stubs.get_entity_extractor", return_value=extractor),
            patch("lilbee.wiki.stubs.get_services") as services,
        ):
            services.return_value.provider = provider
            refresh_stub_index(store, cfg)
        provider.chat.assert_not_called()


class TestUngeneratedStubs:
    def test_only_stubs_without_a_page_are_returned(self, isolated_env: Path):
        wiki_root = isolated_env / cfg.wiki_dir
        published = wiki_root / WikiSubdir.ENTITIES / "ford.md"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text("# Ford\n", encoding="utf-8")
        stubs = {"ford": _stub("ford"), "gm": _stub("gm")}
        assert [s.slug for s in ungenerated_stubs(stubs, wiki_root)] == ["gm"]
