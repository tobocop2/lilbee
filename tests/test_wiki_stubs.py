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
    _recut_refs,
    drop_sources_from_index,
    load_stub_index,
    refresh_stub_index,
    save_stub_index,
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


class _FakeStore:
    """A store that serves chunks and persists wiki mention rows in memory, so a
    refresh can write per-source evidence and aggregate it back the way the real
    store does. The mention table is the source of truth the index derives from,
    so a MagicMock (which never persists) would leave the whole flow unobserved.
    """

    def __init__(self, chunks_by_source: dict[str, list]):
        self._chunks = chunks_by_source
        self._mentions: dict[str, list[dict]] = {}

    def get_sources(self):
        return [{"filename": name} for name in sorted(self._chunks)]

    def get_chunks_by_source(self, name):
        return self._chunks.get(name, [])

    def replace_wiki_mentions_for_source(self, source, rows):
        if rows:
            self._mentions[source] = [dict(r) for r in rows]
        else:
            self._mentions.pop(source, None)

    def wiki_mention_rows(self, slugs=None):
        rows = [dict(r) for src_rows in self._mentions.values() for r in src_rows]
        if slugs is not None:
            wanted = set(slugs)
            rows = [r for r in rows if r["slug"] in wanted]
        return rows

    def has_wiki_mentions(self):
        return any(self._mentions.values())

    def clear_wiki_mentions(self):
        self._mentions = {}
        return True


def _seed_full(store: _FakeStore, entities: list[ExtractedEntity]) -> None:
    """Run a full refresh that writes *entities* into *store* and the index,
    so both the mention table and the file reflect a real prior sync."""
    for entity in entities:
        for ref in entity.chunk_refs:
            store._chunks.setdefault(ref.source, []).append(MagicMock(source=ref.source))

    def _extract(chunks):
        served = {chunk.source for chunk in chunks}
        return [e for e in entities if any(r.source in served for r in e.chunk_refs)]

    extractor = MagicMock()
    extractor.available.return_value = True
    extractor.extract.side_effect = _extract
    with (
        patch("lilbee.wiki.stubs.get_entity_extractor", return_value=extractor),
        patch("lilbee.wiki.stubs.get_services"),
    ):
        refresh_stub_index(store, cfg, sources=None)


def _stub(
    slug: str,
    sources: tuple[str, ...] = ("a.md",),
    kind=EntityKind.ENTITY,
    per_source: int = 1,
) -> WikiStub:
    return WikiStub(
        slug=slug,
        label=slug,
        kind=kind,
        type_hint="ORG",
        source_mentions=tuple((s, per_source) for s in sources),
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
        stub = WikiStub(
            slug="ford",
            label="ford",
            kind=EntityKind.ENTITY,
            type_hint="ORG",
            source_mentions=(("thick.md", 2), ("thin.md", 1)),
            chunk_refs=(("thin.md", 0), ("thick.md", 0), ("thick.md", 1)),
        )
        capped = _recut_refs(stub, cap=2)
        assert capped.chunk_refs == (("thick.md", 0), ("thick.md", 1))
        # Sources still record everything the entity was seen in.
        assert capped.sources == ("thick.md", "thin.md")


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

    @pytest.mark.parametrize(
        ("payload", "readable"),
        [
            ({"version": 3, "stubs": [{"slug": "broken"}, {"nope": 1}]}, False),
            ({"version": 3}, False),
            ({"version": 3, "stubs": "garbage"}, False),
            ({"version": 3, "stubs": []}, True),
        ],
        ids=["all-rows-bad", "no-stubs-key", "stubs-not-a-list", "genuinely-empty"],
    )
    def test_damaged_and_empty_indexes_are_told_apart(self, payload: dict, readable: bool):
        """Only an explicit empty list describes a corpus that names nothing.
        The rest are damaged, and calling them empty would let an incremental
        refresh build on them and drop everything the sync did not touch."""
        from lilbee.wiki.stubs import _read_stub_index

        path = stub_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert (_read_stub_index() is not None) is readable

    def test_a_bad_row_is_dropped_without_losing_the_rest(self):
        path = stub_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": 3,
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

    @pytest.fixture(autouse=True)
    def _low_threshold(self):
        """Most cases here are about merge arithmetic, not the threshold; the
        cases that are about it set their own."""
        cfg.wiki_entity_min_mentions = 1

    @staticmethod
    def _run(entities, store=None, sources=None):
        """Refresh with the extractor reading the chunks it is actually handed.

        The store serves one chunk per source an entity names and persists the
        mention rows the refresh writes; the extractor returns only the entities
        whose sources are represented in its argument. Pass the returned store
        back in to chain an incremental refresh onto a seeded one, since the
        store, not the file, is now what an incremental pass builds on.
        """
        if store is None:
            store = _FakeStore({})
        for entity in entities:
            for ref in entity.chunk_refs:
                chunk = MagicMock()
                chunk.source = ref.source
                store._chunks.setdefault(ref.source, []).append(chunk)

        def _extract(chunks):
            served = {chunk.source for chunk in chunks}
            return [e for e in entities if any(r.source in served for r in e.chunk_refs)]

        extractor = MagicMock()
        extractor.extract.side_effect = _extract
        with (
            patch("lilbee.wiki.stubs.get_entity_extractor", return_value=extractor),
            patch("lilbee.wiki.stubs.get_services"),
        ):
            refresh_stub_index(store, cfg, sources=sources)
        return load_stub_index()

    def test_full_refresh_replaces_the_index(self):
        save_stub_index({"stale": _stub("stale")})
        result = self._run([_entity("ford", [("a.md", 0)])])
        assert list(result) == ["ford"]
        assert list(load_stub_index()) == ["ford"]

    def test_an_unavailable_extractor_leaves_the_index_untouched(self):
        """An unavailable backend extracts nothing, which looks the same as a
        corpus that names nothing. Persisting that would overwrite a good index
        with an empty one and every surface would report success."""
        save_stub_index({"ford": _stub("ford")})
        extractor = MagicMock()
        extractor.available.return_value = False
        extractor.extract.return_value = []
        store = MagicMock()
        store.get_sources.return_value = []
        store.get_chunks_by_source.return_value = []
        with (
            patch("lilbee.wiki.stubs.get_entity_extractor", return_value=extractor),
            patch("lilbee.wiki.stubs.get_services"),
        ):
            result = refresh_stub_index(store, cfg, sources=None)
        assert list(result) == ["ford"]
        assert list(load_stub_index()) == ["ford"]
        extractor.extract.assert_not_called()

    def test_incremental_refresh_keeps_untouched_stubs(self):
        store = _FakeStore({})
        self._run([_entity("gm", [("other.md", 0)])], store=store)
        result = self._run([_entity("ford", [("a.md", 0)])], store=store, sources={"a.md"})
        assert sorted(result) == ["ford", "gm"]

    def test_a_subject_recovers_once_its_evidence_spans_two_synced_documents(self):
        """The headline fix. A subject below the floor in each separately-synced
        document still appears once the store holds all its evidence. The old
        file-only index dropped a.md's sub-floor mentions at write time, so
        b.md's later sync had nothing to accumulate into and the subject was
        lost for good."""
        cfg.wiki_entity_min_mentions = 3
        store = _FakeStore({})
        after_a = self._run(
            [_entity("boeing", [("a.md", 0), ("a.md", 1)])], store=store, sources={"a.md"}
        )
        assert after_a == {}  # two mentions is below the floor on a.md's own
        after_b = self._run(
            [_entity("boeing", [("b.md", 0), ("b.md", 1)])], store=store, sources={"b.md"}
        )
        assert "boeing" in after_b  # four across the corpus clears it
        assert after_b["boeing"].mentions == 4
        assert after_b["boeing"].sources == ("a.md", "b.md")

    def test_an_index_that_is_genuinely_empty_does_not_force_a_rebuild(self):
        """A corpus that names nothing indexes to an empty but valid file.
        Treating that as unreadable would re-scan every chunk of every
        document on every sync, forever, and re-arm itself each time."""
        save_stub_index({})
        store = MagicMock()
        store.get_sources.return_value = []
        store.get_chunks_by_source.return_value = []
        extractor = MagicMock()
        extractor.extract.return_value = []
        with (
            patch("lilbee.wiki.stubs.get_entity_extractor", return_value=extractor),
            patch("lilbee.wiki.stubs.get_services"),
        ):
            refresh_stub_index(store, cfg, sources={"a.md"})
        store.get_sources.assert_not_called()

    def test_the_incremental_pass_does_not_pre_filter_by_the_threshold(self):
        """The extractor drops entities under min_mentions from whatever chunks
        it is given. On an incremental pass that is one document's chunks, so
        the pass must run with the floor lifted and the merged total judged
        instead. Asserted on the config the extractor is actually built with."""
        cfg.wiki_entity_min_mentions = 4
        save_stub_index({"ford": _stub("ford", ("b.md",), per_source=9)})
        seen: list[int] = []

        def fake_get_extractor(mode, provider, config):
            seen.append(config.wiki_entity_min_mentions)
            extractor = MagicMock()
            extractor.extract.return_value = [_entity("ford", [("a.md", 0)])]
            return extractor

        store = MagicMock()
        store.get_sources.return_value = []
        store.get_chunks_by_source.return_value = []
        with (
            patch("lilbee.wiki.stubs.get_entity_extractor", fake_get_extractor),
            patch("lilbee.wiki.stubs.get_services"),
        ):
            refresh_stub_index(store, cfg, sources={"a.md"})
        assert seen == [1]

    def test_extraction_always_uses_a_floor_of_one(self):
        """The store keeps every mention so the corpus-wide floor can be judged
        over the aggregate; the extractor must therefore run at a floor of one,
        whatever the configured threshold, or sub-floor evidence never reaches
        the store to accumulate across syncs."""
        cfg.wiki_entity_min_mentions = 4
        seen: list[int] = []

        def fake_get_extractor(mode, provider, config):
            seen.append(config.wiki_entity_min_mentions)
            extractor = MagicMock()
            extractor.available.return_value = True
            extractor.extract.return_value = []
            return extractor

        store = _FakeStore({})
        with (
            patch("lilbee.wiki.stubs.get_entity_extractor", fake_get_extractor),
            patch("lilbee.wiki.stubs.get_services"),
        ):
            refresh_stub_index(store, cfg)
        assert seen == [1]

    def test_an_empty_store_rebuilds_in_full_rather_than_truncating(self):
        """An incremental pass over a store with no mention rows -- a cold or
        file-only-migrated store -- would derive an empty index. It must rebuild
        in full to seed the table, not index only the changed sources and call
        that the whole corpus."""
        store = _FakeStore({})
        # b.md is already in the corpus but is not the changed source.
        for src in ("a.md", "b.md"):
            store._chunks.setdefault(src, []).append(MagicMock(source=src))

        entities = [_entity("ford", [("a.md", 0)]), _entity("gm", [("b.md", 0)])]

        def _extract(chunks):
            served = {c.source for c in chunks}
            return [e for e in entities if any(r.source in served for r in e.chunk_refs)]

        extractor = MagicMock()
        extractor.available.return_value = True
        extractor.extract.side_effect = _extract
        with (
            patch("lilbee.wiki.stubs.get_entity_extractor", return_value=extractor),
            patch("lilbee.wiki.stubs.get_services"),
        ):
            refresh_stub_index(store, cfg, sources={"a.md"})
        # The full corpus was indexed, not just the changed source.
        assert sorted(load_stub_index()) == ["ford", "gm"]

    def test_the_mention_threshold_is_judged_across_the_whole_corpus(self):
        """An entity named twice in the document being re-indexed and often
        elsewhere must not lose that document. Applying the threshold to one
        source's chunks would erode the entry on every later sync."""
        cfg.wiki_entity_min_mentions = 3
        store = _FakeStore({})
        self._run([_entity("ford", [("b.md", i) for i in range(40)])], store=store)
        result = self._run(
            [_entity("ford", [("a.md", 0), ("a.md", 1)])], store=store, sources={"a.md"}
        )
        assert result["ford"].sources == ("a.md", "b.md")
        assert result["ford"].mentions == 42

    def test_a_subject_below_the_threshold_corpus_wide_is_dropped(self):
        cfg.wiki_entity_min_mentions = 5
        result = self._run([_entity("ford", [("a.md", 0)])])
        assert result == {}

    def test_a_subject_its_other_sources_still_name_survives_losing_its_refs(self):
        """The cap can hand every ref to the source being re-indexed, so
        subtracting it empties the refs while b.md's five mentions remain.
        Dropping the entry would lose a subject the corpus still names."""
        cfg.wiki_entity_min_mentions = 1
        store = _FakeStore({})
        self._run([_entity("ford", [("a.md", 0)] + [("b.md", i) for i in range(5)])], store=store)
        result = self._run([], store=store, sources={"a.md"})
        assert result["ford"].sources == ("b.md",)

    def test_a_source_that_stopped_naming_an_entity_drops_it(self):
        """Re-ingesting a document that no longer mentions an entity must not
        leave the entity's page in the browse tree forever."""
        store = _FakeStore({})
        self._run([_entity("ford", [("a.md", 0)])], store=store)
        result = self._run([], store=store, sources={"a.md"})
        assert result == {}

    def test_an_entity_in_several_sources_survives_one_being_reindexed(self):
        store = _FakeStore({})
        self._run([_entity("ford", [("a.md", 0), ("b.md", 0)])], store=store)
        result = self._run([], store=store, sources={"a.md"})
        assert result["ford"].sources == ("b.md",)

    def test_reindexing_merges_rather_than_duplicating(self):
        cfg.wiki_entity_min_mentions = 1
        store = _FakeStore({})
        self._run([_entity("ford", [("b.md", 0)])], store=store)
        result = self._run([_entity("ford", [("a.md", 0)])], store=store, sources={"a.md"})
        assert result["ford"].sources == ("a.md", "b.md")

    def test_a_new_sources_evidence_survives_a_full_cap(self):
        """Head-truncating the concatenation gave a stub already holding cap
        refs nothing at all from a newly ingested source, however much that
        source had to say."""
        cfg.wiki_entity_min_mentions = 1
        cfg.wiki_stub_max_chunk_refs = 2
        save_stub_index(
            {
                "ford": WikiStub(
                    slug="ford",
                    label="ford",
                    kind=EntityKind.ENTITY,
                    type_hint="ORG",
                    source_mentions=(("old.md", 2),),
                    chunk_refs=(("old.md", 0), ("old.md", 1)),
                )
            }
        )
        result = self._run([_entity("ford", [("new.md", i) for i in range(9)])], sources={"new.md"})
        assert any(source == "new.md" for source, _ in result["ford"].chunk_refs)

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


class TestDropSourcesFromIndex:
    """Removed documents leave the index, without an extraction pass."""

    @pytest.fixture(autouse=True)
    def _low_threshold(self):
        cfg.wiki_entity_min_mentions = 1

    @staticmethod
    def _drop(seed_entities, removed):
        """Seed store and index from a real sync, then drop *removed* the way
        remove_documents does: its mention rows are already gone with its chunks
        before drop_sources_from_index re-aggregates the affected slugs."""
        store = _FakeStore({})
        _seed_full(store, seed_entities)
        for source in removed:
            store.replace_wiki_mentions_for_source(source, [])
        services = MagicMock()
        services.store = store
        with patch("lilbee.wiki.stubs.get_services", return_value=services):
            drop_sources_from_index(removed)
        return load_stub_index()

    def test_a_removed_document_stops_contributing(self):
        result = self._drop([_entity("ford", [("a.md", 0), ("b.md", 0)])], {"a.md"})
        assert result["ford"].sources == ("b.md",)

    def test_a_subject_only_that_document_named_disappears(self):
        """Its skip marker keeps it out of later syncs, so no refresh would
        ever revisit the entry and the tree would offer it forever."""
        result = self._drop([_entity("ford", [("a.md", 0)])], {"a.md"})
        assert result == {}

    def test_removing_nothing_leaves_the_index_alone(self):
        save_stub_index({"ford": _stub("ford")})
        drop_sources_from_index(set())
        assert list(load_stub_index()) == ["ford"]

    def test_a_subject_left_below_the_floor_is_dropped(self):
        """The floor is a corpus-wide judgement, so losing a document can put a
        subject under it just as re-indexing can."""
        cfg.wiki_entity_min_mentions = 3
        result = self._drop(
            [_entity("ford", [("a.md", 0), ("a.md", 1), ("b.md", 0), ("b.md", 1)])], {"a.md"}
        )
        assert result == {}

    def test_a_subject_another_document_still_names_survives(self):
        """Removing a.md still leaves b.md naming the subject four times, so the
        entry must stay with its surviving evidence."""
        result = self._drop(
            [_entity("ford", [("a.md", 0)] + [("b.md", i) for i in range(4)])], {"a.md"}
        )
        assert result["ford"].sources == ("b.md",)

    def test_an_absent_index_is_not_created(self):
        drop_sources_from_index({"a.md"})
        assert not stub_index_path().exists()


class TestUngeneratedStubs:
    def test_an_archived_page_still_reads_as_unwritten(self, isolated_env: Path):
        """Prune retires a page when the sources it cited go, while other
        documents can still name the subject. Nothing lists or restores
        archive/, so suppressing the stub would make the subject unreachable
        from either pane."""
        wiki_root = isolated_env / cfg.wiki_dir
        archived = wiki_root / WikiSubdir.ARCHIVE / WikiSubdir.ENTITIES / "ford.md"
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_text("# Ford\n", encoding="utf-8")
        assert [s.slug for s in ungenerated_stubs({"ford": _stub("ford")}, wiki_root)] == ["ford"]

    def test_an_unreadable_draft_does_not_hide_its_subject(self, isolated_env: Path):
        """A draft we cannot read still means a page was written for review, so
        the subject stays out of the unwritten list. Only a readable PENDING
        marker proves the page is a placeholder rather than content."""
        wiki_root = isolated_env / cfg.wiki_dir
        draft = wiki_root / WikiSubdir.DRAFTS / "ford.md"
        draft.parent.mkdir(parents=True, exist_ok=True)
        draft.write_text("# Ford\n", encoding="utf-8")
        with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
            listed = ungenerated_stubs({"ford": _stub("ford")}, wiki_root)
        assert listed == []

    def test_a_pending_marker_still_reads_as_unwritten(self, isolated_env: Path):
        """A marker records that generation failed to produce the section, so
        the subject has no page yet and must stay offered."""
        from lilbee.wiki.shared import PENDING_MARKER_KEYWORD_PARSE

        wiki_root = isolated_env / cfg.wiki_dir
        draft = wiki_root / WikiSubdir.DRAFTS / "ford.md"
        draft.parent.mkdir(parents=True, exist_ok=True)
        draft.write_text(f"<!-- {PENDING_MARKER_KEYWORD_PARSE} -->\n", encoding="utf-8")
        assert [s.slug for s in ungenerated_stubs({"ford": _stub("ford")}, wiki_root)] == ["ford"]

    def test_a_drafted_page_does_not_read_as_unwritten(self, isolated_env: Path):
        """The faithfulness gate routed it to drafts and it is awaiting review.
        Listing it as unwritten invites writing it again for another call while
        the draft sits there."""
        wiki_root = isolated_env / cfg.wiki_dir
        draft = wiki_root / WikiSubdir.DRAFTS / "ford.md"
        draft.parent.mkdir(parents=True, exist_ok=True)
        draft.write_text("# Ford\n", encoding="utf-8")
        assert ungenerated_stubs({"ford": _stub("ford")}, wiki_root) == []

    def test_only_stubs_without_a_page_are_returned(self, isolated_env: Path):
        wiki_root = isolated_env / cfg.wiki_dir
        published = wiki_root / WikiSubdir.ENTITIES / "ford.md"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text("# Ford\n", encoding="utf-8")
        stubs = {"ford": _stub("ford"), "gm": _stub("gm")}
        assert [s.slug for s in ungenerated_stubs(stubs, wiki_root)] == ["gm"]
