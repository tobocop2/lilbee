"""Adding the parent of a registered source takes the parent and folds the source in."""

from __future__ import annotations

import json
import shutil
import zlib
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

import lilbee.app.services as svc_mod
from lilbee.app.ingest import (
    forget_unfinished_on_cancel,
    register_sources,
    remove_documents_durably,
)
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.data.ingest import sync
from lilbee.data.store import REMOVED_REASON, HeldOut, Store
from lilbee.runtime.lock import (
    LibraryUnlockableError,
    LockingUnsupportedError,
)
from tests._held import held_hashes, held_kinds, held_reasons, hold_failed, hold_removed
from tests.conftest import (
    filesystem_without_locking,
    make_mock_services,
    sync_lock_not_a_lock_file,
)
from tests.test_store import _KEY_COLUMNS, _dump, _holders, _mention


class _Killed(BaseException):
    """Stands in for a process that dies: no handler in the product catches it."""


@pytest.fixture(autouse=True)
def library(tmp_path, monkeypatch):
    """A data root with a real store, a counting embedder, and config.toml read by each sync."""
    snapshot = cfg.model_copy()
    root = tmp_path / "library"
    cfg.data_root = root
    cfg.documents_dir = root / "documents"
    cfg.data_dir = root / "data"
    cfg.lancedb_dir = root / "data" / "lancedb"
    cfg.concept_graph = False
    cfg.linked_roots = {}
    monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
    store = Store(cfg)
    services = make_mock_services(store=store)
    svc_mod.set_services(services)
    yield services
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def notes(tmp_path) -> Path:
    """``notes`` with one loose file and the folder ``work`` holding two."""
    base = tmp_path / "src" / "notes"
    _write(base / "loose.md", "# Loose\n\nA loose page about the harbour schedule.\n")
    _write(base / "work" / "plan.md", "# Plan\n\nThe plan for the spring release.\n")
    _write(base / "work" / "budget.md", "# Budget\n\nThe budget for the spring release.\n")
    return base


def _keys(services) -> list[str]:
    return sorted(source["filename"] for source in services.store.get_sources())


def _embedded(services) -> list[str]:
    """Every text the embedder was given, in order."""
    return [text for call in services.embedder.embed_batch.call_args_list for text in call.args[0]]


async def _add(path: Path):
    result = register_sources([path])
    return result, await sync(quiet=True)


def _registry() -> dict[str, str]:
    return dict(settings.load(cfg.data_root).get("linked_roots") or {})


def _citation_key(key: str) -> str:
    """The footnote key the seed gives the citation of *key*: one per key on the page."""
    return f"src{zlib.crc32(key.encode())}"


def _seed(store: Store, key: str) -> None:
    """Write one row for *key* into every table that holds a source key, by the store's writers."""
    from lilbee.data.store import ConceptRecords, SourceType
    from lilbee.retrieval.concepts.graph import ConceptGraph
    from tests.test_store import _titled_records

    store.add_chunks(_titled_records(key, 1, title=None, dim=cfg.embedding_dim))
    store.add_page_texts([{"source": key, "page": 1, "text": "t", "content_type": "text"}])
    ConceptGraph(cfg, store).write_concept_records(
        ConceptRecords(
            nodes=[],
            edges=[],
            chunk_concepts=[{"chunk_source": key, "chunk_index": 0, "concept": "roadmap"}],
        )
    )
    store.add_entities(
        [
            {
                "entity": "Boeing",
                "type": "ORG",
                "normalized_value": "boeing",
                "source": key,
                "page": 1,
                "chunk_index": 0,
                "confidence": 1.0,
            }
        ]
    )
    store.replace_wiki_mentions_for_source(key, [_mention("boeing", key, 2, [0])])
    store.add_citations(
        [
            {
                "wiki_source": "wiki/entities/boeing.md",
                "wiki_chunk_index": 0,
                "citation_key": _citation_key(key),
                "claim_type": "fact",
                "source_filename": key,
                "source_hash": "h",
                "page_start": 0,
                "page_end": 0,
                "line_start": 1,
                "line_end": 2,
                "excerpt": "plain body words",
                "created_at": "",
            }
        ]
    )
    store.upsert_source(key, f"hash of {key}", 1, SourceType.DOCUMENT)


def _write_wiki_page(store: Store, keys: list[str], subdir: str = "entities") -> Path:
    """A page about Boeing written by the wiki's own writers from the chunks of *keys*."""
    from lilbee.wiki.citations import render_citation_block
    from lilbee.wiki.page import assemble_content, build_frontmatter, write_page

    chunks = [chunk for key in keys for chunk in store.get_chunks_by_source(key)]
    records = [rec for key in keys for rec in store.get_citations_for_source(key)]
    content = assemble_content(
        build_frontmatter(cfg, keys, 0.9, chunks=chunks),
        "# Boeing\n\nBoeing builds aircraft.[^src1]\n",
        render_citation_block(records),
    )
    wiki_root = cfg.data_root / cfg.wiki_dir
    source_ids = store.source_ids(keys)
    return write_page(
        wiki_root, subdir, "boeing", content, 1.0, keys, "entities", source_ids=source_ids
    )


def _save_stub_index(store: Store) -> None:
    from lilbee.wiki.stubs import _stubs_from_mention_rows, save_stub_index

    cfg.wiki_entity_min_mentions = 1
    stubs = _stubs_from_mention_rows(store.wiki_mention_rows(), cfg, cfg.wiki_stub_max_chunk_refs)
    save_stub_index(stubs, cfg)


def _hold_out(key: str, kind: HeldOut) -> None:
    """Record *key* as held out, by the store writer a sync or a remove uses."""
    if kind is HeldOut.REMOVED:
        hold_removed({key: f"hash of {key}"})
    else:
        hold_failed({key: f"hash of {key}"}, reason="no text")


def _shown_index(store: Store) -> str:
    """The subject index with the path of each source it names by id, or the id with none."""
    from lilbee.wiki.stubs import load_stub_index

    stubs = load_stub_index(cfg)
    paths = store.source_paths(i for stub in stubs.values() for i in stub.sources)
    return json.dumps(
        {
            slug: [
                sorted((paths.get(i, i), n) for i, n in stub.source_mentions),
                sorted((paths.get(i, i), n) for i, n in stub.chunk_refs),
            ]
            for slug, stub in sorted(stubs.items())
        }
    )


def _wiki_file(store: Store, path: Path) -> str:
    """A wiki file as a person reads it, less its generation time."""
    if path.name == "stubs.json":
        return _shown_index(store)
    text = path.read_text(encoding="utf-8")
    return "\n".join(line for line in text.split("\n") if not line.startswith("generated_at: "))


def _everything(store: Store) -> dict[str, object]:
    """What an absorb can change, less the time each source was indexed at."""
    wiki_root = cfg.data_root / cfg.wiki_dir
    tables = {
        name: sorted(row.split("'ingested_at'")[0] for row in rows)
        for name, rows in _dump(store).items()
        if name != "_meta"
    }
    return {
        "tables": tables,
        "held": {
            path: (record.kind, record.file_hash, record.reason)
            for path, record in sorted(store.held_records().items())
        },
        "wiki": {
            path.relative_to(wiki_root).as_posix(): _wiki_file(store, path)
            for path in sorted(wiki_root.rglob("*"))
            if path.is_file()
        },
        "registry": sorted(_registry()),
    }


def _seeded_child(store: Store, parent: Path, child: str = "work") -> Path:
    """Register ``parent/child`` and seed two of its files into every key holder."""
    child_path = parent / child
    assert register_sources([child_path]).registered == [child]
    for name in ("plan.md", "budget.md"):
        _seed(store, f"{child}/{name}")
    _seed(store, "workshop/x.md")
    _hold_out(f"{child}/gone.md", HeldOut.REMOVED)
    _hold_out(f"{child}/broken.md", HeldOut.FAILED)
    _write_wiki_page(store, [f"{child}/budget.md", f"{child}/plan.md"])
    _save_stub_index(store)
    return child_path


def _raw_rows(store: Store) -> dict[str, list[str]]:
    """Every row of every table as stored, sorted."""
    from lilbee.data.store.lance_helpers import table_names

    db = store.get_db()
    return {
        name: sorted(map(str, db.open_table(name).to_arrow().to_pylist()))
        for name in table_names(db)
    }


def _register_as(roots: dict[str, Path]) -> None:
    """Write *roots* as the registry, label to path, as a hand edit leaves it."""
    stored = {label: str(path.resolve()) for label, path in roots.items()}
    settings.set_value(cfg.data_root, "linked_roots", stored)
    cfg.linked_roots = dict(stored)


class TestTheParentTakesTheChild:
    async def test_adding_the_parent_registers_it_and_unregisters_the_child(self, library, notes):
        await _add(notes / "work")
        assert _keys(library) == ["work/budget.md", "work/plan.md"]

        result, synced = await _add(notes)

        assert result.registered == ["notes"]
        assert result.absorbed == ["work"]
        assert result.overlapping == result.containing == []
        assert _registry() == {"notes": str(notes.resolve())}
        assert cfg.linked_roots == {"notes": str(notes.resolve())}
        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md", "notes/work/plan.md"]
        assert synced.added == ["notes/loose.md"]
        assert synced.unchanged == 2
        assert synced.relocated == []

    async def test_absorbed_files_are_not_embedded_again(self, library, notes):
        await _add(notes / "work")
        stamps = library.store.source_ingested_at_map()
        embedded_before = len(_embedded(library))
        assert embedded_before > 0

        _result, synced = await _add(notes)

        assert synced.relocated == [] and synced.unchanged == 2
        new_texts = _embedded(library)[embedded_before:]
        assert new_texts and all("harbour" in text for text in new_texts)
        after = library.store.source_ingested_at_map()
        assert after["notes/work/plan.md"] == stamps["work/plan.md"]
        assert after["notes/work/budget.md"] == stamps["work/budget.md"]

    async def test_a_file_edited_before_the_absorb_is_indexed_once(self, library, notes):
        await _add(notes / "work")
        _write(notes / "work" / "budget.md", "# Budget\n\nThe budget doubled since.\n")

        _result, synced = await _add(notes)

        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md", "notes/work/plan.md"]
        assert synced.updated == ["notes/work/budget.md"]
        chunks = library.store.get_chunks_by_source("notes/work/budget.md")
        assert chunks and all("doubled" in chunk.chunk for chunk in chunks)
        assert library.store.get_chunks_by_source("work/budget.md") == []

    def test_the_add_changes_no_row_of_the_index(self, library, notes):
        store = library.store
        _seeded_child(store, notes)
        before = _raw_rows(store)
        assert _holders(store, (notes / "work" / "plan.md").resolve().as_posix()) == _KEY_COLUMNS

        register_sources([notes])

        assert _raw_rows(store) == before
        shown = {row["filename"] for row in store.get_sources()}
        assert {"notes/work/plan.md", "notes/work/budget.md", "workshop/x.md"} <= shown
        assert not shown & {"work/plan.md", "work/budget.md"}

    def test_the_wiki_page_and_the_subject_index_follow_the_absorb(self, library, notes):
        from lilbee.wiki.shared import parse_frontmatter
        from lilbee.wiki.stubs import load_stub_index, shown_sources

        store = library.store
        _seeded_child(store, notes)
        page = cfg.data_root / cfg.wiki_dir / "entities" / "boeing.md"
        before = page.read_text(encoding="utf-8")
        assert "work/plan.md, lines 1-2" in before
        listed = load_stub_index(cfg)["boeing"].sources
        assert shown_sources(load_stub_index(cfg)["boeing"], store) == [
            "work/budget.md",
            "work/plan.md",
            "workshop/x.md",
        ]

        register_sources([notes])

        after = page.read_text(encoding="utf-8")
        frontmatter = parse_frontmatter(after)
        assert frontmatter["sources"] == ["notes/work/budget.md", "notes/work/plan.md"]
        assert [chunk["source"] for chunk in frontmatter["provenance"]["chunks"]] == [
            "notes/work/budget.md",
            "notes/work/plan.md",
        ]
        footnotes = [line for line in after.splitlines() if line.startswith("[^src")]
        assert len(footnotes) == 2
        assert all(line.split(": ", 1)[1].startswith("notes/work/") for line in footnotes)
        assert after.replace("notes/work/", "work/") == before
        stub = load_stub_index(cfg)["boeing"]
        assert stub.sources == listed
        assert shown_sources(stub, store) == [
            "notes/work/budget.md",
            "notes/work/plan.md",
            "workshop/x.md",
        ]
        assert {source for source, _index in stub.chunk_refs} == set(stub.sources)

    def test_a_removed_file_stays_removed_and_a_failed_file_keeps_its_record(self, library, notes):
        _seeded_child(library.store, notes)
        _hold_out("workshop/kept.md", HeldOut.FAILED)

        register_sources([notes])

        assert held_hashes() == {
            "notes/work/gone.md": "hash of work/gone.md",
            "notes/work/broken.md": "hash of work/broken.md",
            "workshop/kept.md": "hash of workshop/kept.md",
        }
        assert held_kinds() == {
            "notes/work/gone.md": HeldOut.REMOVED,
            "notes/work/broken.md": HeldOut.FAILED,
            "workshop/kept.md": HeldOut.FAILED,
        }
        assert held_reasons() == {
            "notes/work/gone.md": REMOVED_REASON,
            "notes/work/broken.md": "no text",
            "workshop/kept.md": "no text",
        }

    def test_a_retry_after_the_absorb_releases_the_failure_and_keeps_the_removal(
        self, library, notes
    ):
        _seeded_child(library.store, notes)
        register_sources([notes])

        assert library.store.release_failed() == ["notes/work/broken.md"]
        assert held_kinds() == {"notes/work/gone.md": HeldOut.REMOVED}

    async def test_a_file_removed_from_the_child_does_not_return(self, library, notes):
        await _add(notes / "work")
        remove_documents_durably(["work/plan.md"])
        assert _keys(library) == ["work/budget.md"]

        _result, synced = await _add(notes)

        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md"]
        assert synced.added == ["notes/loose.md"]

    async def test_naming_the_parent_outranks_a_removal_on_its_own_file(self, library, notes):
        """The rule every add applies to the path it names, kept for the parent's own files."""
        await _add(notes / "work")
        _hold_out("notes/loose.md", HeldOut.REMOVED)
        _hold_out("work/gone.md", HeldOut.REMOVED)

        register_sources([notes])

        assert sorted(held_hashes()) == ["notes/work/gone.md"]

    async def test_several_children_are_absorbed_in_one_operation(self, library, notes):
        _write(notes / "home" / "list.md", "# List\n\nThe list for the market.\n")
        register_sources([notes / "work", notes / "home"])
        await sync(quiet=True)
        written = []
        real = settings.mutate_value

        def _record(*args, **kwargs):
            written.append(args[1])
            return real(*args, **kwargs)

        with mock.patch.object(settings, "mutate_value", _record):
            result = register_sources([notes])

        assert result.absorbed_into == {"notes": ["home", "work"]}
        assert _registry() == {"notes": str(notes.resolve())}
        assert _keys(library) == [
            "notes/home/list.md",
            "notes/work/budget.md",
            "notes/work/plan.md",
        ]
        assert written == ["linked_roots"]

    async def test_a_grandparent_carries_every_folder_between(self, library, notes):
        await _add(notes / "work")

        result = register_sources([notes.parent])

        assert result.absorbed_into == {"src": ["work"]}
        assert _keys(library) == ["src/notes/work/budget.md", "src/notes/work/plan.md"]

    async def test_a_single_file_child_takes_its_path_below_the_parent(self, library, notes):
        await _add(notes / "work" / "plan.md")
        assert _keys(library) == ["plan.md"]

        result, synced = await _add(notes)

        assert result.absorbed == ["plan.md"]
        assert "notes/work/plan.md" in _keys(library) and "plan.md" not in _keys(library)
        assert synced.unchanged == 1 and synced.added == ["notes/loose.md", "notes/work/budget.md"]

    async def test_a_child_whose_folder_is_gone_is_absorbed_by_its_stored_path(
        self, library, notes
    ):
        await _add(notes / "work")
        shutil.rmtree(notes / "work")

        result = register_sources([notes])

        assert result.absorbed == ["work"]
        assert _registry() == {"notes": str(notes.resolve())}
        assert _keys(library) == ["notes/work/budget.md", "notes/work/plan.md"]

    async def test_a_child_registered_through_a_link_is_absorbed(self, library, notes, tmp_path):
        link = tmp_path / "link-to-work"
        link.symlink_to(notes / "work", target_is_directory=True)
        await _add(link)
        assert _keys(library) == ["work/budget.md", "work/plan.md"]

        result = register_sources([notes])

        assert result.absorbed == ["work"]

    async def test_a_parent_named_through_a_link_absorbs(self, library, notes, tmp_path):
        await _add(notes / "work")
        link = tmp_path / "link-to-notes"
        link.symlink_to(notes, target_is_directory=True)

        result = register_sources([link])

        assert result.absorbed_into == {"notes": ["work"]}

    async def test_a_link_inside_the_parent_that_points_outside_is_not_absorbed(
        self, library, notes, tmp_path
    ):
        outside = tmp_path / "outside"
        _write(outside / "far.md", "# Far\n\nA page that lives elsewhere.\n")
        (notes / "far").symlink_to(outside, target_is_directory=True)
        await _add(outside)

        result, _synced = await _add(notes)

        assert result.registered == ["notes"] and result.absorbed == []
        assert sorted(_registry()) == ["notes", "outside"]
        assert "outside/far.md" in _keys(library)
        assert not [key for key in _keys(library) if key.startswith("notes/far")]

    async def test_a_child_with_a_label_that_is_not_its_folder_name_is_absorbed(
        self, library, notes
    ):
        _register_as({"renamed": notes / "work"})
        await sync(quiet=True)
        assert _keys(library) == ["renamed/budget.md", "renamed/plan.md"]
        library.embedder.embed_batch.reset_mock()

        result, synced = await _add(notes)

        assert result.registered == ["notes"] and result.absorbed_into == {"notes": ["renamed"]}
        assert result.containing == []
        assert _registry() == {"notes": str(notes.resolve())}
        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md", "notes/work/plan.md"]
        assert synced.added == ["notes/loose.md"] and synced.relocated == []
        assert all("harbour" in text for text in _embedded(library)) and _embedded(library)

    async def test_a_single_file_child_under_another_label_takes_its_file_name(
        self, library, notes
    ):
        _register_as({"renamed": notes / "work" / "plan.md"})
        await sync(quiet=True)
        assert _keys(library) == ["renamed"]

        result, synced = await _add(notes)

        assert result.absorbed_into == {"notes": ["renamed"]}
        assert "notes/work/plan.md" in _keys(library) and "renamed" not in _keys(library)
        assert "notes/work/plan.md" not in synced.added

    async def test_children_that_lie_inside_each_other_are_both_absorbed(self, library, notes):
        _write(notes / "work" / "deep" / "a.md", "# A\n\nA deep page.\n")
        _register_as({"work": notes / "work", "deep": notes / "work" / "deep"})
        await sync(quiet=True)
        # The walk of the outer source leaves the inner one to its own entry.
        assert _keys(library) == ["deep/a.md", "work/budget.md", "work/plan.md"]
        library.embedder.embed_batch.reset_mock()

        result, synced = await _add(notes)

        assert result.absorbed_into == {"notes": ["deep", "work"]} and result.containing == []
        assert _registry() == {"notes": str(notes.resolve())}
        assert _keys(library) == [
            "notes/loose.md",
            "notes/work/budget.md",
            "notes/work/deep/a.md",
            "notes/work/plan.md",
        ]
        assert synced.added == ["notes/loose.md"] and synced.relocated == []
        assert all("harbour" in text for text in _embedded(library)) and _embedded(library)


async def _moved_folder_with_a_source_inside(tmp_path: Path) -> Path:
    """Leave ``work`` holding a row for ``deep/a.md`` that the source ``deep`` also indexes.

    ``work`` is indexed, then its folder moves and ``deep/a.md`` changes. ``deep``
    is added at the new place, and ``work`` again while a pattern hides ``deep``.
    Returns the new place of ``work``.
    """
    before = tmp_path / "before" / "work"
    _write(before / "plan.md", "# Plan\n\nThe plan for the spring release.\n")
    _write(before / "deep" / "a.md", "# A\n\nThe first text of the deep page.\n")
    await _add(before)
    after = tmp_path / "after" / "work"
    after.parent.mkdir()
    before.rename(after)
    _write(after / "deep" / "a.md", "# A\n\nThe second text of the deep page.\n")
    await _add(after / "deep")
    ignore = _write(after / ".lilbeeignore", "deep/\n")
    await _add(after)
    ignore.unlink()
    await sync(quiet=True)
    return after


class TestAMovedFolderWithASourceInside:
    async def test_the_adds_leave_the_file_of_the_inner_source_at_one_key(self, library, tmp_path):
        """The outer row of the moved file lands on the inner source's row, which stays."""
        after = await _moved_folder_with_a_source_inside(tmp_path)

        assert sorted(_registry()) == ["deep", "work"]
        assert _registry()["work"] == str(after.resolve())
        assert _keys(library) == ["deep/a.md", "work/plan.md"]
        assert "second text" in library.store.get_chunks_by_source("deep/a.md")[0].chunk

    async def test_the_entry_of_the_inner_source_lands(self, library, tmp_path):
        after = await _moved_folder_with_a_source_inside(tmp_path)
        store = library.store
        inner = {row["filename"]: row["file_hash"] for row in store.get_sources()}["deep/a.md"]

        result, synced = await _add(after.parent)

        assert result.absorbed_into == {"after": ["deep", "work"]}
        assert _keys(library) == ["after/work/deep/a.md", "after/work/plan.md"]
        landed = {row["filename"]: row["file_hash"] for row in store.get_sources()}
        assert landed["after/work/deep/a.md"] == inner
        assert len(store.get_chunks_by_source("after/work/deep/a.md")) == 1
        assert synced.added == [] and synced.updated == []


class TestAMovedFolderAddedAgain:
    async def test_its_documents_keep_their_names_and_nothing_is_read_again(
        self, library, tmp_path
    ):
        before = tmp_path / "before" / "work"
        _write(before / "plan.md", "# Plan\n\nThe plan for the spring release.\n")
        await _add(before)
        after = tmp_path / "after" / "work"
        after.parent.mkdir()
        before.rename(after)
        library.embedder.embed_batch.reset_mock()

        result, synced = await _add(after)

        assert result.registered == ["work"]
        assert _registry() == {"work": str(after.resolve())}
        assert _keys(library) == ["work/plan.md"]
        assert (synced.added, synced.updated, synced.relocated) == ([], [], [])
        assert synced.unchanged == 1
        assert _embedded(library) == []


class TestASyncWhileTheRegistryChanges:
    async def test_the_sync_names_sources_by_the_registry_it_started_with(self, library, notes):
        await _add(notes / "work")
        from lilbee.data.ingest import pipeline

        real = pipeline.discover_corpus
        seen: list[list[str]] = []

        def _registry_changes_meanwhile(*args, **kwargs):
            scan = real(*args, **kwargs)
            cfg.linked_roots = {"notes": str(notes.resolve())}
            seen.append(_keys(library))
            return scan

        with mock.patch.object(pipeline, "discover_corpus", _registry_changes_meanwhile):
            await sync(quiet=True)

        assert seen == [["work/budget.md", "work/plan.md"]]
        assert _keys(library) == ["notes/work/budget.md", "notes/work/plan.md"]


class TestWhatIsNotAbsorbed:
    @pytest.mark.parametrize(
        ("child", "ignore"),
        [
            (".hidden/work", None),
            ("node_modules/work", None),
            ("skipped/work", "skipped/\n"),
        ],
        ids=["dot-directory", "ignored-name", "lilbeeignore"],
    )
    async def test_a_child_below_a_pruned_directory_stays_its_own_source(
        self, library, tmp_path, child, ignore
    ):
        base = tmp_path / "src" / "notes"
        _write(base / "loose.md", "# Loose\n\nA loose page about the harbour schedule.\n")
        _write(base / child / "plan.md", "# Plan\n\nThe plan for the spring release.\n")
        if ignore is not None:
            _write(base / ".lilbeeignore", ignore)
        cfg.ignore_dirs = frozenset({"node_modules"})
        await _add(base / child)

        result, synced = await _add(base)

        assert result.registered == ["notes"] and result.absorbed == []
        assert sorted(_registry()) == ["notes", "work"]
        assert _keys(library) == ["notes/loose.md", "work/plan.md"]
        assert synced.added == ["notes/loose.md"]

    async def test_an_absorbing_add_keeps_the_records_of_a_source_it_does_not_take_in(
        self, library, notes
    ):
        _write(notes / ".hidden" / "other" / "x.md", "# X\n\nA page about the tide tables.\n")
        await _add(notes / ".hidden" / "other")
        await _add(notes / "work")
        _hold_out("other/gone.md", HeldOut.REMOVED)
        _hold_out("work/gone.md", HeldOut.REMOVED)

        result = register_sources([notes])

        assert result.absorbed == ["work"]
        assert sorted(held_hashes()) == ["notes/work/gone.md", "other/gone.md"]

    async def test_a_child_below_an_ignored_folder_is_indexed_once_when_the_pattern_goes(
        self, library, tmp_path
    ):
        base = tmp_path / "src" / "notes"
        _write(base / "loose.md", "# Loose\n\nA loose page about the harbour schedule.\n")
        _write(base / "skipped" / "other.md", "# Other\n\nA page about the tide tables.\n")
        _write(base / "skipped" / "work" / "plan.md", "# Plan\n\nThe spring release plan.\n")
        ignore = _write(base / ".lilbeeignore", "skipped/\n")
        await _add(base / "skipped" / "work")
        await _add(base)
        ignore.unlink()

        synced = await sync(quiet=True)

        # The parent's walk enters the folder and leaves the source inside it to its own entry.
        assert synced.added == ["notes/skipped/other.md"]
        assert _keys(library) == ["notes/loose.md", "notes/skipped/other.md", "work/plan.md"]
        assert sorted(_registry()) == ["notes", "work"]

    async def test_a_single_file_child_is_indexed_once_when_the_pattern_goes(self, library, notes):
        ignore = _write(notes / ".lilbeeignore", "plan.md\n")
        await _add(notes / "work" / "plan.md")
        await _add(notes)
        ignore.unlink()

        synced = await sync(quiet=True)

        assert synced.added == []
        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md", "plan.md"]

    async def test_a_single_file_child_that_is_a_dot_file_stays_its_own_source(
        self, library, notes
    ):
        hidden = _write(notes / ".draft.md", "# Draft\n\nA hidden draft.\n")
        register_sources([hidden])

        result = register_sources([notes])

        assert result.registered == ["notes"] and result.absorbed == []
        assert sorted(_registry()) == [".draft.md", "notes"]

    async def test_a_single_file_child_an_ignore_pattern_excludes_stays(self, library, notes):
        _write(notes / ".lilbeeignore", "work/plan.md\n")
        register_sources([notes / "work" / "plan.md"])

        result = register_sources([notes])

        assert result.absorbed == []
        assert sorted(_registry()) == ["notes", "plan.md"]

    async def test_a_parent_inside_another_source_is_refused_as_before(self, library, notes):
        vault = notes.parent
        await _add(notes / "work")
        settings.set_value(
            cfg.data_root,
            "linked_roots",
            {"work": str((notes / "work").resolve()), "src": str(vault.resolve())},
        )
        before = _keys(library)

        result = register_sources([notes])

        assert result.overlapping == ["notes"] and result.registered == []
        assert result.containing == [] and result.absorbed == []
        assert sorted(_registry()) == ["src", "work"]
        assert _keys(library) == before

    async def test_a_parent_of_the_documents_directory_is_refused_as_before(self, library):
        cfg.documents_dir.mkdir(parents=True)

        result = register_sources([cfg.data_root])

        assert result.overlapping == result.containing == ["library"]
        assert result.registered == [] and result.absorbed == []

    async def test_a_taken_parent_label_absorbs_nothing(self, library, notes, tmp_path):
        other = tmp_path / "elsewhere" / "notes"
        _write(other / "other.md", "# Other\n\nAnother folder with the same name.\n")
        await _add(other)
        await _add(notes / "work")
        before = _everything(library.store)

        result = register_sources([notes])

        assert result.name_taken == ["notes"]
        assert result.registered == [] and result.absorbed == []
        assert _everything(library.store) == before

    async def test_force_takes_the_label_and_absorbs(self, library, notes, tmp_path):
        other = tmp_path / "elsewhere" / "notes"
        _write(other / "other.md", "# Other\n\nAnother folder with the same name.\n")
        register_sources([other])
        await _add(notes / "work")

        result = register_sources([notes], force=True)

        assert result.absorbed_into == {"notes": ["work"]}
        assert _registry() == {"notes": str(notes.resolve())}

    async def test_a_child_named_beside_its_parent_was_never_a_source(self, library, notes):
        result = register_sources([notes / "work", notes])

        assert result.registered == ["notes"]
        assert result.absorbed_into == {}
        assert _registry() == {"notes": str(notes.resolve())}

    async def test_a_registered_child_named_beside_its_parent_is_absorbed(self, library, notes):
        await _add(notes / "work")

        result = register_sources([notes / "work", notes])

        assert result.registered == ["notes"]
        assert result.absorbed_into == {"notes": ["work"]}
        assert _registry() == {"notes": str(notes.resolve())}
        assert _keys(library) == ["notes/work/budget.md", "notes/work/plan.md"]


class TestACancelledAdd:
    async def test_a_cancelled_add_keeps_a_parent_that_absorbed(self, library, notes):
        import asyncio
        import threading

        await _add(notes / "work")
        result = register_sources([notes])
        cancel = threading.Event()

        with (
            pytest.raises(asyncio.CancelledError),
            forget_unfinished_on_cancel([notes], result.revocable, cancel, lambda: True),
        ):
            cancel.set()
            raise asyncio.CancelledError

        assert result.registered == ["notes"] and result.revocable == []
        assert _registry() == {"notes": str(notes.resolve())}
        assert _keys(library) == ["notes/work/budget.md", "notes/work/plan.md"]

    async def test_the_rollback_would_unregister_the_parent_if_it_were_handed_over(
        self, library, notes
    ):
        """The control for the test above: the same cancel with the parent's label."""
        import asyncio
        import threading

        await _add(notes / "work")
        result = register_sources([notes])
        cancel = threading.Event()

        with (
            pytest.raises(asyncio.CancelledError),
            forget_unfinished_on_cancel([notes], result.registered, cancel, lambda: True),
        ):
            cancel.set()
            raise asyncio.CancelledError

        assert _registry() == {}

    def test_a_parent_that_absorbed_nothing_is_revocable(self, library, notes):
        assert register_sources([notes]).revocable == ["notes"]


class TestEveryEntryPoint:
    """Each surface an add comes through names the source the parent took in."""

    async def test_the_cli_line_says_what_the_parent_now_includes(self, library, notes, capsys):
        from lilbee.cli.helpers import add_paths
        from lilbee.runtime.console import PlainConsole

        await _add(notes / "work")
        rollback_labels: list[list[str]] = []

        add_paths(
            [notes],
            PlainConsole(),
            run_sync=lambda registration: rollback_labels.append(registration.revocable),
        )

        assert "Registered 1 source(s); notes now includes work" in capsys.readouterr().out
        assert rollback_labels == [[]]

    async def test_the_cli_command_hands_the_rollback_no_absorbing_parent(self, library, notes):
        import threading

        from lilbee.app.ingest import AddRollback
        from lilbee.cli.commands import ingest_sync

        await _add(notes / "work")
        rollback = AddRollback(paths=[notes], at_sync=False)
        cfg.json_mode = True

        def _run_sync(_cancel, before_sync):
            before_sync()
            return MagicMock()

        with (
            mock.patch.object(ingest_sync, "_run_sync", _run_sync),
            mock.patch.object(ingest_sync, "sync_result_to_json", return_value={}),
        ):
            payload = ingest_sync._register_and_sync(
                [notes],
                [],
                sync_urls=False,
                force=False,
                cancel_event=threading.Event(),
                rollback=rollback,
            )

        assert payload is not None
        assert payload["copied"] == ["notes"] and payload["absorbed"] == ["work"]
        assert payload["overlapping"] == []
        assert json.loads(json.dumps(payload)) == payload
        assert rollback._roots == []

    async def test_the_tui_names_the_source_and_its_rollback_spares_the_parent(
        self, library, notes
    ):
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
        from lilbee.data.ingest import SyncResult

        await _add(notes / "work")
        screen = ChatScreen.__new__(ChatScreen)
        notify = MagicMock()
        reporter = MagicMock(spec=ProgressReporter)
        reporter.is_set.return_value = False
        with (
            mock.patch("lilbee.cli.tui.screens.chat.call_from_thread", notify),
            mock.patch("lilbee.runtime.asyncio_loop.run", return_value=SyncResult()),
            mock.patch("lilbee.cli.tui.screens.chat.forget_unfinished_on_cancel") as rollback,
        ):
            screen._do_add([notes], reporter)

        toasts = [call.args[2] for call in notify.call_args_list]
        assert "notes now includes the source work." in toasts
        assert msg.CMD_ADD_ABSORBED.format(parent="notes", children="work") in toasts
        assert msg.CMD_ADD_CONTAINING.format(names="notes") not in toasts
        assert rollback.call_args.args[1] == []

    async def test_a_failed_tui_sync_keeps_a_parent_that_absorbed(self, library, notes):
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter

        await _add(notes / "work")
        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock(spec=ProgressReporter)
        reporter.is_set.return_value = False
        with (
            mock.patch("lilbee.cli.tui.screens.chat.call_from_thread"),
            mock.patch("lilbee.runtime.asyncio_loop.run", side_effect=RuntimeError("disk")),
            pytest.raises(RuntimeError, match="disk"),
        ):
            screen._do_add([notes], reporter)

        assert _registry() == {"notes": str(notes.resolve())}

    async def test_the_http_summary_lists_the_absorbed_source(self, library, notes):
        from lilbee.server.handlers.ingest import _run_add
        from lilbee.server.handlers.sse import SseStream

        await _add(notes / "work")

        summary = await _run_add([str(notes)], False, None, None, SseStream())

        assert summary.copied == ["notes"] and summary.absorbed == ["work"]
        assert summary.overlapping == []
        assert summary.model_dump()["absorbed"] == ["work"]
        assert summary.sync is not None and summary.sync.added == ["notes/loose.md"]

    async def test_a_cancelled_http_add_still_lists_the_absorbed_source(self, library, notes):
        from lilbee.server.handlers.ingest import _run_add
        from lilbee.server.handlers.sse import SseStream

        await _add(notes / "work")
        sse = SseStream()
        sse.cancel.set()

        summary = await _run_add([str(notes)], False, None, None, sse)

        assert summary.absorbed == ["work"] and summary.sync is None

    async def test_the_mcp_result_lists_the_absorbed_source(self, library, notes):
        from lilbee.mcp_server import add

        await _add(notes / "work")

        result = await add([str(notes)])

        assert result["copied"] == ["notes"] and result["absorbed"] == ["work"]
        assert result["overlapping"] == [] and result["errors"] == []
        assert result["sync"]["added"] == ["notes/loose.md"]


_SEEDED_KEYS = ["work/budget.md", "work/plan.md", "workshop/x.md"]


# Each way a data root cannot be locked: what makes it so, what an add raises, and what it says.
_CANNOT_BE_LOCKED = {
    "unlockable": (
        sync_lock_not_a_lock_file,
        LibraryUnlockableError,
        "Stop every lilbee process, delete",
    ),
    "no-locking": (
        lambda _root: filesystem_without_locking(),
        LockingUnsupportedError,
        "is on a filesystem that does not support file locking",
    ),
}


def _data_root_that_cannot_be_locked(cause: str):
    return _CANNOT_BE_LOCKED[cause][0](cfg.data_root)


@pytest.mark.parametrize("cause", list(_CANNOT_BE_LOCKED))
class TestAnAddThatTakesASourceInOnADataRootItCannotLock:
    """Each surface refuses in its own shape, with a text about the add, and moves nothing."""

    @staticmethod
    def _assert_about_the_add(text: str, cause: str) -> None:
        assert _CANNOT_BE_LOCKED[cause][2] in text
        assert "reset" not in text.lower() and "Traceback" not in text

    @staticmethod
    def _assert_nothing_moved(library) -> None:
        assert _keys(library) == _SEEDED_KEYS
        assert sorted(_registry()) == ["work"]

    def test_the_library_call_raises_the_refusal(self, library, notes, cause):
        from lilbee import Lilbee

        bee = Lilbee(config=cfg, provider=MagicMock())
        bee._services = library
        _seeded_child(library.store, notes)
        with (
            _data_root_that_cannot_be_locked(cause),
            pytest.raises(_CANNOT_BE_LOCKED[cause][1]) as refused,
        ):
            bee.add([notes])

        self._assert_about_the_add(str(refused.value), cause)
        self._assert_nothing_moved(library)

    @pytest.mark.parametrize("json_mode", [False, True])
    def test_the_cli_prints_one_error(self, library, notes, cause, json_mode):
        from typer.testing import CliRunner

        from lilbee.cli.app import app

        _seeded_child(library.store, notes)
        command = ["add", "--data-dir", str(cfg.data_root), str(notes)]
        with _data_root_that_cannot_be_locked(cause):
            outcome = CliRunner().invoke(app, ["--json", *command] if json_mode else command)

        assert outcome.exit_code == 1 and isinstance(outcome.exception, SystemExit)
        if json_mode:
            payload = json.loads(outcome.output)
            assert list(payload) == ["error"]
            printed = payload["error"]
        else:
            printed = " ".join(outcome.output.split())
            assert printed.startswith("Error: ") and printed.count("Error: ") == 1
        self._assert_about_the_add(printed, cause)
        self._assert_nothing_moved(library)

    async def test_the_http_stream_ends_in_an_error(self, library, notes, cause):
        from lilbee.server.handlers import add_files_stream

        _seeded_child(library.store, notes)
        with _data_root_that_cannot_be_locked(cause):
            text = "".join([frame async for frame in add_files_stream([str(notes)])])

        assert "event: error" in text and "event: done" not in text
        self._assert_about_the_add(text, cause)
        self._assert_nothing_moved(library)

    async def test_the_mcp_tool_returns_an_error(self, library, notes, cause):
        from lilbee.mcp_server import add

        _seeded_child(library.store, notes)
        with _data_root_that_cannot_be_locked(cause):
            result = await add([str(notes)])

        assert list(result) == ["error"]
        self._assert_about_the_add(result["error"], cause)
        self._assert_nothing_moved(library)

    def test_the_tui_task_fails_with_the_refusal(self, library, notes, cause):
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter

        _seeded_child(library.store, notes)
        screen = ChatScreen.__new__(ChatScreen)
        reporter = MagicMock(spec=ProgressReporter)
        reporter.is_set.return_value = False
        with (
            mock.patch("lilbee.cli.tui.screens.chat.call_from_thread"),
            mock.patch("lilbee.runtime.asyncio_loop.run") as run,
            _data_root_that_cannot_be_locked(cause),
            pytest.raises(_CANNOT_BE_LOCKED[cause][1]) as refused,
        ):
            screen._do_add([notes], reporter)

        run.assert_not_called()
        self._assert_about_the_add(str(refused.value), cause)
        self._assert_nothing_moved(library)


class TestAnMcpToolOnAFilesystemWithoutLocking:
    """A sync or a plain add returns the tool's error in place of raising."""

    async def test_the_sync_tool_returns_an_error(self, library):
        from lilbee.mcp_server import sync as mcp_sync

        with filesystem_without_locking():
            result = await mcp_sync()

        assert list(result) == ["error"]
        assert "does not support file locking" in result["error"]

    async def test_an_add_returns_an_error_and_registers_nothing(self, library, tmp_path):
        from lilbee.mcp_server import add

        other = tmp_path / "src" / "other"
        _write(other / "page.md", "# Page\n\nA page about the ferry.\n")
        with filesystem_without_locking():
            result = await add([str(other)])

        assert list(result) == ["error"]
        assert "does not support file locking" in result["error"]
        assert _keys(library) == []
        assert _registry() == {}


async def _one_child(parent: Path) -> None:
    await _add(parent / "work")


async def _renamed_folder(parent: Path) -> None:
    _register_as({"renamed": parent / "work"})
    await sync(quiet=True)


async def _nested(parent: Path) -> None:
    _write(parent / "work" / "deep" / "a.md", "# A\n\nA deep page.\n")
    _register_as({"work": parent / "work", "deep": parent / "work" / "deep"})
    await sync(quiet=True)


async def _nested_and_renamed(parent: Path) -> None:
    _write(parent / "work" / "deep" / "a.md", "# A\n\nA deep page.\n")
    _register_as({"outer": parent / "work", "inner": parent / "work" / "deep"})
    await sync(quiet=True)


class TestTheOracle:
    @pytest.mark.parametrize(
        "registered_before",
        [_one_child, _renamed_folder, _nested, _nested_and_renamed],
    )
    async def test_an_absorb_and_a_sync_equal_an_index_built_from_the_parent_alone(
        self, library, notes, tmp_path, registered_before
    ):
        await registered_before(notes)
        keys = _keys(library)
        # No file has two keys, whatever the sources that hold it.
        assert len(keys) == len({key.rsplit("/", 1)[-1] for key in keys}) > 0
        await _add(notes)
        absorbed = _everything(library.store)

        saved = (cfg.data_root, cfg.documents_dir, cfg.data_dir, cfg.lancedb_dir)
        root = tmp_path / "fresh-library"
        cfg.data_root, cfg.documents_dir = root, root / "documents"
        cfg.data_dir, cfg.lancedb_dir = root / "data", root / "data" / "lancedb"
        cfg.linked_roots = {}
        fresh_store = Store(cfg)
        svc_mod.set_services(make_mock_services(store=fresh_store))
        await _add(notes)
        fresh = _everything(fresh_store)
        cfg.data_root, cfg.documents_dir, cfg.data_dir, cfg.lancedb_dir = saved

        assert absorbed == fresh
        assert sum(len(rows) for rows in fresh["tables"].values()) > 3


def _pending_draft(store: Store, key: str, excerpt: str) -> Path:
    """A draft about the spring release that cites *key*, written by the wiki's own writer."""
    from lilbee.wiki.page import assemble_content, build_frontmatter
    from lilbee.wiki.persistence import persist_and_finalize
    from lilbee.wiki.shared import PageTarget, WikiSubdir

    chunk = store.get_chunks_by_source(key)[0]
    assert excerpt in chunk.chunk
    content = assemble_content(
        build_frontmatter(cfg, [key], 0.2, chunks=[chunk]),
        "# Spring release\n\nThe release has a plan.[^src1]\n",
        f'[^src1]: {key}, excerpt: "{excerpt}"\n',
    )
    target = PageTarget(
        wiki_root=cfg.data_root / cfg.wiki_dir,
        subdir=WikiSubdir.DRAFTS,
        slug="spring-release",
        wiki_source=f"{cfg.wiki_dir}/drafts/spring-release.md",
        page_type=WikiSubdir.CONCEPTS,
        label="Spring release",
    )
    return persist_and_finalize(content, target, [], [key], store, cfg)


class TestAPendingDraft:
    _EXCERPT = "The plan for the spring release."

    async def test_a_pending_draft_is_accepted_after_an_absorb(self, library, notes):
        from lilbee.wiki.drafts import accept_draft
        from lilbee.wiki.shared import parse_frontmatter

        await _add(notes / "work")
        store = library.store
        draft = _pending_draft(store, "work/plan.md", self._EXCERPT)
        written = draft.read_text(encoding="utf-8")

        register_sources([notes])

        assert draft.read_text(encoding="utf-8") == written, "a move rewrote the draft"
        with mock.patch("lilbee.wiki.drafts.rewrite_links_across_wiki"):
            accepted = accept_draft("spring-release", cfg.data_root / cfg.wiki_dir, store, cfg)

        published = accepted.moved_to.read_text(encoding="utf-8")
        assert f'[^src1]: notes/work/plan.md, excerpt: "{self._EXCERPT}"' in published
        assert parse_frontmatter(published)["sources"] == ["notes/work/plan.md"]
        assert "source_ids" not in published
        assert "work/plan.md" not in published.replace("notes/work/plan.md", "")
        rows = store.get_citations_for_source("notes/work/plan.md")
        assert [row["wiki_source"] for row in rows] == ["wiki/concepts/spring-release.md"]
        assert store.get_citations_for_source("work/plan.md") == []

    async def test_a_draft_beside_its_published_page_is_accepted_after_an_absorb(
        self, library, notes
    ):
        import os
        import time

        from lilbee.data.store import CitationRecord
        from lilbee.wiki.citations import render_citation_block
        from lilbee.wiki.drafts import accept_draft
        from lilbee.wiki.page import assemble_content, build_frontmatter

        await _add(notes / "work")
        store = library.store
        record = CitationRecord(
            wiki_source=f"{cfg.wiki_dir}/concepts/spring-release.md",
            wiki_chunk_index=0,
            citation_key="src1",
            claim_type="fact",
            source_filename="work/plan.md",
            source_hash="h",
            page_start=0,
            page_end=0,
            line_start=0,
            line_end=0,
            excerpt=self._EXCERPT,
            created_at="",
        )
        store.add_citations([record])
        published = cfg.data_root / record["wiki_source"]
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_text(
            assemble_content(
                build_frontmatter(cfg, ["work/plan.md"], 0.9),
                "The published body.[^src1]\n",
                render_citation_block([record]),
            ),
            encoding="utf-8",
        )
        an_hour_ago = time.time() - 3600
        os.utime(published, (an_hour_ago, an_hour_ago))
        _pending_draft(store, "work/plan.md", self._EXCERPT)

        register_sources([notes])

        assert "notes/work/plan.md" in published.read_text(encoding="utf-8"), "no re-render"
        assert published.stat().st_mtime == pytest.approx(an_hour_ago)
        with mock.patch("lilbee.wiki.drafts.rewrite_links_across_wiki"):
            accepted = accept_draft("spring-release", cfg.data_root / cfg.wiki_dir, store, cfg)
        shown = accepted.moved_to.read_text(encoding="utf-8")
        assert "The release has a plan." in shown
        assert f'[^src1]: notes/work/plan.md, excerpt: "{self._EXCERPT}"' in shown

    async def test_the_diff_of_a_pending_draft_shows_the_new_path(self, library, notes):
        from lilbee.wiki.drafts import diff_draft

        await _add(notes / "work")
        _pending_draft(library.store, "work/plan.md", self._EXCERPT)

        register_sources([notes])

        diff = diff_draft("spring-release", cfg.data_root / cfg.wiki_dir, library.store)
        assert f'+[^src1]: notes/work/plan.md, excerpt: "{self._EXCERPT}"' in diff
        assert '+sources: ["notes/work/plan.md"]' in diff


class TestKeysDiscoveryWrites:
    @pytest.mark.skipif(__import__("sys").platform == "win32", reason="not file names on Windows")
    def test_a_backslash_a_newline_and_a_tab_are_in_keys_discovery_writes(self, library, tmp_path):
        from lilbee.data.ingest.discovery import discover_files

        root = tmp_path / "odd"
        for name in ("plain.md", "back\\slash.md", "new\nline.md", "tab\there.md"):
            _write(root / name, "# Odd\n\nA page.\n")
        cfg.linked_roots = {"odd": str(root)}

        assert sorted(discover_files()) == [
            "odd/back\\slash.md",
            "odd/new\nline.md",
            "odd/plain.md",
            "odd/tab\there.md",
        ]


class TestWalkReaches:
    def test_it_agrees_with_the_walk_on_every_path_of_a_tree(self, library, tmp_path):
        from lilbee.data.ingest.discovery import discover_files, walk_reaches
        from lilbee.data.ingest.ignore import IgnoreRules

        base = tmp_path / "tree"
        files = [
            "a.md",
            ".dot.md",
            "sub/b.md",
            "sub/.git/c.md",
            ".venv/d.md",
            "node_modules/e.md",
            "pkg.egg-info/f.md",
            "skipped/g.md",
            "kept/skip-me.md",
            "kept/h.md",
        ]
        for name in files:
            _write(base / name, "# Page\n\nSome words.\n")
        _write(base / ".lilbeeignore", "skipped/\nskip-me.md\n")
        cfg.ignore_dirs = frozenset({"node_modules"})
        cfg.linked_roots = {"tree": str(base)}
        walked = {key.removeprefix("tree/") for key in discover_files()}
        rules = IgnoreRules.for_corpus(cfg.data_root)

        reached = {name for name in files if walk_reaches(base, base / name, rules)}

        assert reached == walked == {"a.md", "sub/b.md", "kept/h.md"}
        assert walk_reaches(base, base / "sub", rules)
        assert not walk_reaches(base, base / "skipped", rules)
        assert walk_reaches(base, base / "gone" / "folder", rules)
        assert not walk_reaches(base, base / ".gone", rules)


class TestACancelledJsonAdd:
    async def test_it_names_the_folder_that_stays_and_what_it_took_in(self, library, notes, capsys):
        import threading

        from lilbee.app.ingest import AddRollback
        from lilbee.cli.commands import ingest_sync

        await _add(notes / "work")
        rollback = AddRollback(paths=[notes], at_sync=False)
        cfg.json_mode = True

        def _cancelled(_cancel, before_sync):
            raise KeyboardInterrupt

        with (
            mock.patch.object(ingest_sync, "_run_sync", _cancelled),
            pytest.raises(KeyboardInterrupt),
        ):
            ingest_sync._register_and_sync(
                [notes],
                [],
                sync_urls=False,
                force=False,
                cancel_event=threading.Event(),
                rollback=rollback,
            )
        with pytest.raises(SystemExit):
            ingest_sync._exit_cancelled(rollback)

        payload = json.loads(capsys.readouterr().out)
        assert payload["copied"] == ["notes"] and payload["absorbed"] == ["work"]
        assert payload["error"] and "not_added" not in payload

    def test_a_cancel_that_dropped_a_folder_names_no_absorb(self, capsys):
        from lilbee.app.ingest import AddRollback
        from lilbee.cli.commands import ingest_sync

        cfg.json_mode = True
        try:
            with pytest.raises(SystemExit):
                ingest_sync._exit_cancelled(AddRollback(not_added=["notes"]))
        finally:
            cfg.json_mode = False

        payload = json.loads(capsys.readouterr().out)
        assert payload["not_added"] == ["notes"]
        assert "copied" not in payload and "absorbed" not in payload
