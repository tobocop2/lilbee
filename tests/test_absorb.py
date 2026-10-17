"""Adding the parent of a registered source takes the parent and folds the source in."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import shutil
import sqlite3
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar
from unittest import mock
from unittest.mock import MagicMock

import pytest
from filelock import ReadWriteLock
from filelock import Timeout as FileLockTimeout

import lilbee.app.services as svc_mod
from lilbee.app import absorb as absorb_mod
from lilbee.app import ingest as ingest_mod
from lilbee.app.absorb import SYNC_RUNNING_ADD_AGAIN, finish_pending_absorb
from lilbee.app.ingest import (
    forget_unfinished_on_cancel,
    register_sources,
    remove_documents_durably,
)
from lilbee.app.reset import perform_reset
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.data.ingest import sync
from lilbee.data.ingest.skip_marker import (
    REMOVED_SKIP_REASON,
    SKIP_KIND_FILENAME,
    SKIP_MARKER_FILENAME,
    SKIP_REASON_FILENAME,
    SkipKind,
    clear_failed_markers,
    load_skip_kinds,
    load_skip_markers,
    load_skip_reasons,
)
from lilbee.data.store import Store
from lilbee.runtime.absorb_journal import (
    AbsorbJournal,
    AbsorbJournalError,
    AbsorbPhase,
    HeldOut,
    absorb_pending,
    journal_path,
    read_journal,
    write_journal,
)
from lilbee.runtime.lock import (
    LibraryUnlockableError,
    LockingUnsupportedError,
    ResetRefusedError,
    SyncRunningError,
    _no_sync_running,
    library_held,
    source_keys_in_use,
    sync_running,
)
from tests.conftest import (
    UNFINISHED_ADD_REFUSALS,
    UnfinishedAdd,
    cancel_set_when_the_sync_lock_is_granted,
    filesystem_without_locking,
    leave_unfinished_add,
    make_mock_services,
    reset_held_on_a_thread,
    sync_lock_not_a_lock_file,
    unfinished_add_refused,
)
from tests.test_lock import (
    _mark_held_on_a_thread,
    _reset_held_in_another_process,
    _sync_take_that_ends_busy,
    _sync_write_takes,
)
from tests.test_skip_marker import (
    _records_lock_is_free,
    _SetAsTheLockFrees,
    _SetOnSecondPoll,
    record_file_reads,
)
from tests.test_store import _KEY_COLUMNS, _dump, _holders, _mention
from tests.test_wiki_shared import leave_temp_of_a_dead_writer


def _finish_as(names: list[str] | None, notes: Path, cancel) -> None:
    """Finish an interrupted add as a remove or a sync does, or as the add of *notes* does."""
    if names is None:
        finish_pending_absorb(cancel=cancel)
    else:
        register_sources([notes], cancel=cancel)


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
                "citation_key": f"src{abs(hash(key)) % 97}",
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
    return write_page(wiki_root, subdir, "boeing", content, 1.0, keys, "entities")


def _save_stub_index(store: Store) -> None:
    from lilbee.wiki.stubs import _stubs_from_mention_rows, save_stub_index

    cfg.wiki_entity_min_mentions = 1
    stubs = _stubs_from_mention_rows(store.wiki_mention_rows(), cfg, cfg.wiki_stub_max_chunk_refs)
    save_stub_index(stubs, cfg)


def _hold_out(key: str, kind: SkipKind) -> None:
    """Record *key* as held out, by the writer a sync or a remove uses."""
    from lilbee.data.ingest.skip_marker import SkipRecords, mark_removed, update_skip_records

    if kind is SkipKind.REMOVED:
        mark_removed(cfg.data_root, {key: f"hash of {key}"})
        return

    def _fail(records: SkipRecords) -> None:
        records.markers[key] = f"hash of {key}"
        records.reasons[key] = "no text"
        records.kinds[key] = SkipKind.FAILED

    update_skip_records(cfg.data_root, _fail)


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
        "markers": load_skip_markers(cfg.data_root),
        "reasons": load_skip_reasons(cfg.data_root),
        "kinds": load_skip_kinds(cfg.data_root),
        "wiki": {
            path.relative_to(wiki_root).as_posix(): "\n".join(
                line
                for line in path.read_text(encoding="utf-8").split("\n")
                if not line.startswith("generated_at: ")
            )
            for path in sorted(wiki_root.rglob("*"))
            if path.is_file()
        },
        "registry": sorted(_registry()),
        "journal": absorb_pending(cfg.data_root),
    }


def _seeded_child(store: Store, parent: Path, child: str = "work") -> Path:
    """Register ``parent/child`` and seed two of its files into every key holder."""
    child_path = parent / child
    assert register_sources([child_path]).registered == [child]
    for name in ("plan.md", "budget.md"):
        _seed(store, f"{child}/{name}")
    _seed(store, "workshop/x.md")
    _hold_out(f"{child}/gone.md", SkipKind.REMOVED)
    _hold_out(f"{child}/broken.md", SkipKind.FAILED)
    _write_wiki_page(store, [f"{child}/budget.md", f"{child}/plan.md"])
    _save_stub_index(store)
    return child_path


def _register_as(roots: dict[str, Path]) -> None:
    """Write *roots* as the registry, label to path, as a hand edit leaves it."""
    stored = {label: str(path.resolve()) for label, path in roots.items()}
    settings.set_value(cfg.data_root, "linked_roots", stored)
    cfg.linked_roots = dict(stored)


def _seeded_renamed_child(store: Store, parent: Path) -> Path:
    """``parent/work`` registered under the label ``renamed``, seeded into every key holder."""
    _register_as({"renamed": parent / "work"})
    for name in ("plan.md", "budget.md"):
        _seed(store, f"renamed/{name}")
    _seed(store, "workshop/x.md")
    _hold_out("renamed/gone.md", SkipKind.REMOVED)
    _hold_out("renamed/broken.md", SkipKind.FAILED)
    _write_wiki_page(store, ["renamed/budget.md", "renamed/plan.md"])
    _save_stub_index(store)
    return parent


def _seeded_nested_children(store: Store, parent: Path) -> Path:
    """``parent/work`` and ``parent/work/deep`` both registered, so ``deep`` is indexed twice."""
    _write(parent / "work" / "deep" / "a.md", "# A\n\nA deep page.\n")
    _register_as({"work": parent / "work", "deep": parent / "work" / "deep"})
    for key in ("work/plan.md", "work/deep/a.md", "work/deep/b.md", "deep/a.md", "deep/b.md"):
        _seed(store, key)
    _seed(store, "work/deep/gone.md")  # removed from the inner source only
    outer_copy = dict(store.get_citations_for_source("deep/a.md")[0])
    store.add_citations([{**outer_copy, "source_filename": "work/deep/a.md", "created_at": "t"}])
    _seed(store, "workshop/x.md")
    _hold_out("deep/gone.md", SkipKind.REMOVED)
    _hold_out("work/deep/kept.md", SkipKind.REMOVED)
    _hold_out("work/broken.md", SkipKind.FAILED)
    _write_wiki_page(store, ["deep/a.md", "work/deep/a.md", "work/plan.md"])
    _save_stub_index(store)
    return parent


_LEFT_FOR_THE_NEXT_SYNC = "An interrupted add is not finished yet; the next sync finishes it."

_NESTED_KEYS = [
    "notes/work/deep/a.md",
    "notes/work/deep/b.md",
    "notes/work/plan.md",
    "workshop/x.md",
]


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

    def test_every_per_source_table_follows_the_absorb(self, library, notes):
        store = library.store
        _seeded_child(store, notes)
        assert _holders(store, "work/plan.md") == _KEY_COLUMNS

        register_sources([notes])

        assert _holders(store, "notes/work/plan.md") == _KEY_COLUMNS
        assert _holders(store, "notes/work/budget.md") == _KEY_COLUMNS
        assert _holders(store, "work/plan.md") == set()
        assert _holders(store, "workshop/x.md") == _KEY_COLUMNS
        assert not [row for rows in _dump(store).values() for row in rows if ".absorb-" in row]

    def test_the_wiki_page_and_the_subject_index_follow_the_absorb(self, library, notes):
        from lilbee.wiki.shared import parse_frontmatter
        from lilbee.wiki.stubs import load_stub_index

        store = library.store
        _seeded_child(store, notes)
        page = cfg.data_root / cfg.wiki_dir / "entities" / "boeing.md"
        before = page.read_text(encoding="utf-8")
        assert "work/plan.md, lines 1-2" in before
        assert load_stub_index(cfg)["boeing"].sources == (
            "work/budget.md",
            "work/plan.md",
            "workshop/x.md",
        )

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
        assert stub.sources == ("notes/work/budget.md", "notes/work/plan.md", "workshop/x.md")
        assert {source for source, _index in stub.chunk_refs} == set(stub.sources)

    def test_a_removed_file_stays_removed_and_a_failed_file_keeps_its_record(self, library, notes):
        _seeded_child(library.store, notes)
        _hold_out("workshop/kept.md", SkipKind.FAILED)

        register_sources([notes])

        assert load_skip_markers(cfg.data_root) == {
            "notes/work/gone.md": "hash of work/gone.md",
            "notes/work/broken.md": "hash of work/broken.md",
            "workshop/kept.md": "hash of workshop/kept.md",
        }
        assert load_skip_kinds(cfg.data_root) == {
            "notes/work/gone.md": SkipKind.REMOVED,
            "notes/work/broken.md": SkipKind.FAILED,
            "workshop/kept.md": SkipKind.FAILED,
        }
        reasons = load_skip_reasons(cfg.data_root)
        assert reasons["notes/work/broken.md"] == "no text"
        stored = json.loads((cfg.data_root / SKIP_KIND_FILENAME).read_text(encoding="utf-8"))
        assert sorted(stored) == sorted(reasons) == sorted(load_skip_markers(cfg.data_root))

    async def test_a_file_removed_from_the_child_does_not_return(self, library, notes):
        await _add(notes / "work")
        remove_documents_durably(["work/plan.md"])
        assert _keys(library) == ["work/budget.md"]

        _result, synced = await _add(notes)

        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md"]
        assert synced.added == ["notes/loose.md"]

    async def test_naming_the_parent_outranks_a_marker_on_its_own_file(self, library, notes):
        """The rule every add applies to the path it names, kept for the parent's own files."""
        await _add(notes / "work")
        _hold_out("notes/loose.md", SkipKind.REMOVED)
        _hold_out("work/gone.md", SkipKind.REMOVED)

        register_sources([notes])

        assert sorted(load_skip_markers(cfg.data_root)) == ["notes/work/gone.md"]

    async def test_several_children_are_absorbed_in_one_operation(self, library, notes):
        _write(notes / "home" / "list.md", "# List\n\nThe list for the market.\n")
        register_sources([notes / "work", notes / "home"])
        await sync(quiet=True)
        journals: list[AbsorbJournal] = []
        real = absorb_mod.write_journal

        def _record(data_root, journal):
            journals.append(journal)
            real(data_root, journal)

        with mock.patch.object(absorb_mod, "write_journal", _record):
            result = register_sources([notes])

        assert result.absorbed_into == {"notes": ["home", "work"]}
        assert _registry() == {"notes": str(notes.resolve())}
        assert _keys(library) == [
            "notes/home/list.md",
            "notes/work/budget.md",
            "notes/work/plan.md",
        ]
        assert [journal.phase for journal in journals] == [AbsorbPhase.LIFT, AbsorbPhase.LAND]
        assert journals[0].moves == {"work": "notes/work", "home": "notes/home"}
        assert len({journal.id for journal in journals}) == 1

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

    def test_the_inner_source_decides_what_its_files_keep(self, library, notes):
        """Of two entries for one file, the one the inner source wrote lands."""
        store = library.store
        _seeded_nested_children(store, notes)

        register_sources([notes])

        assert _keys(library) == _NESTED_KEYS
        rows = {source["filename"]: source["file_hash"] for source in store.get_sources()}
        assert rows["notes/work/deep/a.md"] == "hash of deep/a.md"
        assert _holders(store, "notes/work/deep/a.md") == _KEY_COLUMNS
        assert len(store.get_chunks_by_source("notes/work/deep/a.md")) == 1
        markers = load_skip_markers(cfg.data_root)
        assert sorted(markers) == ["notes/work/broken.md", "notes/work/deep/gone.md"]
        assert load_skip_kinds(cfg.data_root)["notes/work/broken.md"] is SkipKind.FAILED
        page = cfg.data_root / cfg.wiki_dir / "entities" / "boeing.md"
        from lilbee.wiki.shared import parse_frontmatter

        assert parse_frontmatter(page.read_text(encoding="utf-8"))["sources"] == [
            "notes/work/deep/a.md",
            "notes/work/plan.md",
        ]

    def test_landing_without_the_drop_indexes_what_the_inner_source_removed(self, library, notes):
        """The control for the test above: the outer source's entry for a removed file lands."""
        _seeded_nested_children(library.store, notes)

        with mock.patch.object(absorb_mod, "_drop_shadows"):
            register_sources([notes])

        assert "notes/work/deep/gone.md" in _keys(library)

    def test_a_citation_both_nested_sources_hold_lands_once(self, library, notes):
        store = library.store
        _seeded_nested_children(store, notes)
        assert len(store.get_citations_for_source("work/deep/a.md")) == 2

        register_sources([notes])

        cited = store.get_citations_for_source("notes/work/deep/a.md")
        keys = {f"src{abs(hash(key)) % 97}" for key in ("deep/a.md", "work/deep/a.md")}
        assert sorted(row["citation_key"] for row in cited) == sorted(keys)
        page = (cfg.data_root / cfg.wiki_dir / "entities" / "boeing.md").read_text(encoding="utf-8")
        footnotes = [line for line in page.split("\n") if line.startswith("[^src")]
        assert footnotes and len(footnotes) == len(set(footnotes))
        assert page.count("- source: notes/work/deep/a.md") == 1

    def test_a_drop_after_the_land_finds_nothing_to_drop(self, library, notes):
        """The wrong order: the outer source's entries have left the temporary keys."""
        store = library.store
        _seeded_nested_children(store, notes)
        real_drop = absorb_mod._drop_shadows
        journals: list[AbsorbJournal] = []

        with mock.patch.object(absorb_mod, "_drop_shadows", lambda _s, j: journals.append(j)):
            register_sources([notes])
        real_drop(store, journals[0])

        assert "notes/work/deep/gone.md" in _keys(library)


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
    async def test_the_adds_leave_the_file_of_the_inner_source_at_two_keys(self, library, tmp_path):
        after = await _moved_folder_with_a_source_inside(tmp_path)

        assert sorted(_registry()) == ["deep", "work"]
        assert _registry()["work"] == str(after.resolve())
        assert _keys(library) == ["deep/a.md", "work/deep/a.md", "work/plan.md"]

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

    async def test_without_the_drop_the_outer_entry_lands_beside_it(self, library, tmp_path):
        """The control for the test above."""
        after = await _moved_folder_with_a_source_inside(tmp_path)

        with mock.patch.object(absorb_mod, "_drop_shadows"):
            register_sources([after.parent])

        assert _keys(library).count("after/work/deep/a.md") == 2

    async def test_a_removal_of_the_outer_entry_does_not_hold_the_file_out(self, library, tmp_path):
        after = await _moved_folder_with_a_source_inside(tmp_path)
        remove_documents_durably(["work/deep/a.md"])
        assert sorted(load_skip_markers(cfg.data_root)) == ["work/deep/a.md"]

        await _add(after.parent)

        assert load_skip_markers(cfg.data_root) == {}
        assert _keys(library) == ["after/work/deep/a.md", "after/work/plan.md"]


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
        assert not absorb_pending(cfg.data_root)

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
        with mock.patch.object(absorb_mod, "write_journal") as journal:
            result = register_sources([notes / "work", notes])

        assert result.registered == ["notes"]
        assert result.absorbed_into == {}
        assert _registry() == {"notes": str(notes.resolve())}
        journal.assert_not_called()

    async def test_a_registered_child_named_beside_its_parent_is_absorbed(self, library, notes):
        await _add(notes / "work")

        result = register_sources([notes / "work", notes])

        assert result.registered == ["notes"]
        assert result.absorbed_into == {"notes": ["work"]}
        assert _registry() == {"notes": str(notes.resolve())}
        assert _keys(library) == ["notes/work/budget.md", "notes/work/plan.md"]


class TestTheCommonAddPaysNothing:
    def test_no_child_means_no_journal_and_no_lock(self, library, notes):
        opened: list[str] = []
        real_exists = Path.exists

        def _exists(path, **kwargs):
            if path.name == "pending_absorb.json":
                opened.append(path.name)
            return real_exists(path, **kwargs)

        with (
            mock.patch.object(ingest_mod, "library_held") as held_off,
            mock.patch.object(absorb_mod, "write_journal") as journal,
            mock.patch.object(absorb_mod, "_rekey") as rekey,
            mock.patch("lilbee.app.ingest.IgnoreRules.for_corpus") as rules,
            mock.patch.object(Path, "exists", _exists),
        ):
            result = register_sources([notes])

        assert result.registered == ["notes"] and result.absorbed == []
        # Once when the add arrives and once under the records lock.
        assert opened == ["pending_absorb.json"] * 2
        held_off.assert_not_called()
        journal.assert_not_called()
        rekey.assert_not_called()
        rules.assert_not_called()
        # The add opened the sync lock to check it and left it free.
        with _no_sync_running(cfg.data_root):
            pass
        assert library.store.get_sources() == []

    async def test_an_add_without_records_reads_no_record_file(self, library, notes, tmp_path):
        with record_file_reads() as reads:
            await _add(notes)

        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md", "notes/work/plan.md"]
        assert reads == []

    async def test_an_add_reads_each_record_file_once_per_step(self, library, notes, tmp_path):
        """The control for the test above: with records on disk the same add reads them."""
        await _add(notes)
        remove_documents_durably(["notes/loose.md"])
        other = _write(tmp_path / "src" / "other.md", "# Other\n\nA page about tide tables.\n")

        with record_file_reads() as register_reads:
            register_sources([other])
        with record_file_reads() as sync_reads:
            await sync(quiet=True)

        assert Counter(register_reads) == {
            SKIP_MARKER_FILENAME: 1,
            SKIP_REASON_FILENAME: 1,
            SKIP_KIND_FILENAME: 1,
        }
        # The plan reads the markers; the merge of the verdicts and the result read all three.
        assert Counter(sync_reads) == {
            SKIP_MARKER_FILENAME: 3,
            SKIP_REASON_FILENAME: 2,
            SKIP_KIND_FILENAME: 2,
        }
        assert load_skip_kinds(cfg.data_root) == {"notes/loose.md": SkipKind.REMOVED}

    async def test_an_absorb_does_take_the_lock_and_write_the_journal(self, library, notes):
        """The control for the test above: the same probes fire when a child is there."""
        await _add(notes / "work")
        real_held_off = ingest_mod.library_held
        with (
            mock.patch.object(ingest_mod, "library_held", side_effect=real_held_off) as held_off,
            mock.patch("lilbee.app.ingest.IgnoreRules.for_corpus", wraps=None) as rules,
        ):
            rules.return_value.excludes_entry.return_value = False
            register_sources([notes])

        held_off.assert_called_once()
        # Once to choose the hold and once under it.
        assert rules.call_count == 2

    async def test_a_sync_without_a_journal_checks_once_and_finishes_nothing(self, library, notes):
        with mock.patch("lilbee.data.ingest.pipeline._finish_pending_absorb") as finish:
            await _add(notes)

        finish.assert_not_called()


class TestASyncIsRunning:
    async def test_an_absorb_is_refused_while_a_sync_runs(self, library, notes):
        await _add(notes / "work")
        before = _everything(library.store)

        async with sync_running(cfg.data_root):
            with pytest.raises(SyncRunningError) as refused:
                register_sources([notes])

        assert (
            str(refused.value) == "A sync or a wiki build is running. Add notes again when it ends."
        )
        assert _everything(library.store) == before
        assert cfg.linked_roots == {"work": str((notes / "work").resolve())}

    async def test_a_plain_add_goes_through_while_a_sync_runs(self, library, notes):
        async with sync_running(cfg.data_root):
            result = register_sources([notes])

        assert result.registered == ["notes"]

    async def test_a_folder_and_its_parent_in_one_add_go_through_while_a_sync_runs(
        self, library, notes
    ):
        async with sync_running(cfg.data_root):
            result = register_sources([notes / "work", notes])

        assert result.registered == ["notes"] and result.absorbed == []
        assert _registry() == {"notes": str(notes.resolve())}

    async def test_a_sync_mark_taken_during_the_absorb_waits_until_the_journal_is_gone(
        self, library, notes
    ):
        """A thread that takes the mark a sync takes gets it only when the absorb has ended."""
        import threading

        await _add(notes / "work")
        waiting = threading.Event()
        states: list[bool] = []

        def _second_sync() -> None:
            waiting.set()
            with source_keys_in_use(cfg.data_root):
                states.append(absorb_pending(cfg.data_root))

        real_land = absorb_mod._land

        def _land_while_a_sync_waits(config, store, journal):
            thread = threading.Thread(target=_second_sync)
            thread.start()
            assert waiting.wait(5)
            thread.join(0.3)
            assert thread.is_alive(), "the sync started in the middle of the absorb"
            real_land(config, store, journal)
            threads.append(thread)

        threads: list[threading.Thread] = []
        with mock.patch.object(absorb_mod, "_land", _land_while_a_sync_waits):
            register_sources([notes])
        threads[0].join(5)

        assert states == [False]


class TestInterruptedAbsorb:
    _KILL_POINTS = (
        "journal-written",
        "lift-third-table",
        "lift-wiki-pages",
        "clear-targets",
        "phase-write",
        "land-start",
        "land-third-table",
        "land-wiki-pages",
        "skip-records",
        "wiki-index",
        "registry",
        "journal-delete",
    )

    @staticmethod
    def _nth_call(target, attribute, nth):
        """Patch *attribute* so its *nth* call dies before it runs."""
        real = getattr(target, attribute)
        calls = {"count": 0}

        def _dies(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == nth:
                raise _Killed
            return real(*args, **kwargs)

        return mock.patch.object(target, attribute, _dies)

    def _kill_at(self, point: str):
        from lancedb.table import LanceTable

        import lilbee.wiki.rekey as wiki_rekey_mod

        tables = len(_KEY_COLUMNS)
        return {
            "journal-written": lambda: self._nth_call(Store, "rekey_sources_under", 1),
            "lift-third-table": lambda: self._nth_call(LanceTable, "update", 4),
            "lift-wiki-pages": lambda: self._nth_call(wiki_rekey_mod, "rekey_wiki_pages", 1),
            "clear-targets": lambda: self._nth_call(absorb_mod, "_clear_targets", 1),
            "phase-write": lambda: self._nth_call(absorb_mod, "write_journal", 2),
            "land-start": lambda: self._nth_call(absorb_mod, "_land", 1),
            "land-third-table": lambda: self._nth_call(LanceTable, "update", tables + 4),
            "skip-records": lambda: self._nth_call(absorb_mod, "_write_records", 1),
            "land-wiki-pages": lambda: self._nth_call(wiki_rekey_mod, "rekey_wiki_pages", 2),
            "wiki-index": lambda: self._nth_call(absorb_mod, "_refresh_wiki_index", 1),
            "registry": lambda: self._nth_call(absorb_mod, "_write_registry", 1),
            "journal-delete": lambda: self._nth_call(absorb_mod, "delete_journal", 1),
        }[point]()

    def _clean_absorb(self, tmp_path, label="notes", child="work", seed=None) -> dict[str, object]:
        """The state a whole absorb leaves, built in a second library."""
        saved = (cfg.data_root, cfg.documents_dir, cfg.data_dir, cfg.lancedb_dir, cfg.linked_roots)
        services = svc_mod.get_services()
        root = tmp_path / "twin-library"
        cfg.data_root, cfg.documents_dir = root, root / "documents"
        cfg.data_dir, cfg.lancedb_dir = root / "data", root / "data" / "lancedb"
        cfg.linked_roots = {}
        store = Store(cfg)
        svc_mod.set_services(make_mock_services(store=store))
        parent = tmp_path / "src" / label
        if seed is None:
            _seeded_child(store, parent, child)
        else:
            seed(store, parent)
        register_sources([parent])
        state = _everything(store)
        svc_mod.set_services(services)
        cfg.data_root, cfg.documents_dir, cfg.data_dir, cfg.lancedb_dir, cfg.linked_roots = saved
        return state

    @pytest.mark.parametrize("point", _KILL_POINTS)
    def test_a_kill_after_each_step_ends_in_the_new_state(self, library, notes, tmp_path, point):
        store = library.store
        _seeded_child(store, notes)
        before = _everything(store)

        with self._kill_at(point), pytest.raises(_Killed):
            register_sources([notes])

        assert absorb_pending(cfg.data_root)
        killed = _everything(store)
        cfg.linked_roots = {"work": str((notes / "work").resolve())}  # what a new process loads
        finish_pending_absorb()

        clean = self._clean_absorb(tmp_path)
        finished = _everything(store)
        assert finished["tables"] == clean["tables"]
        assert finished == clean
        assert clean != before and clean["registry"] == ["notes"]
        assert killed != clean
        assert cfg.linked_roots == {"notes": str(notes.resolve())}

    _TABLES = len(_KEY_COLUMNS)
    _SHAPES: ClassVar[dict[str, tuple[Callable[[Store, Path], Path], dict[str, str]]]] = {
        "renamed": (_seeded_renamed_child, {"renamed": "work"}),
        "nested": (_seeded_nested_children, {"work": "work", "deep": "work/deep"}),
    }
    # One source moves three times (lift, the lift again, land) and two sources six.
    _SHAPE_POINTS = (
        *(("renamed", "rekey", nth) for nth in (1, 2)),
        ("renamed", "records", 1),
        ("renamed", "table", 4),
        ("renamed", "table", _TABLES + 6),
        *(("nested", "rekey", nth) for nth in range(1, 5)),
        ("nested", "records", 1),
        ("nested", "table", 4),
        ("nested", "table", _TABLES + 4),
        ("nested", "table", 2 * _TABLES + 4),
        *(("nested", "clear", nth) for nth in (1, 2)),
        ("nested", "citations", 1),
        *((shape, step, 1) for shape in _SHAPES for step in ("wiki-index", "registry", "delete")),
        *((shape, "phase", 2) for shape in _SHAPES),
    )

    def _kill_nth(self, step: str, nth: int):
        from lancedb.table import LanceTable

        target, attribute = {
            "rekey": (Store, "rekey_sources_under"),
            "table": (LanceTable, "update"),
            "records": (absorb_mod, "_write_records"),
            "clear": (absorb_mod, "_clear_targets"),
            "citations": (Store, "drop_citations_repeated_under"),
            "phase": (absorb_mod, "write_journal"),
            "wiki-index": (absorb_mod, "_refresh_wiki_index"),
            "registry": (absorb_mod, "_write_registry"),
            "delete": (absorb_mod, "delete_journal"),
        }[step]
        return self._nth_call(target, attribute, nth)

    @pytest.mark.parametrize(("shape", "step", "nth"), _SHAPE_POINTS)
    def test_a_kill_in_a_renamed_or_nested_absorb_ends_in_the_new_state(
        self, library, notes, tmp_path, shape, step, nth
    ):
        store = library.store
        seed, roots = self._SHAPES[shape]
        seed(store, notes)
        before = _everything(store)

        with self._kill_nth(step, nth), pytest.raises(_Killed):
            register_sources([notes])

        assert absorb_pending(cfg.data_root)
        killed = _everything(store)
        cfg.linked_roots = {label: str((notes / below).resolve()) for label, below in roots.items()}
        finish_pending_absorb()
        finish_pending_absorb()

        clean = self._clean_absorb(tmp_path, seed=seed)
        finished = _everything(store)
        assert finished["tables"] == clean["tables"]
        assert finished == clean
        assert clean != before and clean["registry"] == ["notes"]
        assert killed != clean
        keys = _keys(library)
        assert len(keys) == len(set(keys)) and "notes/work/plan.md" in keys
        assert cfg.linked_roots == {"notes": str(notes.resolve())}

    @pytest.mark.parametrize("point", _KILL_POINTS)
    def test_parent_and_child_with_one_label_are_rekeyed_once(self, library, tmp_path, point):
        store = library.store
        parent = tmp_path / "src" / "project"
        _write(parent / "project" / "plan.md", "# Plan\n\nThe plan.\n")
        _write(parent / "project" / "project" / "deep.md", "# Deep\n\nA deep page.\n")
        _seeded_child(store, parent, "project")
        _seed(store, "project/project/deep.md")

        with self._kill_at(point), pytest.raises(_Killed):
            register_sources([parent])
        finish_pending_absorb()
        finish_pending_absorb()

        keys = sorted(source["filename"] for source in store.get_sources())
        assert keys == [
            "project/project/budget.md",
            "project/project/plan.md",
            "project/project/project/deep.md",
            "workshop/x.md",
        ]
        assert _holders(store, "project/project/plan.md") == _KEY_COLUMNS
        assert _holders(store, "project/project/project/plan.md") == set()
        assert sorted(load_skip_markers(cfg.data_root)) == [
            "project/project/broken.md",
            "project/project/gone.md",
        ]
        page = cfg.data_root / cfg.wiki_dir / "entities" / "boeing.md"
        text = page.read_text(encoding="utf-8")
        assert "project/project/plan.md" in text and "project/project/project" not in text
        assert _registry() == {"project": str(parent.resolve())}

    def test_a_remove_finishes_an_interrupted_absorb_before_it_removes(self, library, tmp_path):
        """A process that was running when another one's add died removes on the moved keys."""
        store = library.store
        parent = tmp_path / "src" / "project"
        _write(parent / "project" / "budget.md", "# Budget\n\nThe budget.\n")
        _seeded_child(store, parent, "project")
        with self._kill_at("journal-written"), pytest.raises(_Killed):
            register_sources([parent])

        stale = remove_documents_durably(["project/budget.md"])

        assert not absorb_pending(cfg.data_root)
        assert stale.removed == [] and stale.not_found == ["project/budget.md"]
        assert _keys(library) == [
            "project/project/budget.md",
            "project/project/plan.md",
            "workshop/x.md",
        ]

        moved = remove_documents_durably(["project/project/budget.md"])

        assert moved.removed == ["project/project/budget.md"]
        assert _keys(library) == ["project/project/plan.md", "workshop/x.md"]
        assert load_skip_kinds(cfg.data_root)["project/project/budget.md"] is SkipKind.REMOVED

    def test_a_remove_expands_its_names_again_after_it_finished_an_absorb(self, library, notes):
        store = library.store
        _seeded_child(store, notes)
        with self._kill_at("land-start"), pytest.raises(_Killed):
            register_sources([notes])

        result = remove_documents_durably(["work"], targets=["work/plan.md", "work/budget.md"])

        assert result.removed == [] and result.not_found == ["work"]
        assert _keys(library) == ["notes/work/budget.md", "notes/work/plan.md", "workshop/x.md"]
        assert _registry() == {"notes": str(notes.resolve())}

    def test_a_remove_without_a_journal_uses_the_names_it_was_given(self, library, notes):
        store = library.store
        _seeded_child(store, notes)

        with mock.patch("lilbee.app.ingest.finish_absorb") as finish:
            result = remove_documents_durably(["anything"], targets=["work/plan.md"])

        assert result.removed == ["work/plan.md"]
        finish.assert_not_called()

    def test_a_single_phase_rekey_would_double_the_prefix(self, library):
        """The control for the test above: the store method alone, run twice, doubles it."""
        store = library.store
        _seed(store, "project/plan.md")

        store.rekey_sources_under("project", "project/project")
        store.rekey_sources_under("project", "project/project")

        assert _holders(store, "project/project/project/plan.md") == _KEY_COLUMNS

    def test_a_root_named_like_the_temporary_prefix_keeps_its_keys(self, library, notes):
        store = library.store
        _seeded_child(store, notes)
        _seed(store, ".absorb-00000000/work/plan.md")
        _seed(store, ".absorb/plan.md")

        register_sources([notes])

        assert _holders(store, ".absorb-00000000/work/plan.md") == _KEY_COLUMNS
        assert _holders(store, ".absorb/plan.md") == _KEY_COLUMNS
        assert _holders(store, "notes/work/plan.md") == _KEY_COLUMNS

    async def test_a_sync_finishes_an_interrupted_absorb_before_it_walks(self, library, notes):
        await _add(notes / "work")
        with self._kill_at("phase-write"), pytest.raises(_Killed):
            register_sources([notes])
        assert absorb_pending(cfg.data_root)
        assert [key for key in _keys(library) if key.startswith(".absorb-")]

        synced = await sync(quiet=True)

        assert not absorb_pending(cfg.data_root)
        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md", "notes/work/plan.md"]
        assert synced.added == ["notes/loose.md"]
        assert synced.unchanged == 2

    async def test_the_next_add_finishes_an_interrupted_absorb_first(
        self, library, notes, tmp_path
    ):
        await _add(notes / "work")
        with self._kill_at("clear-targets"), pytest.raises(_Killed):
            register_sources([notes])
        other = _write(tmp_path / "other" / "o.md", "# O\n\nAnother page.\n").parent

        result = register_sources([other])

        assert result.registered == ["other"]
        assert not absorb_pending(cfg.data_root)
        assert sorted(_registry()) == ["notes", "other"]
        assert _keys(library) == ["notes/work/budget.md", "notes/work/plan.md"]

    async def test_the_next_add_is_refused_while_a_sync_runs_on_a_journal(
        self, library, notes, tmp_path
    ):
        await _add(notes / "work")
        with self._kill_at("phase-write"), pytest.raises(_Killed):
            register_sources([notes])
        other = _write(tmp_path / "other" / "o.md", "# O\n\nAnother page.\n").parent

        async with sync_running(cfg.data_root):
            with pytest.raises(SyncRunningError, match="Add other again when it ends"):
                register_sources([other])

        assert absorb_pending(cfg.data_root)
        assert "other" not in _registry()

    async def test_a_new_process_finishes_an_interrupted_absorb_when_it_starts(
        self, library, notes, monkeypatch
    ):
        await _add(notes / "work")
        with self._kill_at("land-start"), pytest.raises(_Killed):
            register_sources([notes])
        store = library.store
        svc_mod.set_services(None)
        monkeypatch.setattr(svc_mod._state, "singleton", None)
        built = make_mock_services(store=store)

        with (
            mock.patch.object(svc_mod, "build_services", return_value=built),
            mock.patch("lilbee.app.settings.reconcile_embedding_dim"),
            mock.patch("lilbee.modelhub.registry.ModelRegistry"),
        ):
            cfg.worker_pool_eager_start = False
            assert svc_mod.get_services() is built

        assert not absorb_pending(cfg.data_root)
        assert _keys(built) == ["notes/work/budget.md", "notes/work/plan.md"]
        assert _registry() == {"notes": str(notes.resolve())}

    async def test_a_start_while_a_sync_runs_leaves_the_journal_and_says_so_at_info(
        self, library, notes, caplog
    ):
        from lilbee.app.absorb import finish_pending_absorb_at_start

        await _add(notes / "work")
        with self._kill_at("phase-write"), pytest.raises(_Killed):
            register_sources([notes])

        with caplog.at_level("INFO", logger="lilbee.app.absorb"):
            async with sync_running(cfg.data_root):
                finish_pending_absorb_at_start(cfg, library.store)

        assert absorb_pending(cfg.data_root)
        assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
            ("INFO", _LEFT_FOR_THE_NEXT_SYNC)
        ]

    async def test_a_start_on_a_data_root_it_cannot_lock_warns_with_the_refusal(
        self, library, notes, caplog
    ):
        from lilbee.app.absorb import finish_pending_absorb_at_start

        await _add(notes / "work")
        with self._kill_at("phase-write"), pytest.raises(_Killed):
            register_sources([notes])

        with (
            caplog.at_level("INFO", logger="lilbee.app.absorb"),
            _data_root_that_cannot_be_locked(UnfinishedAdd.UNLOCKABLE),
        ):
            finish_pending_absorb_at_start(cfg, library.store)

        assert absorb_pending(cfg.data_root)
        [record] = caplog.records
        assert record.levelname == "WARNING"
        assert UNFINISHED_ADD_REFUSALS[UnfinishedAdd.UNLOCKABLE] in record.getMessage()

    async def test_a_process_that_starts_while_a_sync_runs_still_gets_its_services(
        self, library, notes, monkeypatch, caplog
    ):
        await _add(notes / "work")
        with self._kill_at("phase-write"), pytest.raises(_Killed):
            register_sources([notes])
        svc_mod.set_services(None)
        monkeypatch.setattr(svc_mod._state, "singleton", None)
        built = make_mock_services(store=library.store)

        with (
            mock.patch.object(svc_mod, "build_services", return_value=built),
            mock.patch("lilbee.app.settings.reconcile_embedding_dim"),
            mock.patch("lilbee.modelhub.registry.ModelRegistry"),
        ):
            cfg.worker_pool_eager_start = False
            with caplog.at_level("INFO", logger="lilbee.app.absorb"):
                async with sync_running(cfg.data_root):
                    # The process that starts holds no lock: a context of its own.
                    assert contextvars.Context().run(svc_mod.get_services) is built

        assert absorb_pending(cfg.data_root)
        assert _LEFT_FOR_THE_NEXT_SYNC in caplog.text

    async def test_a_journal_another_process_finished_is_left_alone(self, library, notes):
        """The journal is read again under the lock, after the wait for it."""
        await _add(notes / "work")
        with self._kill_at("phase-write"), pytest.raises(_Killed):
            register_sources([notes])
        real_read = absorb_mod.read_journal

        def _finished_meanwhile(data_root):
            journal_path(data_root).unlink()
            return real_read(data_root)

        with (
            mock.patch.object(absorb_mod, "read_journal", _finished_meanwhile),
            mock.patch.object(absorb_mod, "_roll_forward") as roll_forward,
        ):
            finish_pending_absorb()

        roll_forward.assert_not_called()

    async def test_a_stale_process_sync_does_not_move_keys_back(self, library, notes):
        await _add(notes / "work")
        register_sources([notes])
        cfg.linked_roots = {"work": str((notes / "work").resolve())}  # loaded before the absorb

        synced = await sync(quiet=True)

        assert synced.relocated == []
        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md", "notes/work/plan.md"]
        assert cfg.linked_roots == {"notes": str(notes.resolve())}

    async def test_stale_rows_at_the_keys_the_child_lands_on_are_replaced(self, library, notes):
        """A label re-pointed from a vanished folder can still hold rows at those keys."""
        await _add(notes / "work")
        _seed(library.store, "notes/work/plan.md")
        _seed(library.store, "notes/elsewhere.md")
        assert len([key for key in _keys(library) if key == "notes/work/plan.md"]) == 1

        register_sources([notes])

        keys = _keys(library)
        assert keys.count("notes/work/plan.md") == 1
        assert "notes/elsewhere.md" in keys
        row = next(s for s in library.store.get_sources() if s["filename"] == "notes/work/plan.md")
        assert row["file_hash"] != "hash of notes/work/plan.md"


class TestTheJournal:
    def test_it_round_trips(self, tmp_path):
        journal = AbsorbJournal(
            id="ab12cd34",
            moves={"work": "notes/work"},
            add={"notes": "/x/notes"},
            drop=["work"],
            records={
                "notes/work/gone.md": HeldOut("h1", "removed", removed=True),
                "notes/work/bad.pdf": HeldOut("h2", None, removed=False),
            },
            phase=AbsorbPhase.LAND,
        )

        write_journal(tmp_path, journal)

        assert read_journal(tmp_path) == journal
        assert journal.lifted("work") == ".absorb-ab12cd34/work"
        assert not (tmp_path / "pending_absorb.json.tmp").exists()

    def test_none_without_a_file(self, tmp_path):
        assert read_journal(tmp_path) is None
        assert not absorb_pending(tmp_path)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "{not json",
            "[]",
            '{"id": "a"}',
            '{"id": "a", "moves": [], "add": {}, "drop": []}',
            '{"id": "a", "moves": {}, "add": {}, "drop": [], "phase": "lift"}',
            '{"id": "a", "moves": {}, "add": {}, "drop": [], "phase": "lift", "records": {"k": 1}}',
        ],
        ids=[
            "empty",
            "not-json",
            "a-list",
            "keys-missing",
            "moves-not-a-table",
            "no-records",
            "a-record-without-its-hash",
        ],
    )
    def test_an_unreadable_journal_raises_and_names_the_file(self, tmp_path, text):
        journal_path(tmp_path).write_text(text, encoding="utf-8")

        with pytest.raises(AbsorbJournalError, match=r"pending_absorb\.json") as raised:
            read_journal(tmp_path)

        assert "delete that file and run `lilbee rebuild`" in str(raised.value)

    def test_an_unknown_phase_raises(self, tmp_path):
        journal_path(tmp_path).write_text(
            json.dumps(
                {"id": "a", "moves": {}, "add": {}, "drop": [], "records": {}, "phase": "undo"}
            ),
            encoding="utf-8",
        )

        with pytest.raises(AbsorbJournalError, match="undo"):
            read_journal(tmp_path)

    def test_a_journal_that_cannot_be_written_moves_no_key(self, library, notes):
        store = library.store
        _seeded_child(store, notes)
        before = _everything(store)

        with (
            mock.patch(
                "lilbee.runtime.absorb_journal.os.replace", side_effect=OSError("disk full")
            ),
            pytest.raises(OSError, match="disk full"),
        ):
            register_sources([notes])

        assert _everything(store) == before

    async def test_a_sync_refuses_to_run_on_a_journal_it_cannot_read(self, library, notes):
        await _add(notes / "work")
        journal_path(cfg.data_root).write_text("{not json", encoding="utf-8")

        with pytest.raises(AbsorbJournalError):
            await sync(quiet=True)

        assert _keys(library) == ["work/budget.md", "work/plan.md"]


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
                force=False,
                cancel_event=threading.Event(),
                rollback=rollback,
            )

        assert payload is not None
        assert payload["copied"] == ["notes"] and payload["absorbed"] == ["work"]
        assert payload["overlapping"] == []
        assert json.loads(json.dumps(payload)) == payload
        assert rollback._roots == []

    async def test_the_cli_refusal_is_one_error_line(self, library, notes):
        from typer.testing import CliRunner

        from lilbee.cli.app import app

        await _add(notes / "work")
        async with sync_running(cfg.data_root):
            outcome = CliRunner().invoke(app, ["add", "--data-dir", str(cfg.data_root), str(notes)])

        assert outcome.exit_code == 1
        assert "A sync or a wiki build is running. Add notes again when it ends." in outcome.output
        assert sorted(_registry()) == ["work"]

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

    async def test_the_tui_says_a_sync_is_running_and_starts_none(self, library, notes):
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter

        await _add(notes / "work")
        screen = ChatScreen.__new__(ChatScreen)
        notify = MagicMock()
        reporter = MagicMock(spec=ProgressReporter)
        reporter.is_set.return_value = False
        with (
            mock.patch("lilbee.cli.tui.screens.chat.call_from_thread", notify),
            mock.patch("lilbee.runtime.asyncio_loop.run") as run,
        ):
            async with sync_running(cfg.data_root):
                screen._do_add([notes], reporter)

        run.assert_not_called()
        assert [call.args[2] for call in notify.call_args_list] == [
            SYNC_RUNNING_ADD_AGAIN.format(names="notes")
        ]
        assert notify.call_args.kwargs == {"severity": "warning"}

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

    async def test_the_http_stream_ends_in_an_error_while_a_sync_runs(self, library, notes):
        from lilbee.server.handlers import add_files_stream

        await _add(notes / "work")
        async with sync_running(cfg.data_root):
            frames = [frame async for frame in add_files_stream([str(notes)])]

        text = "".join(frames)
        assert "A sync or a wiki build is running. Add notes again when it ends." in text
        assert "event: done" not in text
        assert sorted(_registry()) == ["work"]

    async def test_the_mcp_result_lists_the_absorbed_source(self, library, notes):
        from lilbee.mcp_server import add

        await _add(notes / "work")

        result = await add([str(notes)])

        assert result["copied"] == ["notes"] and result["absorbed"] == ["work"]
        assert result["overlapping"] == [] and result["errors"] == []
        assert result["sync"]["added"] == ["notes/loose.md"]

    async def test_the_mcp_tool_returns_an_error_while_a_sync_runs(self, library, notes):
        from lilbee.mcp_server import add

        await _add(notes / "work")
        async with sync_running(cfg.data_root):
            result = await add([str(notes)])

        assert result == {
            "error": "A sync or a wiki build is running. Add notes again when it ends."
        }


_UNFINISHED_ADD_ERRORS = {
    UnfinishedAdd.HELD: SyncRunningError,
    UnfinishedAdd.UNREADABLE: AbsorbJournalError,
    UnfinishedAdd.UNLOCKABLE: LibraryUnlockableError,
    UnfinishedAdd.NO_LOCKING: LockingUnsupportedError,
}
_SEEDED_KEYS = ["work/budget.md", "work/plan.md", "workshop/x.md"]


def _add_that_died(store: Store, parent: Path) -> None:
    """Leave the journal of an add of *parent* that died before it moved a key."""
    _seeded_child(store, parent)
    with (
        mock.patch.object(Store, "rekey_sources_under", side_effect=_Killed),
        pytest.raises(_Killed),
    ):
        register_sources([parent])
    assert absorb_pending(cfg.data_root)


@pytest.mark.parametrize("cause", list(UnfinishedAdd))
class TestARemoveThatCannotFinishAnAdd:
    """Each surface a remove comes through refuses in its own shape and removes nothing."""

    async def test_the_library_call_raises_the_refusal(self, library, notes, cause):
        from lilbee import Lilbee

        bee = Lilbee(config=cfg, provider=MagicMock())
        _add_that_died(library.store, notes)
        async with unfinished_add_refused(cfg.data_root, cause) as text:
            with pytest.raises(_UNFINISHED_ADD_ERRORS[cause], match=text):
                bee.remove("work/plan.md")

        assert _keys(library) == _SEEDED_KEYS

    async def test_the_cli_prints_one_error_line(self, library, notes, cause):
        from typer.testing import CliRunner

        from lilbee.cli.app import app

        _add_that_died(library.store, notes)
        command = ["remove", "--data-dir", str(cfg.data_root), "work/plan.md"]
        async with unfinished_add_refused(cfg.data_root, cause) as text:
            outcome = CliRunner().invoke(app, command)

        assert outcome.exit_code == 1 and isinstance(outcome.exception, SystemExit)
        printed = " ".join(outcome.output.split())
        assert printed.startswith("Error: ") and printed.count("Error: ") == 1
        assert text in printed and "Traceback" not in printed
        assert _keys(library) == _SEEDED_KEYS

    async def test_the_cli_in_json_mode_prints_one_error_object(self, library, notes, cause):
        from typer.testing import CliRunner

        from lilbee.cli.app import app

        _add_that_died(library.store, notes)
        command = ["--json", "remove", "--data-dir", str(cfg.data_root), "work/plan.md"]
        async with unfinished_add_refused(cfg.data_root, cause) as text:
            outcome = CliRunner().invoke(app, command)

        assert outcome.exit_code == 1 and isinstance(outcome.exception, SystemExit)
        payload = json.loads(outcome.output.strip().splitlines()[-1])
        assert list(payload) == ["error"] and text in payload["error"]
        assert _keys(library) == _SEEDED_KEYS

    async def test_the_mcp_tool_returns_an_error(self, library, notes, cause):
        from lilbee.mcp_server import remove

        _add_that_died(library.store, notes)
        async with unfinished_add_refused(cfg.data_root, cause) as text:
            result = await remove(["work/plan.md"])

        assert list(result) == ["error"] and text in result["error"]
        assert _keys(library) == _SEEDED_KEYS

    async def test_the_http_route_answers_409_with_the_refusal(self, library, notes, cause):
        from litestar.exceptions import ClientException

        from lilbee.server.handlers.documents import delete_documents

        _add_that_died(library.store, notes)
        async with unfinished_add_refused(cfg.data_root, cause) as text:
            with pytest.raises(ClientException) as refused:
                await delete_documents(["work/plan.md"])

        assert refused.value.status_code == 409 and text in refused.value.detail
        assert _keys(library) == _SEEDED_KEYS


class TestARemoveAfterTheLibraryIsFree:
    async def test_the_mcp_tool_finishes_the_add_and_removes_at_the_new_key(self, library, notes):
        from lilbee.mcp_server import remove

        _add_that_died(library.store, notes)
        async with unfinished_add_refused(cfg.data_root, UnfinishedAdd.HELD):
            refused = await remove(["work/plan.md"])

        result = await remove(["notes/work/plan.md"])

        assert list(refused) == ["error"]
        assert result["removed"] == ["notes/work/plan.md"]
        assert _keys(library) == ["notes/work/budget.md", "workshop/x.md"]


@pytest.mark.parametrize("cause", list(UnfinishedAdd))
class TestAnMcpSyncThatCannotFinishAnAdd:
    async def test_the_sync_tool_returns_an_error(self, library, notes, cause):
        from lilbee.mcp_server import sync as mcp_sync

        _add_that_died(library.store, notes)
        async with unfinished_add_refused(cfg.data_root, cause) as text:
            result = await mcp_sync()

        assert list(result) == ["error"] and text in result["error"]
        assert _keys(library) == _SEEDED_KEYS

    async def test_an_add_refused_at_its_registration_returns_an_error(self, library, notes, cause):
        from lilbee.mcp_server import add

        _add_that_died(library.store, notes)
        async with unfinished_add_refused(cfg.data_root, cause):
            result = await add([str(notes / "loose.md")])

        assert list(result) == ["error"]
        assert sorted(_registry()) == ["work"]

    async def test_the_sync_inside_an_add_returns_an_error(self, library, tmp_path, cause):
        """Another process's add dies between this add's registration and its sync."""
        from lilbee.mcp_server import add

        other = tmp_path / "src" / "other"
        _write(other / "page.md", "# Page\n\nA page about the ferry.\n")
        real_register = ingest_mod.register_sources

        def _then_another_add_dies(paths, **kwargs):
            registered = real_register(paths, **kwargs)
            leave_unfinished_add(cfg.data_root, cause)
            return registered

        async with unfinished_add_refused(cfg.data_root, cause, journal=False) as text:
            with mock.patch.object(ingest_mod, "register_sources", _then_another_add_dies):
                result = await add([str(other)])

        assert list(result) == ["error"] and text in result["error"]
        assert _keys(library) == []
        # A data root that cannot lock refuses the add at its registration.
        refused_first = (UnfinishedAdd.NO_LOCKING, UnfinishedAdd.UNLOCKABLE)
        registered = [] if cause in refused_first else ["other"]
        assert sorted(_registry()) == registered


@contextlib.contextmanager
def _data_root_that_cannot_be_locked(cause: UnfinishedAdd):
    with (
        sync_lock_not_a_lock_file(cfg.data_root)
        if cause is UnfinishedAdd.UNLOCKABLE
        else filesystem_without_locking()
    ):
        yield


_CANNOT_BE_LOCKED = [UnfinishedAdd.UNLOCKABLE, UnfinishedAdd.NO_LOCKING]


@pytest.mark.parametrize("cause", _CANNOT_BE_LOCKED)
class TestAnAddThatTakesASourceInOnADataRootItCannotLock:
    """Each surface refuses in its own shape, with a text about the add, and moves nothing."""

    @staticmethod
    def _assert_about_the_add(text: str, cause: UnfinishedAdd) -> None:
        assert UNFINISHED_ADD_REFUSALS[cause] in text
        assert "reset" not in text.lower() and "Traceback" not in text

    @staticmethod
    def _assert_nothing_moved(library) -> None:
        assert _keys(library) == _SEEDED_KEYS
        assert sorted(_registry()) == ["work"] and not absorb_pending(cfg.data_root)

    def test_the_library_call_raises_the_refusal(self, library, notes, cause):
        from lilbee import Lilbee

        bee = Lilbee(config=cfg, provider=MagicMock())
        bee._services = library
        _seeded_child(library.store, notes)
        with (
            _data_root_that_cannot_be_locked(cause),
            pytest.raises(_UNFINISHED_ADD_ERRORS[cause]) as refused,
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
            pytest.raises(_UNFINISHED_ADD_ERRORS[cause]) as refused,
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


async def _renamed_file(parent: Path) -> None:
    """A file with no heading, so its title is the name it is indexed under."""
    _write(parent / "work" / "ferry_memo.md", "A memo about the ferry, with no heading.\n")
    _register_as({"renamed": parent / "work" / "ferry_memo.md"})
    await sync(quiet=True)


async def _nested(parent: Path) -> None:
    _write(parent / "work" / "deep" / "a.md", "# A\n\nA deep page.\n")
    _register_as({"work": parent / "work", "deep": parent / "work" / "deep"})
    await sync(quiet=True)


async def _nested_and_renamed(parent: Path) -> None:
    _write(parent / "work" / "deep" / "a.md", "# A\n\nA deep page.\n")
    _register_as({"outer": parent / "work", "notes": parent / "work" / "deep"})
    await sync(quiet=True)


class TestTheOracle:
    @pytest.mark.parametrize(
        "registered_before",
        [_one_child, _renamed_folder, _renamed_file, _nested, _nested_and_renamed],
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


class TestAPendingDraft:
    async def test_a_pending_draft_is_accepted_after_an_absorb(self, library, notes):
        from lilbee.wiki.drafts import accept_draft
        from lilbee.wiki.shared import parse_frontmatter

        await _add(notes / "work")
        store = library.store
        chunk = store.get_chunks_by_source("work/plan.md")[0]
        excerpt = "The plan for the spring release."
        assert excerpt in chunk.chunk
        from lilbee.wiki.page import assemble_content, build_frontmatter, write_page

        content = assemble_content(
            build_frontmatter(cfg, ["work/plan.md"], 0.2, chunks=[chunk]),
            "# Spring release\n\nThe release has a plan.[^src1]\n",
            f'[^src1]: work/plan.md, excerpt: "{excerpt}"\n',
        )
        wiki_root = cfg.data_root / cfg.wiki_dir
        draft = write_page(
            wiki_root, "drafts", "spring-release", content, 1.0, ["work/plan.md"], "concepts"
        )

        register_sources([notes])

        assert parse_frontmatter(draft.read_text(encoding="utf-8"))["sources"] == [
            "notes/work/plan.md"
        ]
        with mock.patch("lilbee.wiki.drafts.rewrite_links_across_wiki"):
            accepted = accept_draft("spring-release", wiki_root, store, cfg)

        published = accepted.moved_to.read_text(encoding="utf-8")
        assert f'[^src1]: notes/work/plan.md, excerpt: "{excerpt}"' in published
        rows = store.get_citations_for_source("notes/work/plan.md")
        assert [row["wiki_source"] for row in rows] == ["wiki/concepts/spring-release.md"]
        assert store.get_citations_for_source("work/plan.md") == []


class TestSkipRecordFiles:
    def test_all_three_files_move(self, library, notes):
        _seeded_child(library.store, notes)

        register_sources([notes])

        for filename in (SKIP_MARKER_FILENAME, SKIP_REASON_FILENAME, SKIP_KIND_FILENAME):
            stored = json.loads((cfg.data_root / filename).read_text(encoding="utf-8"))
            assert sorted(stored) == ["notes/work/broken.md", "notes/work/gone.md"], filename


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


class TestRekeyWikiPages:
    _PAGE = (
        "---\n"
        "generated_by: m\n"
        'sources: ["work/a.md", "workshop/x.md", "plan.md"]\n'
        "faithfulness_score: 0.90\n"
        "provenance:\n"
        "  extraction_method: ner_entities\n"
        "  chunks:\n"
        "  - source: work/a.md\n"
        "    chunk_index: 0\n"
        "  - source: workshop/x.md\n"
        "    chunk_index: 1\n"
        "---\n"
        "\n"
        "# Title\n"
        "\n"
        "A claim.[^src1] Another.[^src2] The words work/a.md stay in prose.\n"
        "\n"
        "```\n"
        "[^src9]: work/a.md, an example in a fence\n"
        "```\n"
        "\n"
        "---\n"
        "<!-- citations (auto-generated from _citations table -- do not edit) -->\n"
        '[^src1]: work/a.md, lines 1-2, excerpt: "work/a.md says so"\n'
        "[^src2]: workshop/x.md, page 3\n"
        "[^src3]: plan.md, lines 4-5\n"
        "[^src4]: plan.md\n"
        "[^src5]: see work/a.md\n"
    )

    def test_a_folder_key_moves_in_frontmatter_provenance_and_footnotes_only(self):
        from lilbee.wiki.rekey import rekeyed_page

        moved = rekeyed_page(self._PAGE, "work", "notes/work")

        assert 'sources: ["notes/work/a.md", "plan.md", "workshop/x.md"]' in moved
        assert "  - source: notes/work/a.md\n    chunk_index: 0\n" in moved
        assert "  - source: workshop/x.md\n    chunk_index: 1\n" in moved
        assert '[^src1]: notes/work/a.md, lines 1-2, excerpt: "work/a.md says so"' in moved
        assert "[^src2]: workshop/x.md, page 3" in moved
        assert "[^src9]: work/a.md, an example in a fence" in moved
        assert "[^src5]: see work/a.md" in moved
        assert "The words work/a.md stay in prose." in moved
        assert moved.count("notes/work/a.md") == 3

    def test_a_single_file_key_moves_alone_or_before_its_location(self):
        from lilbee.wiki.rekey import rekeyed_page

        moved = rekeyed_page(self._PAGE, "plan.md", "notes/work/plan.md")

        assert "[^src3]: notes/work/plan.md, lines 4-5" in moved
        assert "[^src4]: notes/work/plan.md\n" in moved
        assert '"notes/work/plan.md"' in moved

    def test_a_page_that_names_no_moved_key_is_returned_as_it_is(self):
        from lilbee.wiki.rekey import rekeyed_page

        assert rekeyed_page(self._PAGE, "other", "notes/other") == self._PAGE
        assert rekeyed_page("# No frontmatter\n\nwork/a.md\n", "work", "n/work") == (
            "# No frontmatter\n\nwork/a.md\n"
        )

    @pytest.mark.parametrize(
        "frontmatter",
        [
            "sources: [not json\n",
            'sources: "work/a.md"\n',
            "sources: [1, 2]\n",
            "provenance:\n  chunks: {unclosed\n",
            "provenance:\n  chunks:\n  - 3\n",
            "provenance:\n  other: 1\n",
        ],
        ids=["not-json", "a-string", "numbers", "bad-yaml", "chunk-not-a-table", "no-chunks"],
    )
    def test_hand_edited_frontmatter_is_left_as_it_is(self, frontmatter):
        from lilbee.wiki.rekey import rekeyed_page

        page = f"---\n{frontmatter}---\n\n# T\n"

        assert rekeyed_page(page, "work", "notes/work") == page

    def test_pages_in_every_folder_are_written_and_the_rest_are_not_touched(self, tmp_path):
        from lilbee.wiki.rekey import rekey_wiki_pages

        root = tmp_path / "wiki"
        moved = [
            _write(root / "entities" / "a.md", self._PAGE),
            _write(root / "drafts" / "b.md", "<!-- origin: concepts -->\n\n" + self._PAGE),
            _write(root / "archive" / "concepts" / "c.md", self._PAGE),
        ]
        untouched = _write(root / "concepts" / "d.md", "# D\n\nNo sources here.\n")
        stamp = untouched.stat().st_mtime_ns
        (root / "summaries").mkdir()
        (root / "summaries" / "bytes.md").write_bytes(b"\xff\xfe not utf-8")

        written = rekey_wiki_pages(root, "work", "notes/work")

        assert written == sorted(moved)
        assert all('"notes/work/a.md"' in page.read_text(encoding="utf-8") for page in moved)
        assert untouched.stat().st_mtime_ns == stamp
        assert (root / "summaries" / "bytes.md").read_bytes() == b"\xff\xfe not utf-8"
        assert rekey_wiki_pages(root, "work", "notes/work") == []
        assert rekey_wiki_pages(tmp_path / "no-wiki", "work", "notes/work") == []


def _page_from(store: Store, keys: list[str]) -> str:
    """The text of a generated page about the spring release, written from *keys*."""
    from lilbee.wiki.page import assemble_content, build_frontmatter

    chunks = [chunk for key in keys for chunk in store.get_chunks_by_source(key)]
    frontmatter = build_frontmatter(cfg, keys, 0.9, chunks=chunks)
    return assemble_content(frontmatter, "# Spring\n\nA release.[^src1]\n", f"[^src1]: {keys[0]}\n")


class TestDraftMarkers:
    @staticmethod
    def _drafts() -> Path:
        return cfg.data_root / cfg.wiki_dir / "drafts"

    @staticmethod
    def _collision(store: Store, drafts: Path, first: str, held: list[str]) -> Path:
        from lilbee.wiki.persistence import divert_concept_collision

        return divert_concept_collision(
            slug="spring",
            source=", ".join(sorted(held)),
            first_source=first,
            content=_page_from(store, held),
            drafts_dir=drafts,
            origin_subdir="concepts",
        )

    @pytest.mark.parametrize(
        ("first", "held", "first_after"),
        [
            ("work/budget.md", ["work/plan.md"], "notes/work/budget.md"),
            ("drafts/spring.md", ["work/budget.md", "work/plan.md"], "drafts/spring.md"),
        ],
    )
    def test_a_collision_draft_is_the_one_its_writer_makes_for_the_new_keys(
        self, library, notes, tmp_path, first, held, first_after
    ):
        """The same two sources that collide again land on this draft, not beside it."""
        from lilbee.wiki.drafts import list_drafts
        from lilbee.wiki.shared import PendingKind

        store = library.store
        _seeded_child(store, notes)
        before = self._collision(store, self._drafts(), first, held)

        register_sources([notes])

        after_keys = [f"notes/{key}" for key in held]
        expected = self._collision(store, tmp_path / "expected", first_after, after_keys)
        assert expected.name != before.name
        assert not before.exists()
        moved = self._drafts() / expected.name
        assert (
            moved.read_text(encoding="utf-8").split("\n")[0]
            == (expected.read_text(encoding="utf-8").split("\n")[0])
        )
        assert f"sources: {json.dumps(after_keys)}" in moved.read_text(encoding="utf-8")
        assert sorted(page.name for page in self._drafts().iterdir()) == [expected.name]
        listed = list_drafts(cfg.data_root / cfg.wiki_dir)
        assert [(draft.slug, draft.pending_kind) for draft in listed] == [
            (expected.stem, PendingKind.COLLISION)
        ]

    def test_a_collision_draft_under_another_name_keeps_that_name(self, library, notes):
        from lilbee.wiki.rekey import rekey_wiki_pages

        store = library.store
        _seeded_child(store, notes)
        written = self._collision(store, self._drafts(), "work/budget.md", ["work/plan.md"])
        renamed = written.with_name("spring-collision-00000000.md")
        written.rename(renamed)
        plain = _write(self._drafts() / "autumn-collision-0a0b0c0d.md", _page_from(store, ["x.md"]))

        pages = rekey_wiki_pages(cfg.data_root / cfg.wiki_dir, "work", "notes/work")

        assert renamed in pages and plain not in pages
        assert "content from notes/work/plan.md held" in renamed.read_text(encoding="utf-8")

    def test_a_kill_between_the_new_draft_and_the_old_one_leaves_one_draft(self, library, notes):
        from lilbee.wiki.rekey import rekey_wiki_pages

        store = library.store
        _seeded_child(store, notes)
        before = self._collision(store, self._drafts(), "work/budget.md", ["work/plan.md"])
        wiki_root = cfg.data_root / cfg.wiki_dir

        with mock.patch.object(Path, "unlink", side_effect=_Killed), pytest.raises(_Killed):
            rekey_wiki_pages(wiki_root, "work", "notes/work")
        assert before.exists() and len(list(self._drafts().iterdir())) == 2
        rekey_wiki_pages(wiki_root, "work", "notes/work")

        drafts = list(self._drafts().iterdir())
        assert len(drafts) == 1 and drafts[0] != before
        assert "content from notes/work/plan.md held" in drafts[0].read_text(encoding="utf-8")

    def _drift(self, store: Store, keys: list[str]) -> Path:
        from lilbee.wiki.persistence import divert_to_drafts

        return divert_to_drafts(
            _page_from(store, keys), self._drafts(), "spring", 0.5, "diff", "concepts", keys
        )

    def test_the_next_build_takes_a_drift_draft_over_in_place(self, library, notes):
        """The marker keeps the hash of the old keys; the draft is known by its sources list."""
        from lilbee.wiki.persistence import delete_drift_draft_if_present

        store = library.store
        _seeded_child(store, notes)
        draft = self._drift(store, ["work/plan.md"])
        marker = draft.read_text(encoding="utf-8").split("\n")[0]

        register_sources([notes])

        assert draft.read_text(encoding="utf-8").split("\n")[0] == marker
        assert self._drift(store, ["notes/work/plan.md"]) == draft
        assert [page.name for page in self._drafts().iterdir()] == ["spring.md"]
        assert draft.read_text(encoding="utf-8").split("\n")[0] != marker
        assert delete_drift_draft_if_present(self._drafts(), "spring", ["notes/work/plan.md"])

    def test_a_drift_draft_whose_sources_list_kept_the_old_keys_is_not_taken_over(
        self, library, notes
    ):
        """The control for the test above: without the re-key of the page, a second draft."""
        store = library.store
        _seeded_child(store, notes)
        draft = self._drift(store, ["work/plan.md"])

        with mock.patch("lilbee.wiki.rekey.rekey_wiki_pages"):
            register_sources([notes])

        assert self._drift(store, ["notes/work/plan.md"]) != draft
        assert len(list(self._drafts().iterdir())) == 2


class TestSyncsHeldOff:
    async def test_it_raises_the_text_it_was_given_while_a_sync_runs(self, tmp_path):
        from lilbee.runtime.lock import _syncs_held_off

        async with sync_running(tmp_path):
            with (
                pytest.raises(SyncRunningError, match=r"^try later$"),
                _syncs_held_off(tmp_path, "try later"),
            ):
                pytest.fail("entered while a sync ran")
        with _syncs_held_off(tmp_path, "try later"):
            pass

    def test_a_sync_in_another_process_waits_until_the_block_ends(self, tmp_path):
        import subprocess
        import sys

        from lilbee.runtime.lock import _syncs_held_off

        script = (
            "import asyncio, sys\n"
            "from pathlib import Path\n"
            "from lilbee.runtime.lock import sync_running\n"
            "async def main():\n"
            "    async with sync_running(Path(sys.argv[1])):\n"
            "        print('IN', flush=True)\n"
            "asyncio.run(main())\n"
        )
        with _syncs_held_off(tmp_path, "busy"):
            child = subprocess.Popen(
                [sys.executable, "-c", script, str(tmp_path)],
                stdout=subprocess.PIPE,
                encoding="utf-8",
            )
            with pytest.raises(subprocess.TimeoutExpired):
                child.wait(timeout=3)
        assert child.communicate(timeout=30)[0].strip() == "IN"
        assert child.returncode == 0

    def test_with_a_wait_it_takes_the_lock_when_the_sync_ends(self, tmp_path):
        import threading

        from lilbee.runtime.lock import _syncs_held_off

        held = threading.Event()
        release = threading.Event()

        def _a_sync() -> None:
            with source_keys_in_use(tmp_path):
                held.set()
                release.wait(10)

        thread = threading.Thread(target=_a_sync)
        thread.start()
        assert held.wait(10)
        with (
            pytest.raises(SyncRunningError, match="busy"),
            _syncs_held_off(tmp_path, "busy", wait=0.2),
        ):
            pytest.fail("entered while a sync ran")
        threading.Timer(0.3, release.set).start()
        with _syncs_held_off(tmp_path, "busy", wait=10):
            assert release.is_set()
        thread.join(10)

    def test_a_lock_that_cannot_be_taken_refuses_the_block(self, tmp_path):
        from lilbee.runtime.lock import _syncs_held_off

        with (
            _sync_lock_take_refused(),
            pytest.raises(LibraryUnlockableError, match="disk I/O error") as refused,
            _syncs_held_off(tmp_path, "busy"),
        ):
            pytest.fail("entered without the lock")

        assert f"delete {tmp_path / 'sync.lock'}, and try again." in str(refused.value)
        assert "reset" not in str(refused.value)
        with _syncs_held_off(tmp_path, "busy"):
            pass

    def test_a_data_root_that_cannot_be_made_refuses_the_block_as_an_add(self, tmp_path):
        from lilbee.runtime.lock import _syncs_held_off

        not_a_folder = tmp_path / "a-file"
        not_a_folder.write_text("in the way", encoding="utf-8")
        data_root = not_a_folder / "library"

        with (
            pytest.raises(LibraryUnlockableError, match="cannot be locked") as refused,
            _syncs_held_off(data_root, "busy"),
        ):
            pytest.fail("entered without the lock")

        assert str(refused.value).startswith(
            f"Cannot tell whether a sync or import is running: {data_root} cannot be locked ("
        )
        assert not isinstance(refused.value, ResetRefusedError)


@contextlib.contextmanager
def _sync_lock_take_refused():
    """Fail every take of a sync lock as a disk that refuses SQLite does; a probe works."""
    real = {
        "acquire_read": ReadWriteLock.acquire_read,
        "acquire_write": ReadWriteLock.acquire_write,
    }

    def _refusing(name: str):
        def _take(lock: ReadWriteLock, *args, **kwargs):
            if Path(lock.lock_file).name.startswith("sync.probe."):
                return real[name](lock, *args, **kwargs)
            raise sqlite3.OperationalError("disk I/O error")

        return _take

    with (
        mock.patch.object(ReadWriteLock, "acquire_read", _refusing("acquire_read")),
        mock.patch.object(ReadWriteLock, "acquire_write", _refusing("acquire_write")),
    ):
        yield


def _cannot_change(data_root: Path) -> str:
    """What a sync or a wiki write says when a disk error stops it at the sync lock."""
    return (
        f"Cannot change the library: {data_root / 'sync.lock'} cannot be locked "
        "(disk I/O error). Fix that problem, then try again."
    )


class TestSourceKeysInUse:
    def test_a_lock_that_cannot_be_taken_refuses_the_block(self, tmp_path, caplog):
        ran = []
        with (
            _sync_lock_take_refused(),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
            pytest.raises(LibraryUnlockableError) as refused,
            source_keys_in_use(tmp_path),
        ):
            ran.append("block")

        assert ran == []
        assert str(refused.value) == _cannot_change(tmp_path)
        assert caplog.text == ""
        with _no_sync_running(tmp_path):
            pass

    def test_a_lock_file_that_cannot_be_opened_refuses_the_block(self, tmp_path, caplog):
        def _probe_only(path, *args, **kwargs):
            if Path(path).name.startswith("sync.probe."):
                return ReadWriteLock(path, *args, **kwargs)
            raise sqlite3.OperationalError("disk I/O error")

        ran = []
        with (
            mock.patch("lilbee.runtime.lock.ReadWriteLock", _probe_only),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
            pytest.raises(LibraryUnlockableError) as refused,
            source_keys_in_use(tmp_path),
        ):
            ran.append("block")

        assert ran == []
        assert str(refused.value) == _cannot_change(tmp_path)
        assert caplog.text == ""


def _record_of(key: str) -> tuple[str | None, SkipKind | None, str | None]:
    """The hash, kind and reason the skip record files hold for *key*."""
    return (
        load_skip_markers(cfg.data_root).get(key),
        load_skip_kinds(cfg.data_root).get(key),
        load_skip_reasons(cfg.data_root).get(key),
    )


class TestRecordsChangeTogether:
    """A kill between the skip record files never splits the record of a file."""

    _REMOVED = ("hash of work/gone.md", SkipKind.REMOVED, REMOVED_SKIP_REASON)

    @staticmethod
    def _dies_replacing(filename: str):
        """Die before the first replace of *filename*, as a killed process does."""
        import os

        real = os.replace

        def _replace(src, dst, *args, **kwargs):
            if Path(dst).name == filename:
                raise _Killed
            return real(src, dst, *args, **kwargs)

        return mock.patch.object(os, "replace", _replace)

    @pytest.mark.parametrize(
        "filename", [SKIP_MARKER_FILENAME, SKIP_REASON_FILENAME, SKIP_KIND_FILENAME]
    )
    def test_a_kill_before_each_record_file_leaves_the_removal_a_removal(
        self, library, notes, filename
    ):
        _seeded_child(library.store, notes)
        assert _record_of("work/gone.md") == self._REMOVED

        with self._dies_replacing(filename), pytest.raises(_Killed):
            register_sources([notes])
        cfg.linked_roots = {"work": str((notes / "work").resolve())}  # what a new process loads
        finish_pending_absorb()

        assert _record_of("notes/work/gone.md") == self._REMOVED
        assert _record_of("notes/work/broken.md") == (
            "hash of work/broken.md",
            SkipKind.FAILED,
            "no text",
        )
        assert sorted(load_skip_markers(cfg.data_root)) == [
            "notes/work/broken.md",
            "notes/work/gone.md",
        ]
        for name in (SKIP_MARKER_FILENAME, SKIP_REASON_FILENAME, SKIP_KIND_FILENAME):
            stored = json.loads((cfg.data_root / name).read_text(encoding="utf-8"))
            assert sorted(stored) == ["notes/work/broken.md", "notes/work/gone.md"], name
        assert clear_failed_markers(cfg.data_root) == ["notes/work/broken.md"]
        assert _record_of("notes/work/gone.md") == self._REMOVED

    def test_a_marker_with_no_reason_moves_as_a_failure_without_one(self, library, notes):
        from lilbee.data.ingest.skip_marker import write_skip_markers

        _seeded_child(library.store, notes)
        write_skip_markers(cfg.data_root, {**load_skip_markers(cfg.data_root), "work/bare.md": "h"})

        register_sources([notes])

        assert _record_of("notes/work/bare.md") == ("h", SkipKind.FAILED, None)

    def test_the_records_stay_at_the_old_keys_until_the_keys_have_moved(self, library, notes):
        """The journal holds them from its first write; the files change after the keys."""
        _seeded_child(library.store, notes)

        with (
            mock.patch.object(absorb_mod, "_write_records", side_effect=_Killed),
            pytest.raises(_Killed),
        ):
            register_sources([notes])

        assert _record_of("work/gone.md") == self._REMOVED
        assert "notes/work/plan.md" in _keys(library)
        journal = read_journal(cfg.data_root)
        assert journal is not None
        assert journal.records["notes/work/gone.md"] == HeldOut(
            "hash of work/gone.md", REMOVED_SKIP_REASON, removed=True
        )


class TestTheAddThatFinishesAnAbsorb:
    def _killed_absorb(self, library, notes) -> None:
        _seeded_child(library.store, notes)
        with mock.patch.object(absorb_mod, "_land", side_effect=_Killed), pytest.raises(_Killed):
            register_sources([notes])
        assert absorb_pending(cfg.data_root)
        cfg.linked_roots = {"work": str((notes / "work").resolve())}

    def test_adding_the_parent_again_keeps_what_the_absorb_took_in(self, library, notes):
        self._killed_absorb(library, notes)

        result = register_sources([notes])

        assert result.tracked == ["notes"] and not absorb_pending(cfg.data_root)
        assert _record_of("notes/work/gone.md")[1] is SkipKind.REMOVED
        assert _record_of("notes/work/broken.md")[1] is SkipKind.FAILED

    def test_an_add_that_waits_while_another_process_finishes_keeps_them(self, library, notes):
        from contextlib import contextmanager

        self._killed_absorb(library, notes)
        real_lock = ingest_mod.skip_records_lock

        @contextmanager
        def _the_other_process_finishes_first(data_root, cancel):
            with library_held(cfg, "a sync is running") as hold:
                absorb_mod.finish_absorb(library.store, hold)
            with real_lock(data_root, cancel):
                yield

        with mock.patch.object(ingest_mod, "skip_records_lock", _the_other_process_finishes_first):
            result = register_sources([notes])

        assert result.tracked == ["notes"] and not absorb_pending(cfg.data_root)
        assert _record_of("notes/work/gone.md")[1] is SkipKind.REMOVED
        assert _record_of("notes/work/broken.md")[1] is SkipKind.FAILED

    def test_a_journal_left_after_the_add_arrived_is_finished_and_kept(self, library, notes):
        """The other process died while this add waited for the lock."""
        self._killed_absorb(library, notes)

        with mock.patch.object(ingest_mod, "keys_of_pending_absorb", return_value=[]):
            result = register_sources([notes])

        assert result.tracked == ["notes"] and not absorb_pending(cfg.data_root)
        assert _record_of("notes/work/gone.md")[1] is SkipKind.REMOVED

    def test_a_journal_that_ends_before_it_is_read_names_no_key(self, library, notes):
        self._killed_absorb(library, notes)

        with mock.patch.object(absorb_mod, "read_journal", return_value=None):
            assert absorb_mod.keys_of_pending_absorb() == []

    def test_an_add_of_the_tracked_parent_after_the_absorb_drops_them(self, library, notes):
        """The rule every add of a tracked folder follows, with no absorb to finish."""
        _seeded_child(library.store, notes)
        register_sources([notes])
        assert _record_of("notes/work/gone.md")[1] is SkipKind.REMOVED
        _write(notes / "work" / "gone.md", "# Gone\n")

        result = register_sources([notes])

        assert result.tracked == ["notes"]
        assert load_skip_markers(cfg.data_root) == {}


class TestWrittenBetweenThePhases:
    def _killed_before_the_land(self, library, notes) -> AbsorbJournal:
        _seeded_child(library.store, notes)
        with mock.patch.object(absorb_mod, "_land", side_effect=_Killed), pytest.raises(_Killed):
            register_sources([notes])
        journal = read_journal(cfg.data_root)
        assert journal is not None
        return journal

    def test_the_journals_records_alone_reach_the_record_files(self, library, notes):
        self._killed_before_the_land(library, notes)
        _hold_out("work/since.md", SkipKind.REMOVED)

        finish_pending_absorb()

        assert load_skip_markers(cfg.data_root) == {
            "notes/work/broken.md": "hash of work/broken.md",
            "notes/work/gone.md": "hash of work/gone.md",
        }

    def test_a_row_written_below_the_old_key_is_left_there(self, library, notes):
        store = library.store
        self._killed_before_the_land(library, notes)
        _seed(store, "work/since.md")

        finish_pending_absorb()

        assert _holders(store, "work/since.md") == _KEY_COLUMNS
        assert _holders(store, "notes/work/since.md") == set()

    def test_a_removed_file_that_is_indexed_again_stays_indexed(self, library, notes):
        from lilbee.data.store import SourceType

        store = library.store
        journal = self._killed_before_the_land(library, notes)
        again = f"{journal.lifted('work')}/gone.md"
        store.upsert_source(again, "hash of work/gone.md", 1, SourceType.DOCUMENT)

        finish_pending_absorb()

        assert "notes/work/gone.md" in _keys(library)

    def test_a_rekey_replaces_no_row_at_the_key_it_takes(self, library):
        store = library.store
        _seed(store, "old/x.md")
        _seed(store, "new/x.md")

        store.rekey_sources_under("old", "new")

        assert len(store.get_chunks_by_source("new/x.md")) == 2

    def test_a_subject_listed_under_a_key_of_the_lift_takes_the_final_key(self, library, notes):
        """A wiki build of a process that was running aggregates the index from the lifted rows."""
        from lilbee.wiki.stubs import load_stub_index

        journal = self._killed_before_the_land(library, notes)
        _save_stub_index(library.store)
        assert f"{journal.lifted('work')}/plan.md" in load_stub_index(cfg)["boeing"].sources

        finish_pending_absorb()

        assert load_stub_index(cfg)["boeing"].sources == (
            "notes/work/budget.md",
            "notes/work/plan.md",
            "workshop/x.md",
        )


@contextlib.contextmanager
def _add_held_on_a_thread(data_root: Path):
    """Hold syncs off *data_root* on a thread, as an add that moves keys does, for the block."""
    import threading

    from lilbee.runtime.lock import _syncs_held_off

    held, release = threading.Event(), threading.Event()

    def _an_add() -> None:
        with _syncs_held_off(data_root, "held by an add"):
            held.set()
            release.wait(30)

    adder = threading.Thread(target=_an_add)
    adder.start()
    try:
        assert held.wait(5)
        yield
    finally:
        release.set()
        adder.join(timeout=5)


_HOLDERS = {
    "an add": _add_held_on_a_thread,
    "a reset": reset_held_on_a_thread,
    "another process": _reset_held_in_another_process,
}
# How long a test waits for a wiki writer to reach a step; reaching it fails the test.
_WIKI_FAILURE_TIMEOUT_S = 10.0


class TestAWikiWriterCancelledWhileItWaitsForTheMark:
    """One cancel ends the wait for an add, a reset or another process; nothing is read."""

    @staticmethod
    def _writers(body: MagicMock) -> dict[str, tuple[Callable[..., object], object]]:
        """Each writer with its body replaced by *body*, and what it returns for a cancel."""
        from lilbee.wiki import generation, lazy

        empty_stats = generation.BuildStats().as_dict()

        def _build(cancel) -> object:
            with mock.patch.object(generation, "get_entity_extractor", body):
                return generation.run_full_build(cfg, cancel=cancel)

        def _synthesis(cancel) -> object:
            with mock.patch.object(generation, "generate_synthesis_pages", body):
                return generation.run_full_synthesize(cfg, cancel=cancel)

        def _page(cancel) -> object:
            with mock.patch.object(lazy, "load_stub_index", body):
                return lazy.generate_stub_page("boeing", MagicMock(), cfg, cancel=cancel)

        return {
            "build": (_build, {"paths": [], "entities": 0, "count": 0, "stats": empty_stats}),
            "synthesis": (_synthesis, {"paths": [], "count": 0, "stats": empty_stats}),
            "page": (_page, None),
        }

    def _cancel_while_held(self, writer: str, *, close_fails: bool = False) -> None:
        """Start *writer* against a held mark, cancel it once, and wait until it returns.

        The caller holds the mark until this returns, so the writer returned while it was held.
        """
        import threading

        body = MagicMock()
        run, cancelled_result = self._writers(body)[writer]
        cancel = threading.Event()
        outcome: list[object] = []
        takes: list[int] = []
        closes: list[int] = []
        found_the_mark_held = threading.Event()
        real_take, real_close = ReadWriteLock.acquire_read, ReadWriteLock.close

        def _take(lock: ReadWriteLock, *args, **kwargs):
            watched = (
                threading.current_thread() is thread and Path(lock.lock_file).name == "sync.lock"
            )
            if watched:
                takes.append(threading.get_ident())
            try:
                return real_take(lock, *args, **kwargs)
            except FileLockTimeout:
                if watched:
                    found_the_mark_held.set()
                raise

        def _close(lock: ReadWriteLock) -> None:
            real_close(lock)
            if threading.current_thread() is thread and Path(lock.lock_file).name == "sync.lock":
                closes.append(threading.get_ident())
                if close_fails:
                    raise sqlite3.OperationalError("disk I/O error")

        def _run() -> None:
            try:
                outcome.append(run(cancel))
            except BaseException as exc:
                outcome.append(exc)

        thread = threading.Thread(target=_run)
        with (
            mock.patch.object(ReadWriteLock, "acquire_read", _take),
            mock.patch.object(ReadWriteLock, "close", _close),
        ):
            thread.start()
            waited = found_the_mark_held.wait(_WIKI_FAILURE_TIMEOUT_S)
            cancel.set()
            thread.join(_WIKI_FAILURE_TIMEOUT_S)
            returned = not thread.is_alive()
        assert waited, "the writer did not wait for the holder"
        assert returned, "one cancel did not end the wait for the holder"
        assert outcome == [cancelled_result]
        body.assert_not_called()
        assert set(takes) == set(closes) == {thread.ident}

    @staticmethod
    def _assert_still_held() -> None:
        """The holder has not released: a reset of the data root is refused."""
        with (
            pytest.raises(ResetRefusedError, match="is running on this library"),
            _no_sync_running(cfg.data_root),
        ):
            pytest.fail("the holder released before the writer returned")

    @pytest.mark.parametrize("holder", list(_HOLDERS))
    @pytest.mark.parametrize("writer", ["build", "synthesis", "page"])
    def test_it_returns_while_the_holder_holds_and_leaves_no_lock(self, library, writer, holder):
        import threading

        cfg.data_root.mkdir(parents=True, exist_ok=True)
        before = set(threading.enumerate())
        with _HOLDERS[holder](cfg.data_root):
            self._cancel_while_held(writer)
            self._assert_still_held()

        assert set(threading.enumerate()) - before == set()
        with _no_sync_running(cfg.data_root):
            pass

    @pytest.mark.parametrize("writer", ["build", "synthesis", "page"])
    def test_a_close_error_does_not_replace_the_cancel(self, library, writer, caplog):
        cfg.data_root.mkdir(parents=True, exist_ok=True)
        with (
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
            reset_held_on_a_thread(cfg.data_root, ends_after_s=3 * _WIKI_FAILURE_TIMEOUT_S),
        ):
            self._cancel_while_held(writer, close_fails=True)
            self._assert_still_held()

        assert f"Could not close {cfg.data_root / 'sync.lock'} (disk I/O error)." in caplog.text
        with _no_sync_running(cfg.data_root):
            pass

    @pytest.mark.parametrize("writer", ["build", "synthesis", "page"])
    def test_a_take_that_fails_as_the_cancel_arrives_does_not_run_the_writer(self, library, writer):
        import threading

        cancel = threading.Event()
        body = MagicMock()
        run, cancelled_result = self._writers(body)[writer]
        real_take = ReadWriteLock.acquire_read

        def _take(lock: ReadWriteLock, *args, **kwargs):
            if Path(lock.lock_file).name != "sync.lock":
                return real_take(lock, *args, **kwargs)
            cancel.set()
            raise sqlite3.OperationalError("disk I/O error")

        with mock.patch.object(ReadWriteLock, "acquire_read", _take):
            assert run(cancel) == cancelled_result

        body.assert_not_called()
        with _no_sync_running(cfg.data_root):
            pass

    @pytest.mark.parametrize("cancel_set", ["before a free take", "at the grant", "after a wait"])
    @pytest.mark.parametrize("writer", ["build", "synthesis", "page"])
    def test_a_writer_granted_the_mark_with_its_cancel_set_writes_nothing(
        self, library, writer, cancel_set
    ):
        import threading

        cancel = threading.Event()
        body = MagicMock()
        run, cancelled_result = self._writers(body)[writer]
        outcome: list[object] = []
        cfg.data_root.mkdir(parents=True, exist_ok=True)
        if cancel_set == "before a free take":
            cancel.set()
        with cancel_set_when_the_sync_lock_is_granted(cancel) as grants:
            if cancel_set == "after a wait":
                with (
                    reset_held_on_a_thread(cfg.data_root),
                    _sync_take_that_ends_busy() as found_the_mark_held,
                ):
                    thread = threading.Thread(target=lambda: outcome.append(run(cancel)))
                    thread.start()
                    assert found_the_mark_held.wait(_WIKI_FAILURE_TIMEOUT_S)
                    assert not cancel.is_set()
                # A guard against a hang; the outcome below is what the test asserts.
                thread.join(_WIKI_FAILURE_TIMEOUT_S)
            else:
                outcome.append(run(cancel))

        if cancel_set != "before a free take":
            assert grants == ["granted"], "the writer never held the mark"
        assert outcome == [cancelled_result]
        body.assert_not_called()
        wiki_root = cfg.data_root / cfg.wiki_dir
        assert not (wiki_root / "index.md").exists()
        assert not (wiki_root / "log.md").exists()
        with _no_sync_running(cfg.data_root):
            pass

    @pytest.mark.parametrize("block_error", [None, ValueError("the block failed")])
    def test_a_close_error_is_logged_and_keeps_the_outcome_of_the_block(
        self, library, tmp_path, caplog, block_error
    ):
        def _close(lock: ReadWriteLock) -> None:
            real_close(lock)
            if Path(lock.lock_file).name == "sync.lock":
                raise sqlite3.OperationalError("disk I/O error")

        def _block() -> str:
            with source_keys_in_use(tmp_path):
                if block_error is not None:
                    raise block_error
                return "finished"

        real_close = ReadWriteLock.close
        with (
            mock.patch.object(ReadWriteLock, "close", _close),
            caplog.at_level(logging.WARNING, logger="lilbee.runtime.lock"),
        ):
            if block_error is None:
                assert _block() == "finished"
            else:
                with pytest.raises(ValueError, match="the block failed"):
                    _block()

        assert f"Could not close {tmp_path / 'sync.lock'} (disk I/O error)." in caplog.text
        with _no_sync_running(tmp_path):
            pass

    def test_a_take_error_without_a_cancel_refuses_the_block(self, library, tmp_path):
        import threading

        ran = []
        with (
            _sync_lock_take_refused(),
            pytest.raises(LibraryUnlockableError) as refused,
            source_keys_in_use(tmp_path, threading.Event()),
        ):
            ran.append("block")

        assert ran == []
        assert str(refused.value) == _cannot_change(tmp_path)


class TestAWikiBuildAndAnAbsorb:
    """A wiki writer holds the sync mark from its first read of a source key to its last page."""

    @staticmethod
    def _refusal(notes: Path) -> str:
        with pytest.raises(SyncRunningError) as raised:
            register_sources([notes])
        return str(raised.value)

    def _full_build(self, notes: Path, refused: list[str]) -> None:
        from lilbee.wiki import generation

        refuse = self._refusal

        class _Extractor:
            def extract(self, chunks: list) -> list:
                refused.append(refuse(notes))
                return []

        def _build(*_args: object, **_kwargs: object) -> list[Path]:
            refused.append(refuse(notes))
            return []

        with (
            mock.patch.object(generation, "get_entity_extractor", return_value=_Extractor()),
            mock.patch.object(generation, "build_wiki", _build),
        ):
            assert generation.run_full_build(cfg)["count"] == 0

    def _synthesis(self, notes: Path, refused: list[str]) -> None:
        from lilbee.wiki import generation

        def _pages(*_args: object, **_kwargs: object) -> list[Path]:
            refused.append(self._refusal(notes))
            return []

        with mock.patch.object(generation, "generate_synthesis_pages", _pages):
            assert generation.run_full_synthesize(cfg)["count"] == 0

    def _one_page(self, notes: Path, refused: list[str]) -> None:
        from lilbee.wiki import lazy

        def _index(_config: object) -> dict:
            refused.append(self._refusal(notes))
            return {}

        with (
            mock.patch.object(lazy, "load_stub_index", _index),
            pytest.raises(lazy.UnknownStubError, match="no indexed page named 'boeing'"),
        ):
            lazy.generate_stub_page("boeing", svc_mod.get_services().store, cfg)

    @pytest.mark.parametrize(
        ("writer", "reads"),
        [("_full_build", 2), ("_synthesis", 1), ("_one_page", 1)],
    )
    def test_an_absorb_is_refused_while_a_wiki_writer_holds_source_keys(
        self, library, notes, writer, reads
    ):
        _seeded_child(library.store, notes)
        refused: list[str] = []

        getattr(self, writer)(notes, refused)

        text = "A sync or a wiki build is running. Add notes again when it ends."
        assert refused == [text] * reads
        assert sorted(_registry()) == ["work"] and not absorb_pending(cfg.data_root)
        assert register_sources([notes]).absorbed == ["work"]

    @pytest.mark.parametrize("writer", ["_full_build", "_synthesis", "_one_page"])
    def test_a_wiki_writer_takes_the_mark_while_the_wiki_mutex_is_free(
        self, library, notes, writer
    ):
        """An absorb holds the mark and then takes the mutex; the other order would deadlock."""
        import threading
        from contextlib import contextmanager

        from lilbee.runtime import lock
        from lilbee.wiki import generation, lazy
        from lilbee.wiki.shared import WIKI_BUILD_LOCK

        _seeded_child(library.store, notes)
        free: list[bool] = []
        real = lock.source_keys_in_use

        def _try() -> None:
            taken = WIKI_BUILD_LOCK.acquire(blocking=False)
            free.append(taken)
            if taken:
                WIKI_BUILD_LOCK.release()

        @contextmanager
        def _marked(data_root: Path, cancel: object = None):
            other = threading.Thread(target=_try)
            other.start()
            other.join()
            with real(data_root, cancel):
                yield

        with (
            mock.patch.object(generation, "source_keys_in_use", _marked),
            mock.patch.object(lazy, "source_keys_in_use", _marked),
        ):
            getattr(self, writer)(notes, [])

        assert free == [True]
        with WIKI_BUILD_LOCK:  # the control: the probe reads a held mutex as held
            control = threading.Thread(target=_try)
            control.start()
            control.join()
        assert free == [True, False]


class TestTempFilesOfAKilledWriter:
    """What a wiki writer leaves when its process dies before the rename."""

    @staticmethod
    def _left(wiki_root: Path) -> tuple[list[Path], Path]:
        """Temp files of a dead writer beside a page and the subject index, and another tool's."""
        dead = [
            leave_temp_of_a_dead_writer(wiki_root / "entities"),
            leave_temp_of_a_dead_writer(wiki_root),
        ]
        other = wiki_root / "entities" / "boeing.md.tmp"
        other.write_text("another tool's", encoding="utf-8")
        return dead, other

    def test_an_absorb_removes_them_and_leaves_another_tools_file(self, library, notes):
        _seeded_child(library.store, notes)
        dead, other = self._left(cfg.data_root / cfg.wiki_dir)

        assert register_sources([notes]).absorbed == ["work"]

        assert [path for path in dead if path.exists()] == [] and other.exists()

    def test_the_finish_of_an_interrupted_absorb_removes_them(self, library, notes):
        _seeded_child(library.store, notes)
        with mock.patch.object(absorb_mod, "_land", side_effect=_Killed), pytest.raises(_Killed):
            register_sources([notes])
        dead, other = self._left(cfg.data_root / cfg.wiki_dir)

        assert finish_pending_absorb() == ["notes/work"]

        assert [path for path in dead if path.exists()] == [] and other.exists()

    def test_a_build_removes_them_and_leaves_another_tools_file(self, library):
        from lilbee.wiki import generation

        dead, other = self._left(cfg.data_root / cfg.wiki_dir)

        assert generation.build_wiki([], library.provider, library.store, cfg) == []

        assert [path for path in dead if path.exists()] == [] and other.exists()

    def test_an_add_that_takes_nothing_in_leaves_them(self, library, notes):
        """The control: only an absorb or a build walks the wiki for them."""
        dead, other = self._left(cfg.data_root / cfg.wiki_dir)

        assert register_sources([notes]).absorbed == []

        assert [path for path in dead if not path.exists()] == [] and other.exists()


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


class TestRepeatsAfterARekey:
    def test_a_footnote_and_a_chunk_two_keys_shared_are_named_once(self):
        from lilbee.wiki.rekey import rekeyed_page

        page = "\n".join(
            [
                "---",
                'sources: ["a/x.md", "b/x.md"]',
                "provenance:",
                "  chunks:",
                "  - source: a/x.md",
                "    chunk_index: 0",
                "  - source: b/x.md",
                "    chunk_index: 0",
                "  - source: b/x.md",
                "    chunk_index: 1",
                "---",
                "",
                "Body.[^src1]",
                "",
                '[^src1]: a/x.md, excerpt: "one"',
                '[^src1]: b/x.md, excerpt: "one"',
                '[^src2]: b/x.md, excerpt: "two"',
            ]
        )

        moved = rekeyed_page(page, "b", "a")

        assert moved.count("- source: a/x.md") == 2
        assert moved.count('[^src1]: a/x.md, excerpt: "one"') == 1
        assert '[^src2]: a/x.md, excerpt: "two"' in moved

    def test_a_page_the_rekey_does_not_touch_keeps_its_repeats(self):
        from lilbee.wiki.rekey import rekeyed_page

        page = '[^src1]: a/x.md, excerpt: "one"\n[^src1]: a/x.md, excerpt: "one"'

        assert rekeyed_page(page, "b", "c") == page


_RESET_REFUSED = (
    "A sync, an import, an add or a wiki build is running on this library. "
    "Reset again when it finishes."
)
# Only a hang guard: no assertion reads a duration.
_RESET_HANG_GUARD_S = 60.0


def _refusal_of_a_reset_from_another_thread() -> str:
    """The text a reset is refused with, run as work that holds no lock of the caller."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        reset = pool.submit(contextvars.Context().run, perform_reset)
        with pytest.raises(ResetRefusedError) as caught:
            reset.result(timeout=_RESET_HANG_GUARD_S)
    return str(caught.value)


class TestAResetAndAnAbsorb:
    async def test_a_reset_during_an_absorb_is_refused_and_the_absorb_ends(self, library, notes):
        await _add(notes / "work")
        refused: list[str] = []
        real_land = absorb_mod._land

        def _reset_then_land(*args, **kwargs):
            refused.append(_refusal_of_a_reset_from_another_thread())
            return real_land(*args, **kwargs)

        with mock.patch.object(absorb_mod, "_land", side_effect=_reset_then_land):
            register_sources([notes])

        assert refused == [_RESET_REFUSED]
        assert _keys(library) == ["notes/work/budget.md", "notes/work/plan.md"]
        assert _registry() == {"notes": str(notes.resolve())}

    def test_a_reset_while_a_wiki_writer_holds_source_keys_says_the_same(self, library):
        with source_keys_in_use(cfg.data_root), pytest.raises(ResetRefusedError) as refused:
            perform_reset()

        assert str(refused.value) == _RESET_REFUSED

    async def test_a_reset_after_a_killed_absorb_leaves_nothing_to_finish(self, library, notes):
        await _add(notes / "work")
        remove_documents_durably(["work/plan.md"])
        with mock.patch.object(absorb_mod, "_land", side_effect=_Killed), pytest.raises(_Killed):
            register_sources([notes])
        assert absorb_pending(cfg.data_root)

        perform_reset()
        svc_mod.set_services(make_mock_services(store=Store(cfg)))
        result = await sync(quiet=True)

        assert not absorb_pending(cfg.data_root)
        assert result.added == []
        assert _registry() == {}
        assert _keys(svc_mod.get_services()) == []
        assert load_skip_markers(cfg.data_root) == {}

    async def test_a_journal_a_reset_cannot_delete_is_reported(self, library, notes):
        await _add(notes / "work")
        with mock.patch.object(absorb_mod, "_land", side_effect=_Killed), pytest.raises(_Killed):
            register_sources([notes])
        real_unlink = Path.unlink

        def _unlink(path, *args, **kwargs):
            if path == journal_path(cfg.data_root):
                raise PermissionError("Access is denied")
            real_unlink(path, *args, **kwargs)

        with mock.patch.object(Path, "unlink", _unlink):
            result = perform_reset()

        assert result.skipped == [str(journal_path(cfg.data_root))]
        assert _registry() == {}
        assert not cfg.lancedb_dir.exists()


@contextlib.contextmanager
def _records_lock_held_elsewhere():
    """Hold the records lock of the library the way another process does."""
    from filelock import FileLock

    holder = FileLock(str(cfg.data_root / SKIP_MARKER_FILENAME) + ".lock")
    holder.acquire()
    try:
        yield
    finally:
        holder.release()


class _SetOnceArmed:
    """A cancel that reads unset until it is armed, and set from then on."""

    def __init__(self) -> None:
        self.armed = False

    def is_set(self) -> bool:
        return self.armed


class TestACancelAtTheRecordsLock:
    """A cancel set while a call waits for the records lock ends the call with nothing changed."""

    def test_an_add_stops_before_it_registers(self, library, notes):
        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _seeded_child(library.store, notes)
        before = _everything(library.store)
        cancel = _SetOnSecondPoll()

        with _records_lock_held_elsewhere(), pytest.raises(SkipRecordsWaitCancelledError) as raised:
            register_sources([notes], cancel=cancel)

        assert cancel.polls >= 2
        assert raised.value.__cause__ is None
        assert _everything(library.store) == before
        assert cfg.linked_roots == {"work": str((notes / "work").resolve())}

    def test_a_remove_stops_before_it_removes(self, library, notes):
        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _seeded_child(library.store, notes)
        before = _everything(library.store)
        cancel = _SetOnSecondPoll()

        with _records_lock_held_elsewhere(), pytest.raises(SkipRecordsWaitCancelledError) as raised:
            remove_documents_durably(["work/plan.md", "work"], cancel=cancel)

        assert cancel.polls >= 2
        assert raised.value.__cause__ is None
        assert _everything(library.store) == before

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param(lambda notes, cancel: register_sources([notes], cancel=cancel), id="add"),
            pytest.param(
                lambda notes, cancel: remove_documents_durably(
                    ["work/plan.md", "work"], cancel=cancel
                ),
                id="remove",
            ),
        ],
    )
    def test_a_cancel_set_as_the_lock_frees_changes_nothing(self, library, notes, command):
        """The lock is granted after a wait with the cancel set: the command ends as a cancel."""
        from filelock import FileLock

        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _seeded_child(library.store, notes)
        before = _everything(library.store)
        holder = FileLock(str(cfg.data_root / SKIP_MARKER_FILENAME) + ".lock")
        holder.acquire()
        cancel = _SetAsTheLockFrees(holder)

        with pytest.raises(SkipRecordsWaitCancelledError):
            command(notes, cancel)

        # The holder let go before the cancel was set, so the take was granted.
        assert not holder.is_locked
        assert _everything(library.store) == before
        assert cfg.linked_roots == {"work": str((notes / "work").resolve())}
        assert _records_lock_is_free(cfg.data_root)

    @pytest.mark.parametrize("names", [None, ["notes"]], ids=["waits", "asks_again"])
    def test_finishing_an_add_stops_before_it_moves_a_key(self, library, notes, names):
        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _add_that_died(library.store, notes)
        before = _everything(library.store)
        cancel = _SetOnSecondPoll()

        with _records_lock_held_elsewhere(), pytest.raises(SkipRecordsWaitCancelledError):
            _finish_as(names, notes, cancel)

        assert cancel.polls >= 2
        assert _everything(library.store) == before
        assert before["journal"] is True

    async def test_a_sync_stops_before_it_finishes_an_add(self, library, notes):
        import asyncio

        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _add_that_died(library.store, notes)
        before = _everything(library.store)
        cancel = _SetOnSecondPoll()

        with _records_lock_held_elsewhere(), pytest.raises(asyncio.CancelledError) as raised:
            await sync(quiet=True, cancel=cancel)

        assert cancel.polls >= 2
        assert isinstance(raised.value.__cause__, SkipRecordsWaitCancelledError)
        assert _everything(library.store) == before
        assert before["journal"] is True

    async def test_a_sync_that_builds_the_services_stops_before_it_finishes_an_add(
        self, library, notes, monkeypatch, caplog
    ):
        """The first sync of a process builds the services; that finish gets the cancel too."""
        import asyncio

        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        caplog.set_level("INFO", logger="lilbee.app.absorb")
        _add_that_died(library.store, notes)
        before = _everything(library.store)
        svc_mod.set_services(None)
        monkeypatch.setattr(svc_mod._state, "singleton", None)
        built = make_mock_services(store=library.store)
        cfg.worker_pool_eager_start = False
        cancel = _SetOnSecondPoll()

        with (
            mock.patch.object(svc_mod, "build_services", return_value=built),
            mock.patch("lilbee.app.settings.reconcile_embedding_dim"),
            mock.patch("lilbee.modelhub.registry.ModelRegistry"),
            _records_lock_held_elsewhere(),
            pytest.raises(asyncio.CancelledError) as raised,
        ):
            await sync(quiet=True, cancel=cancel)

        assert cancel.polls >= 2
        assert isinstance(raised.value.__cause__, SkipRecordsWaitCancelledError)
        # No take ran without the cancel: none ended in the lock error that the start logs.
        assert "An interrupted add is not finished yet" not in caplog.text
        assert svc_mod.peek_services() is built
        assert _everything(built.store) == before
        assert before["journal"] is True

    async def test_a_retry_stops_before_it_clears_a_failure(self, library, notes):
        import asyncio

        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _seeded_child(library.store, notes)
        before = _everything(library.store)
        cancel = _SetOnSecondPoll()

        with _records_lock_held_elsewhere(), pytest.raises(asyncio.CancelledError) as raised:
            await sync(quiet=True, cancel=cancel, retry_skipped=True)

        assert cancel.polls >= 2
        assert isinstance(raised.value.__cause__, SkipRecordsWaitCancelledError)
        assert _everything(library.store) == before
        assert before["kinds"]["work/broken.md"] is SkipKind.FAILED

    async def test_a_rebuild_stops_before_it_clears_the_records(self, library, notes):
        import asyncio

        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _seeded_child(library.store, notes)
        before = _everything(library.store)
        cancel = _SetOnSecondPoll()

        with _records_lock_held_elsewhere(), pytest.raises(asyncio.CancelledError) as raised:
            await sync(force_rebuild=True, quiet=True, cancel=cancel)

        assert cancel.polls >= 2
        assert isinstance(raised.value.__cause__, SkipRecordsWaitCancelledError)
        after = _everything(library.store)
        assert before["markers"] and after["markers"] == before["markers"]
        assert after["kinds"] == before["kinds"] and after["reasons"] == before["reasons"]

    async def test_a_sync_stops_before_it_writes_its_verdicts(self, library, notes):
        """The lock is taken by another holder once the sync reaches its write of the records."""
        import asyncio

        from lilbee.data.ingest import pipeline
        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        register_sources([notes])
        _hold_out("notes/loose.md", SkipKind.FAILED)
        records = (
            load_skip_markers(cfg.data_root),
            load_skip_reasons(cfg.data_root),
            load_skip_kinds(cfg.data_root),
        )
        cancel = _SetOnceArmed()
        real_update = pipeline.update_skip_records

        def _update_with_the_lock_held(*args):
            holding.enter_context(_records_lock_held_elsewhere())
            cancel.armed = True
            real_update(*args)

        with (
            contextlib.ExitStack() as holding,
            mock.patch.object(pipeline, "update_skip_records", _update_with_the_lock_held),
            pytest.raises(asyncio.CancelledError) as raised,
        ):
            await sync(quiet=True, cancel=cancel)

        assert isinstance(raised.value.__cause__, SkipRecordsWaitCancelledError)
        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md", "notes/work/plan.md"]
        assert records[0] == {"notes/loose.md": "hash of notes/loose.md"}
        assert records == (
            load_skip_markers(cfg.data_root),
            load_skip_reasons(cfg.data_root),
            load_skip_kinds(cfg.data_root),
        )


class TestACancelAtTheWaitForARunningSync:
    """A cancel set while a call waits for a running sync ends the call with nothing changed."""

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param(lambda cancel: finish_pending_absorb(cancel=cancel), id="finish"),
            pytest.param(
                lambda cancel: remove_documents_durably(["work/plan.md", "work"], cancel=cancel),
                id="remove",
            ),
        ],
    )
    def test_a_call_that_finishes_an_add_stops_while_the_sync_still_runs(
        self, library, notes, command
    ):
        import threading

        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _add_that_died(library.store, notes)
        before = _everything(library.store)
        cancel = threading.Event()

        with (
            _mark_held_on_a_thread(cfg.data_root),
            _sync_write_takes(after=lambda _outcome: cancel.set()) as takes,
        ):
            with pytest.raises(SkipRecordsWaitCancelledError) as raised:
                command(cancel)
            takes_of_the_wait = list(takes)
            with pytest.raises(ResetRefusedError), _no_sync_running(cfg.data_root):
                pass

        assert takes_of_the_wait == ["busy"], "the wait tried again after the cancel"
        assert raised.value.__cause__ is None
        assert _everything(library.store) == before
        assert before["journal"] is True
        assert _records_lock_is_free(cfg.data_root)
        assert finish_pending_absorb() == ["notes/work"]

    async def test_a_sync_stops_while_the_other_sync_still_runs(self, library, notes):
        import asyncio
        import threading

        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _add_that_died(library.store, notes)
        before = _everything(library.store)
        cancel = threading.Event()

        with (
            _mark_held_on_a_thread(cfg.data_root),
            _sync_write_takes(after=lambda _outcome: cancel.set()) as takes,
        ):
            with pytest.raises(asyncio.CancelledError) as raised:
                await sync(quiet=True, cancel=cancel)
            takes_of_the_wait = list(takes)
            with pytest.raises(ResetRefusedError), _no_sync_running(cfg.data_root):
                pass

        assert takes_of_the_wait == ["busy"], "the wait tried again after the cancel"
        assert isinstance(raised.value.__cause__, SkipRecordsWaitCancelledError)
        assert _everything(library.store) == before
        assert before["journal"] is True
        with _no_sync_running(cfg.data_root):
            pass

    @pytest.mark.parametrize("names", [None, ["notes"]], ids=["waits", "asks_again"])
    def test_finishing_an_add_granted_the_lock_with_the_cancel_set_moves_no_key(
        self, library, notes, names
    ):
        import threading

        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _add_that_died(library.store, notes)
        before = _everything(library.store)
        cancel = threading.Event()

        with (
            _sync_write_takes(after=lambda _outcome: cancel.set()) as takes,
            pytest.raises(SkipRecordsWaitCancelledError),
        ):
            _finish_as(names, notes, cancel)

        assert takes == ["granted"]
        assert _everything(library.store) == before
        assert before["journal"] is True
        with _no_sync_running(cfg.data_root):
            pass

    def test_an_add_granted_the_lock_with_the_cancel_set_moves_no_key(self, library, notes):
        import threading

        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _seeded_child(library.store, notes)
        before = _everything(library.store)
        cancel = threading.Event()

        with (
            _sync_write_takes(after=lambda _outcome: cancel.set()) as takes,
            pytest.raises(SkipRecordsWaitCancelledError),
        ):
            register_sources([notes], cancel=cancel)

        assert takes == ["granted"]
        assert _everything(library.store) == before
        assert before["journal"] is False
        assert cfg.linked_roots == {"work": str((notes / "work").resolve())}
        assert _records_lock_is_free(cfg.data_root)
        with _no_sync_running(cfg.data_root):
            pass

    def test_an_add_that_finishes_another_add_with_the_cancel_set_moves_no_key(
        self, library, notes
    ):
        import threading

        from lilbee.data.ingest.skip_marker import SkipRecordsWaitCancelledError

        _add_that_died(library.store, notes)
        before = _everything(library.store)
        cancel = threading.Event()

        with (
            _sync_write_takes(after=lambda _outcome: cancel.set()) as takes,
            pytest.raises(SkipRecordsWaitCancelledError),
        ):
            register_sources([notes], cancel=cancel)

        assert takes == ["granted"]
        assert _everything(library.store) == before
        assert before["journal"] is True
        assert _records_lock_is_free(cfg.data_root)

    def test_a_cancel_of_the_sync_lock_that_the_block_raises_is_not_the_cancel_of_the_wait(
        self, library
    ):
        from lilbee.runtime.lock import SyncMarkWaitCancelledError

        with (
            pytest.raises(SyncMarkWaitCancelledError),
            library_held(cfg, "a sync is running"),
        ):
            raise SyncMarkWaitCancelledError

        with _no_sync_running(cfg.data_root):
            pass
