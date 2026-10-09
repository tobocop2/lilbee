"""Adding the parent of a registered source takes the parent and folds the source in."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

import lilbee.app.services as svc_mod
from lilbee.app import absorb as absorb_mod
from lilbee.app.absorb import SYNC_RUNNING_ADD_AGAIN, finish_pending_absorb
from lilbee.app.ingest import (
    forget_unfinished_on_cancel,
    register_sources,
    remove_documents_durably,
)
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.data.ingest import sync
from lilbee.data.ingest.skip_marker import (
    SKIP_KIND_FILENAME,
    SKIP_MARKER_FILENAME,
    SKIP_REASON_FILENAME,
    SkipKind,
    load_skip_kinds,
    load_skip_markers,
    load_skip_reasons,
)
from lilbee.data.store import Store
from lilbee.runtime.absorb_journal import (
    AbsorbJournal,
    AbsorbJournalError,
    AbsorbPhase,
    absorb_pending,
    journal_path,
    read_journal,
    write_journal,
)
from lilbee.runtime.lock import SyncRunningError, sync_running
from tests.conftest import make_mock_services
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

    async def test_a_child_with_a_label_that_is_not_its_folder_name_is_refused(
        self, library, notes
    ):
        settings.set_value(cfg.data_root, "linked_roots", {"renamed": str(notes / "work")})

        result = register_sources([notes])

        assert result.containing == ["notes"] and result.registered == []
        assert sorted(_registry()) == ["renamed"]

    async def test_children_that_lie_inside_each_other_are_refused(self, library, notes):
        _write(notes / "work" / "deep" / "a.md", "# A\n\nA deep page.\n")
        settings.set_value(
            cfg.data_root,
            "linked_roots",
            {"work": str(notes / "work"), "deep": str(notes / "work" / "deep")},
        )

        result = register_sources([notes])

        assert result.containing == ["notes"] and result.registered == []

    async def test_a_child_added_in_the_same_call_is_folded_and_not_listed(self, library, notes):
        result = register_sources([notes / "work", notes])

        assert result.registered == ["notes"]
        assert result.absorbed_into == {"notes": ["work"]}
        assert _registry() == {"notes": str(notes.resolve())}


class TestTheCommonAddPaysNothing:
    def test_no_child_means_no_journal_and_no_lock(self, library, notes):
        opened: list[str] = []
        real_exists = Path.exists

        def _exists(path, **kwargs):
            if path.name == "pending_absorb.json":
                opened.append(path.name)
            return real_exists(path, **kwargs)

        with (
            mock.patch.object(absorb_mod, "syncs_held_off") as held_off,
            mock.patch.object(absorb_mod, "write_journal") as journal,
            mock.patch.object(absorb_mod, "_rekey") as rekey,
            mock.patch("lilbee.app.ingest.IgnoreRules.for_corpus") as rules,
            mock.patch.object(Path, "exists", _exists),
        ):
            result = register_sources([notes])

        assert result.registered == ["notes"] and result.absorbed == []
        assert opened == ["pending_absorb.json"]
        held_off.assert_not_called()
        journal.assert_not_called()
        rekey.assert_not_called()
        rules.assert_not_called()
        assert not (cfg.data_root / "sync.lock").exists()
        assert library.store.get_sources() == []

    async def test_an_absorb_does_take_the_lock_and_write_the_journal(self, library, notes):
        """The control for the test above: the same probes fire when a child is there."""
        await _add(notes / "work")
        real_held_off = absorb_mod.syncs_held_off
        with (
            mock.patch.object(absorb_mod, "syncs_held_off", side_effect=real_held_off) as held_off,
            mock.patch("lilbee.app.ingest.IgnoreRules.for_corpus", wraps=None) as rules,
        ):
            rules.return_value.excludes_entry.return_value = False
            register_sources([notes])

        held_off.assert_called_once()
        rules.assert_called_once()

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

        assert str(refused.value) == "A sync is running. Add notes again when it ends."
        assert _everything(library.store) == before
        assert cfg.linked_roots == {"work": str((notes / "work").resolve())}

    async def test_a_plain_add_goes_through_while_a_sync_runs(self, library, notes):
        async with sync_running(cfg.data_root):
            result = register_sources([notes])

        assert result.registered == ["notes"]

    async def test_a_sync_waits_for_the_absorb_and_then_walks_the_new_registry(
        self, library, notes
    ):
        """A second process that syncs during the absorb cannot start until it ends."""
        import threading

        from lilbee.runtime.lock import _acquire_sync_lock, _release_sync_lock

        await _add(notes / "work")
        waiting = threading.Event()
        states: list[bool] = []

        def _second_sync() -> None:
            waiting.set()
            lock = _acquire_sync_lock(cfg.data_root, write=False)
            states.append(absorb_pending(cfg.data_root))
            _release_sync_lock(lock)

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
        "lift-skip-records",
        "lift-wiki-pages",
        "clear-targets",
        "phase-write",
        "land-start",
        "land-third-table",
        "land-skip-records",
        "land-wiki-pages",
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
        import lilbee.data.store.core as core_mod
        import lilbee.wiki.rekey as wiki_rekey_mod

        tables = len(_KEY_COLUMNS)
        return {
            "journal-written": lambda: self._nth_call(Store, "rekey_sources_under", 1),
            "lift-third-table": lambda: self._nth_call(core_mod, "_rekey_sql", 4),
            "lift-skip-records": lambda: self._nth_call(absorb_mod, "rekey_skip_records", 1),
            "lift-wiki-pages": lambda: self._nth_call(wiki_rekey_mod, "rekey_wiki_pages", 1),
            "clear-targets": lambda: self._nth_call(absorb_mod, "_clear_targets", 1),
            "phase-write": lambda: self._nth_call(absorb_mod, "write_journal", 2),
            "land-start": lambda: self._nth_call(absorb_mod, "_land", 1),
            "land-third-table": lambda: self._nth_call(core_mod, "_rekey_sql", tables + 4),
            "land-skip-records": lambda: self._nth_call(absorb_mod, "rekey_skip_records", 2),
            "land-wiki-pages": lambda: self._nth_call(wiki_rekey_mod, "rekey_wiki_pages", 2),
            "wiki-index": lambda: self._nth_call(absorb_mod, "_refresh_wiki_index", 1),
            "registry": lambda: self._nth_call(absorb_mod, "_write_registry", 1),
            "journal-delete": lambda: self._nth_call(absorb_mod, "delete_journal", 1),
        }[point]()

    def _clean_absorb(self, tmp_path, label="notes", child="work") -> dict[str, object]:
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
        _seeded_child(store, parent, child)
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
        with self._kill_at("lift-skip-records"), pytest.raises(_Killed):
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

    async def test_a_start_while_a_sync_runs_leaves_the_journal_and_warns(
        self, library, notes, caplog
    ):
        from lilbee.app.absorb import finish_pending_absorb_at_start

        await _add(notes / "work")
        with self._kill_at("phase-write"), pytest.raises(_Killed):
            register_sources([notes])

        async with sync_running(cfg.data_root):
            finish_pending_absorb_at_start(cfg, library.store)

        assert absorb_pending(cfg.data_root)
        assert "An interrupted add is not finished yet" in caplog.text

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

    async def test_an_older_build_that_syncs_between_the_phases_is_rolled_forward(
        self, library, notes
    ):
        """An older build ignores the journal and its sync moves the lifted keys back."""
        await _add(notes / "work")
        with self._kill_at("land-start"), pytest.raises(_Killed):
            register_sources([notes])
        journal = read_journal(cfg.data_root)
        assert journal is not None and journal.phase is AbsorbPhase.LAND
        library.store.rekey_sources_under(journal.lifted("work"), "work")
        assert _keys(library) == ["work/budget.md", "work/plan.md"]

        synced = await sync(quiet=True)

        assert _keys(library) == ["notes/loose.md", "notes/work/budget.md", "notes/work/plan.md"]
        assert synced.unchanged == 2

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
        ["", "{not json", "[]", '{"id": "a"}', '{"id": "a", "moves": [], "add": {}, "drop": []}'],
        ids=["empty", "not-json", "a-list", "keys-missing", "moves-not-a-table"],
    )
    def test_an_unreadable_journal_raises_and_names_the_file(self, tmp_path, text):
        journal_path(tmp_path).write_text(text, encoding="utf-8")

        with pytest.raises(AbsorbJournalError, match=r"pending_absorb\.json"):
            read_journal(tmp_path)

    def test_an_unknown_phase_raises(self, tmp_path):
        journal_path(tmp_path).write_text(
            json.dumps({"id": "a", "moves": {}, "add": {}, "drop": [], "phase": "undo"}),
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

    async def test_the_cli_refusal_is_one_error_line(self, library, notes):
        from typer.testing import CliRunner

        from lilbee.cli.app import app

        await _add(notes / "work")
        async with sync_running(cfg.data_root):
            outcome = CliRunner().invoke(app, ["add", "--data-dir", str(cfg.data_root), str(notes)])

        assert outcome.exit_code == 1
        assert "A sync is running. Add notes again when it ends." in outcome.output
        assert sorted(_registry()) == ["work"]

    async def test_the_tui_toast_names_the_absorbed_source(self, library, notes):
        from lilbee.cli.tui import messages as msg
        from lilbee.cli.tui.screens.chat import ChatScreen
        from lilbee.cli.tui.widgets.task_bar_controller import ProgressReporter
        from lilbee.data.ingest import SyncResult

        await _add(notes / "work")
        screen = ChatScreen.__new__(ChatScreen)
        notify = MagicMock()
        reporter = MagicMock(spec=ProgressReporter)
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
        with (
            mock.patch("lilbee.cli.tui.screens.chat.call_from_thread", notify),
            mock.patch("lilbee.runtime.asyncio_loop.run") as run,
        ):
            async with sync_running(cfg.data_root):
                screen._do_add([notes], MagicMock(spec=ProgressReporter))

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
        assert "A sync is running. Add notes again when it ends." in text
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

        assert result == {"error": "A sync is running. Add notes again when it ends."}


class TestTheOracle:
    async def test_an_absorb_and_a_sync_equal_an_index_built_from_the_parent_alone(
        self, library, notes, tmp_path
    ):
        await _add(notes / "work")
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


class TestSyncsHeldOff:
    async def test_it_raises_the_text_it_was_given_while_a_sync_runs(self, tmp_path):
        from lilbee.runtime.lock import syncs_held_off

        async with sync_running(tmp_path):
            with (
                pytest.raises(SyncRunningError, match=r"^try later$"),
                syncs_held_off(tmp_path, "try later"),
            ):
                pytest.fail("entered while a sync ran")
        with syncs_held_off(tmp_path, "try later"):
            pass

    def test_a_sync_in_another_process_waits_until_the_block_ends(self, tmp_path):
        import subprocess
        import sys

        from lilbee.runtime.lock import syncs_held_off

        script = (
            "import asyncio, sys\n"
            "from pathlib import Path\n"
            "from lilbee.runtime.lock import sync_running\n"
            "async def main():\n"
            "    async with sync_running(Path(sys.argv[1])):\n"
            "        print('IN', flush=True)\n"
            "asyncio.run(main())\n"
        )
        with syncs_held_off(tmp_path, "busy"):
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

        from lilbee.runtime.lock import _acquire_sync_lock, _release_sync_lock, syncs_held_off

        held = threading.Event()
        release = threading.Event()

        def _a_sync() -> None:
            lock = _acquire_sync_lock(tmp_path, write=False)
            held.set()
            release.wait(10)
            _release_sync_lock(lock)

        thread = threading.Thread(target=_a_sync)
        thread.start()
        assert held.wait(10)
        with (
            pytest.raises(SyncRunningError, match="busy"),
            syncs_held_off(tmp_path, "busy", wait=0.2),
        ):
            pytest.fail("entered while a sync ran")
        threading.Timer(0.3, release.set).start()
        with syncs_held_off(tmp_path, "busy", wait=10):
            assert release.is_set()
        thread.join(10)
