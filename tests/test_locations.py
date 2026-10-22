"""A source row stores where its file is; the name a person sees comes from the registry."""

from __future__ import annotations

import random
from pathlib import Path
from unittest import mock

import pytest

from lilbee.core import settings
from lilbee.core.config import CITATIONS_TABLE, SOURCES_TABLE, cfg
from lilbee.data.store import ChunkType, SourceType, Store
from lilbee.data.store.lance_helpers import table_names
from lilbee.data.store.locations import SourceLocations, root_location, under_sql
from tests.test_store import _citation, _dump, _holders, _seed_source


@pytest.fixture()
def config(tmp_path):
    """A config over a fresh library, with no registry."""
    return cfg.model_copy(
        update={"data_root": tmp_path, "lancedb_dir": tmp_path / "lancedb", "linked_roots": {}}
    )


def _paths(store: Store) -> list[str]:
    """The path column of every source row, as stored."""
    rows = store.open_table(SOURCES_TABLE).search().select(["path"]).limit(None).to_list()
    return sorted(row["path"] for row in rows)


def _names(store: Store) -> list[str]:
    return sorted(row["filename"] for row in store.get_sources())


class TestTheRule:
    def test_the_deepest_root_names_a_location(self):
        rule = SourceLocations({"outer": "/x/outer", "inner": "/x/outer/a/inner"})

        assert rule.name("/x/outer/a/inner/p.md") == "inner/p.md"
        assert rule.name("/x/outer/a/p.md") == "outer/a/p.md"
        assert rule.name("/x/outer") == "outer"
        assert rule.name("/x/outerwear/p.md") == "/x/outerwear/p.md"
        assert rule.name("loose.md") == "loose.md"

    def test_a_name_goes_back_to_its_location(self):
        rule = SourceLocations({"notes": "/x/notes", "paper.pdf": "/y/paper.pdf"})

        assert rule.location("notes/a/p.md") == "/x/notes/a/p.md"
        assert rule.location("notes") == "/x/notes"
        assert rule.location("paper.pdf") == "/y/paper.pdf"
        assert rule.location("paper.pdf/m0.md") == "/y/paper.pdf/m0.md"
        assert rule.location("loose.md") == "loose.md"
        assert rule.location("/x/notes/a.md") == "/x/notes/a.md"

    def test_a_location_is_its_own_location(self):
        rule = SourceLocations({"notes": "/x/notes"})

        for name in ("notes/a.md", "loose.md", "/elsewhere/b.md"):
            assert rule.location(rule.location(name)) == rule.location(name)

    def test_a_root_is_stored_in_its_posix_spelling(self):
        assert root_location("/x/notes") == "/x/notes"
        assert SourceLocations({"notes": "/x/notes/"}).location("notes/a.md") == "/x/notes/a.md"

    def test_a_clause_joined_to_the_root_filter_with_and_holds_for_every_row(self, tmp_path):
        import lancedb

        rows = [
            {"path": path, "kind": kind}
            for path in ("/x/notes", "/x/notes/a.md", "/x/notesy/b.md")
            for kind in ("document", "wiki")
        ]
        table = lancedb.connect(tmp_path / "db").create_table("t", data=rows)

        found = table.search().where(f"{under_sql('path', '/x/notes')} AND kind = 'wiki'")

        assert sorted(row["path"] for row in found.limit(None).to_list()) == [
            "/x/notes",
            "/x/notes/a.md",
        ]


def _registry_of(rng: random.Random) -> dict[str, str]:
    """A registry of nested and side-by-side roots, one label with a quote in it."""
    roots = {"outer": "/r/outer", "inner": "/r/outer/a/inner", "q'o": "/r/q'o"}
    if rng.random() < 0.5:
        roots["work"] = "/r/work/work"
        roots["work2"] = "/r/work"
    return roots


class TestTheSearchFilter:
    @pytest.mark.parametrize("seed", range(4))
    def test_the_listing_search_matches_the_rule_on_every_row(self, config, seed):
        rng = random.Random(seed)
        config.linked_roots = _registry_of(rng)
        store = Store(config)
        segments = ["a", "inner", "B", "q'o", "work", "d_1", "é", "x%y"]
        names: set[str] = set()
        for _ in range(60):
            head = rng.choice([*config.linked_roots, "loose", "_web"])
            names.add("/".join([head, *rng.sample(segments, rng.randint(1, 3))]) + ".md")
        for name in sorted(names):
            store.upsert_source(name, "h", 1)
        shown = _names(store)
        assert len(shown) == len(names) > 30

        for term in ["inner", "work/", "q'o", "OUTER/a", "é", "x%y", "d_1", "/r/", "loose"]:
            want = sorted(name for name in shown if term.lower() in name.lower())
            got = sorted(row["filename"] for row in store.get_sources(search=term))
            assert got == want, term
            assert store.count_sources(search=term) == len(want)

    def test_the_search_finds_a_title(self, config):
        store = Store(config)
        store.upsert_source("a.md", "h", 1)
        table = store.open_table(SOURCES_TABLE)
        table.update(where="path = 'a.md'", values={"title": "Harbour Notes"})

        assert [row["filename"] for row in store.get_sources(search="harbour")] == ["a.md"]


class TestWhatIsStored:
    def test_a_file_under_a_root_is_stored_at_its_location(self, config, tmp_path):
        config.linked_roots = {"notes": str(tmp_path / "notes")}
        store = Store(config)

        _seed_source(store, "notes/a.md")
        store.upsert_source("loose.md", "h", 1)

        assert _paths(store) == sorted([(tmp_path / "notes" / "a.md").as_posix(), "loose.md"])
        assert _names(store) == ["loose.md", "notes/a.md"]
        assert store.source_paths(
            store.source_ids(["notes/a.md"], SourceType.DOCUMENT).values()
        ) == {store.source_ids(["notes/a.md"], SourceType.DOCUMENT)["notes/a.md"]: "notes/a.md"}

    def test_adding_a_parent_changes_no_row_and_every_name(self, config, tmp_path):
        config.linked_roots = {"work": str(tmp_path / "notes" / "work")}
        store = Store(config)
        _seed_source(store, "work/a.md")
        before = _paths(store)
        cited = store.get_citations_for_source("work/a.md")

        config.linked_roots = {"notes": str(tmp_path / "notes")}

        assert _paths(store) == before
        assert _names(store) == ["notes/work/a.md"]
        assert store.get_citations_for_source("notes/work/a.md") == [
            {**cited[0], "source_filename": "notes/work/a.md"}
        ]
        assert store.get_citations_for_source("work/a.md") == []
        assert list(store.held_records()) == []


class TestAFolderNamedLikeTheWikiFolder:
    """Only a document source is stored at its location; a wiki page keeps its key."""

    def test_a_wiki_page_keeps_its_key_and_is_found_after_the_folder_goes(self, config, tmp_path):
        config.linked_roots = {"wiki": str(tmp_path / "home" / "wiki")}
        store = Store(config)
        _seed_source(store, "wiki/notes.md")
        store.upsert_source("wiki/summaries/a.md", "h", 1, SourceType.WIKI)
        store.replace_chunks("wiki/concepts/b.md", ChunkType.WIKI, [])
        pages = ["wiki/summaries/a.md", "wiki/concepts/b.md"]
        ids = store.source_ids(pages, SourceType.WIKI)

        config.linked_roots = {}

        assert set(pages) <= set(_paths(store))
        assert (tmp_path / "home" / "wiki" / "notes.md").as_posix() in _paths(store)
        assert store.source_ids(pages, SourceType.WIKI) == ids
        assert len(ids) == 2

    def test_the_open_leaves_a_wiki_page_under_its_key(self, config, tmp_path):
        old = Store(config)
        _seed_source(old, "wiki/notes.md")
        old.upsert_source("wiki/summaries/a.md", "h", 1, SourceType.WIKI)
        old.close()
        root = tmp_path / "home" / "wiki"
        config.linked_roots = {"wiki": str(root)}

        store = Store(config)

        assert sorted(_paths(store)) == sorted(
            [(root / "notes.md").as_posix(), "wiki/summaries/a.md"]
        )
        assert _names(store) == ["wiki/notes.md"]
        assert len(store.source_ids(["wiki/summaries/a.md"], SourceType.WIKI)) == 1

    def test_the_open_folds_no_wiki_page_into_a_document(self, config, tmp_path):
        root = tmp_path / "home" / "wiki"
        config.linked_roots = {"wiki": str(root)}
        _seed_source(Store(config), "wiki/summaries/a.md")
        config.linked_roots = {}
        old = Store(config)
        old.upsert_source("wiki/summaries/a.md", "h", 1, SourceType.WIKI)
        _seed_source(old, "wiki/notes.md")
        page = old.source_ids(["wiki/summaries/a.md"], SourceType.WIKI)["wiki/summaries/a.md"]
        old.close()
        config.linked_roots = {"wiki": str(root)}

        store = Store(config)

        assert (root / "notes.md").as_posix() in _paths(store)
        assert {"wiki/summaries/a.md", (root / "summaries" / "a.md").as_posix()} <= set(
            _paths(store)
        )
        assert page in store.source_paths([page])

    def test_an_open_with_only_wiki_pages_under_the_label_upgrades_nothing(self, config, tmp_path):
        Store(config).upsert_source("wiki/summaries/a.md", "h", 1, SourceType.WIKI)
        config.linked_roots = {"wiki": str(tmp_path / "home" / "wiki")}
        events: list[object] = []

        Store(config, on_upgrade=lambda _kind, event: events.append(event)).get_db()

        assert events == []


def _label_keyed_library(config, tmp_path) -> Path:
    """Rows a library before stored locations wrote: named by label. Returns the root."""
    root = tmp_path / "src" / "notes"
    old = Store(config)
    for name in ("notes/a.md", "notes/b/c.md", "notes", "loose.md"):
        _seed_source(old, name)
    old.close()
    config.linked_roots = {"notes": str(root)}
    return root


class TestTheOpenStoresLocations:
    def test_the_first_open_stores_each_row_under_a_label_at_its_location(self, config, tmp_path):
        root = _label_keyed_library(config, tmp_path).as_posix()

        store = Store(config)

        assert _paths(store) == sorted([f"{root}/a.md", f"{root}/b/c.md", root, "loose.md"])
        assert _names(store) == ["loose.md", "notes", "notes/a.md", "notes/b/c.md"]
        assert len(store.get_citations_for_source("notes/b/c.md")) == 1

    def test_a_second_open_writes_nothing(self, config, tmp_path):
        _label_keyed_library(config, tmp_path)
        Store(config).get_db()
        version = Store(config).sources_version()

        Store(config).get_db()

        assert Store(config).sources_version() == version

    def test_an_open_killed_after_one_label_is_finished_by_the_next(self, config, tmp_path):
        root = _label_keyed_library(config, tmp_path)
        _seed_source(Store(config.model_copy(update={"linked_roots": {}})), "other/x.md")
        config.linked_roots = {"notes": str(root), "other": str(tmp_path / "other")}
        from lancedb.table import LanceTable

        real = LanceTable.update
        calls: list[object] = []

        def _dies_on_the_second(self, *args, **kwargs):
            calls.append(kwargs.get("where"))
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real(self, *args, **kwargs)

        with (
            mock.patch.object(LanceTable, "update", _dies_on_the_second),
            pytest.raises(KeyboardInterrupt),
        ):
            Store(config).get_db()
        half = Store(config.model_copy(update={"linked_roots": {}}))
        assert any(Path(path).is_absolute() for path in _paths(half))
        assert any(path.startswith(("notes", "other/")) for path in _paths(half))

        store = Store(config)

        assert _names(store) == [
            "loose.md",
            "notes",
            "notes/a.md",
            "notes/b/c.md",
            "other/x.md",
        ]
        assert not any(path.startswith(("notes", "other/")) for path in _paths(store))

    def test_an_outer_row_for_a_file_of_an_inner_root_folds_into_the_inner_row(
        self, config, tmp_path
    ):
        outer, inner = tmp_path / "work", tmp_path / "work" / "deep"
        old = Store(config)
        _seed_source(old, "work/deep/a.md")
        _seed_source(old, "deep/a.md")
        _seed_source(old, "work/deep/only.md")
        inner_id = old.source_ids(["deep/a.md"], SourceType.DOCUMENT)["deep/a.md"]
        config.linked_roots = {"work": str(outer), "deep": str(inner)}

        store = Store(config)

        assert _names(store) == ["deep/a.md", "deep/only.md"]
        assert store.source_ids(["deep/a.md"], SourceType.DOCUMENT) == {"deep/a.md": inner_id}
        assert len(store.get_chunks_by_source("deep/a.md")) == 1
        cited = store.open_table(CITATIONS_TABLE).search().limit(None).to_list()
        assert sorted(row["source_id"] for row in cited).count(inner_id) == 2
        assert _holders(store, (inner / "a.md").as_posix()) == {("_sources", "path")}

    def test_a_label_with_no_root_keeps_its_rows_as_they_are(self, config, tmp_path):
        _label_keyed_library(config, tmp_path)
        config.linked_roots = {}

        store = Store(config)

        assert "notes/a.md" in _paths(store)


class TestTheRegistryIsReadAgain:
    @pytest.fixture(autouse=True)
    def _reads_config_toml(self, overlay_reads_config_toml):
        """The store reads config.toml, as a real process does."""

    def test_a_root_another_process_registered_is_shown_by_its_label(self, config, tmp_path):
        root = tmp_path / "notes"
        settings.set_value(config.data_root, "linked_roots", {"work": str(root / "work")})
        store = Store(config)
        _seed_source(store, "work/a.md")
        assert _names(store) == ["work/a.md"]

        settings.set_value(config.data_root, "linked_roots", {"notes": str(root)})

        assert _names(store) == ["notes/work/a.md"]
        assert config.linked_roots == {"notes": str(root)}

    def test_an_unchanged_file_is_not_read_again(self, config, tmp_path):
        settings.set_value(config.data_root, "linked_roots", {"notes": str(tmp_path)})
        store = Store(config)
        store.get_sources()

        with mock.patch.object(settings, "_read_roots") as read:
            store.get_sources()

        read.assert_not_called()

    def test_a_config_toml_deleted_before_its_stamp_is_read_has_none(self, tmp_path):
        with mock.patch.object(settings, "_overlay_file", return_value=tmp_path / "gone.toml"):
            assert settings.overlay_stamp(tmp_path) is None

    def test_a_pinned_store_keeps_the_registry_it_had(self, config, tmp_path):
        root = tmp_path / "notes"
        config.linked_roots = {"work": str(root / "work")}
        store = Store(config)
        _seed_source(store, "work/a.md")

        with store.registry_pinned():
            settings.set_value(config.data_root, "linked_roots", {"notes": str(root)})
            config.linked_roots = {"notes": str(root)}
            assert _names(store) == ["work/a.md"]
            assert store.source_ids(["work/a.md"], SourceType.DOCUMENT)

        assert _names(store) == ["notes/work/a.md"]


class TestARootThatMoves:
    def test_its_rows_take_the_new_folder(self, config, tmp_path):
        before, after = tmp_path / "before" / "notes", tmp_path / "after" / "notes"
        config.linked_roots = {"notes": str(before)}
        store = Store(config)
        _seed_source(store, "notes/a.md")
        source_id = store.source_ids(["notes/a.md"], SourceType.DOCUMENT)["notes/a.md"]

        store.relocate_root(str(before), str(after))
        config.linked_roots = {"notes": str(after)}

        assert _paths(store) == [(after / "a.md").as_posix()]
        assert store.source_ids(["notes/a.md"], SourceType.DOCUMENT) == {"notes/a.md": source_id}

    def test_a_row_already_at_the_new_place_keeps_it(self, config, tmp_path):
        before, after = tmp_path / "before" / "work", tmp_path / "after" / "work"
        config.linked_roots = {"work": str(before), "deep": str(after / "deep")}
        store = Store(config)
        _seed_source(store, "work/deep/a.md")
        _seed_source(store, "deep/a.md")
        inner_id = store.source_ids(["deep/a.md"], SourceType.DOCUMENT)["deep/a.md"]

        store.relocate_root(str(before), str(after))
        config.linked_roots = {"work": str(after), "deep": str(after / "deep")}

        assert _names(store) == ["deep/a.md"]
        assert store.source_ids(["deep/a.md"], SourceType.DOCUMENT) == {"deep/a.md": inner_id}

    def test_a_root_that_did_not_move_writes_nothing(self, config, tmp_path):
        config.linked_roots = {"notes": str(tmp_path)}
        store = Store(config)
        _seed_source(store, "notes/a.md")
        version = store.sources_version()

        store.relocate_root(str(tmp_path), str(tmp_path))
        store.relocate_root(str(tmp_path / "elsewhere"), str(tmp_path / "new"))

        assert store.sources_version() == version

    def test_a_store_with_no_source_table_moves_nothing(self, config, tmp_path):
        store = Store(config)

        store.relocate_root(str(tmp_path / "a"), str(tmp_path / "b"))

        assert store.open_table(SOURCES_TABLE) is None


class TestARootThatLeaves:
    def test_its_rows_take_its_label_and_a_root_inside_it_keeps_its_own(self, config, tmp_path):
        outer, inner = tmp_path / "outer", tmp_path / "outer" / "a" / "inner"
        config.linked_roots = {"outer": str(outer), "inner": str(inner), "home": str(tmp_path)}
        store = Store(config)
        _seed_source(store, "outer/x.md")
        _seed_source(store, "inner/y.md")
        store.upsert_source("wiki/summaries/a.md", "h", 1, SourceType.WIKI)
        cited = store.source_ids(["outer/x.md"], SourceType.DOCUMENT)["outer/x.md"]

        store.name_by_label(["outer", "home"])
        config.linked_roots = {"inner": str(inner)}

        assert _paths(store) == sorted(
            [(inner / "y.md").as_posix(), "outer/x.md", "wiki/summaries/a.md"]
        )
        assert store.source_paths([cited]) == {cited: "outer/x.md"}
        assert len(store.get_citations_for_source("outer/x.md")) == 1

    def test_a_root_with_no_row_writes_nothing(self, config, tmp_path):
        config.linked_roots = {"notes": str(tmp_path / "notes"), "empty": str(tmp_path / "e")}
        store = Store(config)
        _seed_source(store, "notes/a.md")
        version = store.sources_version()

        store.name_by_label(["empty", "unknown"])

        assert store.sources_version() == version

    def test_a_store_with_no_source_table_writes_nothing(self, config, tmp_path):
        config.linked_roots = {"notes": str(tmp_path / "notes")}
        store = Store(config)

        store.name_by_label(["notes"])

        assert store.open_table(SOURCES_TABLE) is None


class TestTheMoveCarriesTheCitations:
    def test_a_folded_row_gives_its_citation_to_the_row_it_lands_on(self, config, tmp_path):
        store = Store(config)
        _seed_source(store, "work/deep/a.md")
        store.add_citations([_citation("deep/a.md", key="k2")])
        config.linked_roots = {"work": str(tmp_path / "w"), "deep": str(tmp_path / "w" / "deep")}

        reopened = Store(config)

        keys = sorted(row["citation_key"] for row in reopened.get_citations_for_source("deep/a.md"))
        assert keys == ["k", "k2"]

    def test_each_citation_of_many_folds_cites_the_row_its_own_row_lands_on(self, config, tmp_path):
        names = [f"p{i}.md" for i in range(5)]
        store = Store(config)
        for name in names:
            _seed_source(store, f"work/deep/{name}")
            _seed_source(store, f"deep/{name}")
        store.add_citations([_citation(f"work/deep/{name}", key=f"c-{name}") for name in names])
        landing = store.source_ids([f"deep/{name}" for name in names], SourceType.DOCUMENT)
        store.close()
        config.linked_roots = {"work": str(tmp_path / "w"), "deep": str(tmp_path / "w" / "deep")}

        reopened = Store(config)

        cited = reopened.open_table(CITATIONS_TABLE).search().limit(None).to_list()
        pairs = {
            row["citation_key"]: row["source_id"] for row in cited if row["citation_key"] != "k"
        }
        assert pairs == {f"c-{name}": landing[f"deep/{name}"] for name in names}


def _folding_library(config, tmp_path) -> dict[str, str]:
    """A label-keyed library where two rows of an outer label land on rows of an inner root.

    Returns the registry its first open fills by.
    """
    old = Store(config)
    for name in ("work/deep/a.md", "deep/a.md", "work/deep/b.md", "deep/b.md", "work/c.md"):
        _seed_source(old, name)
    old.add_citations([_citation("work/deep/a.md", key="k2"), _citation("work/c.md", key="k3")])
    old.close()
    return {"work": str(tmp_path / "w"), "deep": str(tmp_path / "w" / "deep")}


def _copied(config, tmp_path, name: str, roots: dict[str, str]):
    """*config* over a copy of its library at *tmp_path*/*name*, filled by *roots*."""
    import shutil

    shutil.copytree(config.lancedb_dir, tmp_path / name / "lancedb")
    return config.model_copy(
        update={
            "data_root": tmp_path / name,
            "lancedb_dir": tmp_path / name / "lancedb",
            "linked_roots": roots,
        }
    )


def _dies_on(nth: int, where_starts: str):
    """A ``LanceTable.update`` that raises on its *nth* call whose filter starts *where_starts*."""
    from lancedb.table import LanceTable

    real = LanceTable.update
    calls: list[str] = []

    def _update(self, *args, **kwargs):
        if str(kwargs.get("where", "")).startswith(where_starts):
            calls.append(kwargs["where"])
            if len(calls) == nth:
                raise KeyboardInterrupt
        return real(self, *args, **kwargs)

    return mock.patch.object(LanceTable, "update", _update)


@pytest.fixture()
def small_batches():
    """A fill that reads, writes and folds one row at a time."""
    from lilbee.data.store import core

    with (
        mock.patch.object(core, "_FILL_BATCH_ROWS", 1),
        mock.patch.object(core, "_CITATION_LOOKUP_IDS", 1),
    ):
        yield


@pytest.mark.usefixtures("small_batches")
class TestTheFillInBatches:
    def test_each_write_of_the_fill_names_one_batch_of_rows(self, config, tmp_path):
        root = _label_keyed_library(config, tmp_path).as_posix()
        from lancedb.table import LanceTable

        real = LanceTable.update
        wheres: list[str] = []

        def _update(self, *args, **kwargs):
            wheres.append(str(kwargs.get("where")))
            return real(self, *args, **kwargs)

        with mock.patch.object(LanceTable, "update", _update):
            store = Store(config)
            assert _paths(store) == sorted([f"{root}/a.md", f"{root}/b/c.md", root, "loose.md"])

        rewrites = [where for where in wheres if where.startswith("_rowid IN (")]
        assert len(rewrites) == 3
        assert all("," not in where for where in rewrites)

    def test_rows_that_land_on_rows_of_an_inner_root_fold_into_them(self, config, tmp_path):
        roots = _folding_library(config, tmp_path)
        before = Store(config)
        inner = before.source_ids(["deep/a.md", "deep/b.md"], SourceType.DOCUMENT)
        outer = before.source_ids(["work/c.md"], SourceType.DOCUMENT)["work/c.md"]
        config.linked_roots = roots

        store = Store(config)

        assert _names(store) == ["deep/a.md", "deep/b.md", "work/c.md"]
        assert store.source_ids(["deep/a.md", "deep/b.md"], SourceType.DOCUMENT) == inner
        cited = store.open_table(CITATIONS_TABLE).search().limit(None).to_list()
        by_key = sorted((row["citation_key"], row["source_id"]) for row in cited)
        assert by_key == sorted(
            [
                *[("k", inner["deep/a.md"])] * 2,
                ("k", inner["deep/b.md"]),
                ("k", inner["deep/b.md"]),
                ("k", outer),
                ("k2", inner["deep/a.md"]),
                ("k3", outer),
            ]
        )
        assert len(store.get_chunks_by_source("deep/a.md")) == 1
        assert "_fill_pairs" not in table_names(store.get_db())

    @pytest.mark.parametrize(
        ("nth", "where_starts"),
        [(1, "source_id IN ("), (2, "source_id IN ("), (1, "_rowid IN ("), (2, "_rowid IN (")],
        ids=["first-citation", "second-citation", "first-rewrite", "second-rewrite"],
    )
    def test_a_fill_killed_partway_is_finished_by_the_next_open(
        self, config, tmp_path, nth, where_starts
    ):
        roots = _folding_library(config, tmp_path)
        whole = Store(_copied(config, tmp_path, "whole", roots))
        killed = _copied(config, tmp_path, "killed", roots)
        with _dies_on(nth, where_starts), pytest.raises(KeyboardInterrupt):
            Store(killed).get_db()
        half = Store(killed.model_copy(update={"linked_roots": {}}))
        assert any(path.startswith("work/") for path in _paths(half))
        folding = where_starts.startswith("source_id")
        assert ("_fill_pairs" in table_names(half.get_db())) == folding

        reopened = Store(killed)

        assert _dump(reopened) == _dump(whole)
        assert "_fill_pairs" not in table_names(reopened.get_db())


_HOLD_THE_LOCK = """
import sys, time
from filelock import FileLock
lock = FileLock(sys.argv[1])
lock.acquire()
print("held", flush=True)
time.sleep(60)
"""


def _in_scope(config, fn):
    """*fn* run in a thread that names *config* as the active one."""
    from lilbee.core.config import config_scope

    def run(*args):
        with config_scope(config):
            fn(*args)

    return run


class TestAResolverInALongRunningProcess:
    """A server that ran no store call still names a root another process registered."""

    @pytest.fixture()
    def server(self, config, overlay_reads_config_toml):
        from lilbee.core.config import config_scope

        with config_scope(config):
            yield config

    def _another_process_registers(self, config, label, root):
        settings.set_value(config.data_root, "linked_roots", {label: str(root)})

    def test_a_label_added_since_resolves_to_its_root(self, server, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path

        assert resolve_source_path("notes/a.md") == server.documents_dir / "notes" / "a.md"

        self._another_process_registers(server, "notes", tmp_path / "elsewhere")

        assert resolve_source_path("notes/a.md") == tmp_path / "elsewhere" / "a.md"

    def test_the_walked_root_follows_the_file(self, server, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_root

        assert resolve_source_root("notes/a.md") == (
            server.documents_dir,
            server.documents_dir / "notes" / "a.md",
        )

        self._another_process_registers(server, "notes", tmp_path / "elsewhere")

        assert resolve_source_root("notes/a.md") == (
            tmp_path / "elsewhere",
            tmp_path / "elsewhere" / "a.md",
        )

    def test_the_checked_resolver_accepts_a_new_label(self, server, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path_checked

        elsewhere = tmp_path / "elsewhere"
        assert resolve_source_path_checked("notes/a.md") is not None
        assert resolve_source_path_checked("notes/../../x") is None
        self._another_process_registers(server, "notes", elsewhere)

        assert resolve_source_path_checked("notes/a.md") == (elsewhere / "a.md").resolve()
        assert resolve_source_path_checked("notes/../../x") is None

    def test_a_file_in_a_new_root_is_found_on_disk(self, server, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path

        live = tmp_path / "elsewhere" / "a.md"
        live.parent.mkdir()
        live.write_text("x", encoding="utf-8")
        self._another_process_registers(server, "notes", tmp_path / "elsewhere")

        assert resolve_source_path("notes/a.md").exists()

    def test_a_pinned_registry_ignores_the_file(self, server, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path

        server.linked_roots = {"notes": str(tmp_path / "before")}
        with settings.pin_registry(server):
            self._another_process_registers(server, "notes", tmp_path / "after")
            assert resolve_source_path("notes/a.md") == tmp_path / "before" / "a.md"

        assert resolve_source_path("notes/a.md") == tmp_path / "after" / "a.md"

    def test_an_unparsable_file_leaves_the_registry_it_read_before(self, server, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path

        self._another_process_registers(server, "notes", tmp_path / "read")
        assert resolve_source_path("notes/a.md") == tmp_path / "read" / "a.md"

        (server.data_root / "config.toml").write_text("not [valid", encoding="utf-8")

        assert resolve_source_path("notes/a.md") == tmp_path / "read" / "a.md"

    def test_a_deleted_file_leaves_the_registry_it_read_before(self, server, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path

        self._another_process_registers(server, "notes", tmp_path / "read")
        assert resolve_source_path("notes/a.md") == tmp_path / "read" / "a.md"

        (server.data_root / "config.toml").unlink()

        assert resolve_source_path("notes/a.md") == tmp_path / "read" / "a.md"

    def test_an_unchanged_file_is_not_read_again(self, server, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path

        self._another_process_registers(server, "notes", tmp_path / "read")
        resolve_source_path("notes/a.md")

        with mock.patch.object(settings, "_read_roots") as read:
            resolve_source_path("notes/a.md")

        read.assert_not_called()

    def test_a_caller_does_not_wait_for_a_read_another_process_blocks(self, server, tmp_path):
        import subprocess
        import sys
        import threading
        import time

        from lilbee.data.ingest.discovery import resolve_source_path

        self._another_process_registers(server, "notes", tmp_path / "read")
        assert resolve_source_path("notes/a.md") == tmp_path / "read" / "a.md"
        self._another_process_registers(server, "notes", tmp_path / "next")

        holder = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                _HOLD_THE_LOCK,
                str(server.data_root / "config.toml.lock"),
            ],
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "held"
            first = threading.Thread(
                target=_in_scope(server, resolve_source_path), args=("notes/a.md",)
            )
            first.start()
            time.sleep(0.2)  # the first caller is in the file-lock wait

            started = time.monotonic()
            second = resolve_source_path("notes/a.md")
            waited = time.monotonic() - started
        finally:
            holder.terminate()
            holder.wait(5)
        first.join(15)

        assert waited < 0.5
        assert second == tmp_path / "read" / "a.md"
        assert resolve_source_path("notes/a.md") == tmp_path / "next" / "a.md"

    def test_a_pin_taken_during_a_read_keeps_the_registry_it_holds(self, server, tmp_path):
        import contextlib

        from lilbee.data.ingest.discovery import resolve_source_path

        server.linked_roots = {"notes": str(tmp_path / "before")}
        self._another_process_registers(server, "notes", tmp_path / "after")
        real = settings._read_roots
        pins = contextlib.ExitStack()

        def pin_after_reading(config, accept):
            def pin_then_accept(roots):
                pins.enter_context(settings.pin_registry(config))  # another thread pins mid-read
                accept(roots)

            real(config, pin_then_accept)

        with mock.patch.object(settings, "_read_roots", pin_after_reading):
            assert resolve_source_path("notes/a.md") == tmp_path / "before" / "a.md"
        pins.close()

        assert resolve_source_path("notes/a.md") == tmp_path / "after" / "a.md"

    def test_a_pinned_registry_reports_that_it_was_not_refreshed(self, server):
        assert settings.refresh_registry(server) is True
        with settings.pin_registry(server):
            assert settings.refresh_registry(server) is False
        assert settings.refresh_registry(server) is True

    def test_a_read_that_fails_is_tried_again(self, server, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path

        self._another_process_registers(server, "notes", tmp_path / "read")
        with (
            mock.patch.object(settings, "_read_roots", side_effect=OSError),
            pytest.raises(OSError),
        ):
            resolve_source_path("notes/a.md")

        assert resolve_source_path("notes/a.md") == tmp_path / "read" / "a.md"

    def test_the_checked_resolver_holds_one_registry(self, server, tmp_path):
        from lilbee.data.ingest import discovery

        self._another_process_registers(server, "notes", tmp_path / "first")
        real = discovery.resolve_source_path

        def resolve_then_another_thread_refreshes(filename):
            resolved = real(filename)
            self._another_process_registers(server, "notes", tmp_path / "second")
            settings.refresh_registry(server)  # a pinned registry ignores it
            return resolved

        with mock.patch.object(
            discovery, "resolve_source_path", resolve_then_another_thread_refreshes
        ):
            checked = discovery.resolve_source_path_checked("notes/a.md")

        assert checked == (tmp_path / "first" / "a.md").resolve()

    def test_a_removal_reads_the_roots_another_process_registered(self, server, tmp_path):
        from lilbee.app import ingest

        self._another_process_registers(server, "notes", tmp_path / "elsewhere")
        with (
            mock.patch.object(ingest, "get_services"),
            mock.patch.object(ingest, "removable_names", return_value=[]),
            mock.patch.object(ingest, "held_failures", return_value={}),
            mock.patch.object(ingest, "forget_roots", return_value=[]),
            mock.patch.object(ingest, "forget_removed_from_wiki_index"),
            mock.patch.object(ingest, "_removal_holds", return_value=[]) as holds,
        ):
            ingest.remove_documents_durably(["notes"], targets=[])

        assert holds.call_args.args[1] == ["notes"]

    def test_forgetting_a_root_reads_the_roots_another_process_registered(self, server, tmp_path):
        from lilbee.app import ingest

        self._another_process_registers(server, "notes", tmp_path / "elsewhere")
        with (
            mock.patch.object(ingest, "get_services"),
            mock.patch.object(ingest, "unmark_sources_under") as unmark,
            mock.patch.object(ingest, "unregister_roots", return_value=["notes"]),
        ):
            assert ingest.forget_roots(["notes"]) == ["notes"]

        assert unmark.call_args.args[0] == [tmp_path / "elsewhere"]

    def test_held_keys_are_matched_against_one_registry(self, server, tmp_path):
        from lilbee.app.ingest import _held_covering

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        self._another_process_registers(server, "docs", elsewhere)

        covered, inside = _held_covering(["docs/x.md"], [elsewhere])

        assert covered == {"docs/x.md"}
        assert inside == set()

    def test_the_store_names_sources_by_what_a_resolver_read(self, server, tmp_path):
        from lilbee.data.ingest.discovery import resolve_source_path

        store = Store(server)
        _seed_source(store, "notes/a.md")
        self._another_process_registers(server, "notes", tmp_path / "elsewhere")
        resolve_source_path("notes/a.md")

        assert _names(store) == ["notes/a.md"]
        assert server.linked_roots == {"notes": str(tmp_path / "elsewhere")}
