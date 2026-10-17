"""A source row stores where its file is; the name a person sees comes from the registry."""

from __future__ import annotations

import random
from pathlib import Path
from unittest import mock

import pytest

from lilbee.core import settings
from lilbee.core.config import CITATIONS_TABLE, SOURCES_TABLE, cfg
from lilbee.data.store import ChunkType, SourceType, Store
from lilbee.data.store.locations import SourceLocations, root_location
from tests.test_store import _citation, _holders, _seed_source


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
        assert any(path.startswith("/") for path in _paths(half))
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

        with mock.patch.object(settings, "overlay_persisted_roots") as overlay:
            store.get_sources()

        overlay.assert_not_called()

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
