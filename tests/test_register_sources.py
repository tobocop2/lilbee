"""What an add writes, and which folder it takes for a source that is registered."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

import lilbee.app.services as svc_mod
from lilbee.app.ingest import names_outside_corpus, register_sources, source_label_taken
from lilbee.cli.app import app
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.core.system import DiskSpelling
from lilbee.data.ingest.discovery import discover_files
from lilbee.data.ingest.skip_marker import SkipKind, load_skip_kinds, mark_removed
from lilbee.data.store import Store
from lilbee.runtime.lock import LockingUnsupportedError
from tests.conftest import filesystem_without_locking, make_mock_services

_NO_LOCKING = "is on a filesystem that does not support file locking"


@pytest.fixture(autouse=True)
def library(tmp_path, monkeypatch):
    """A data root of this test alone, with a real store and config.toml read from disk."""
    snapshot = cfg.model_copy()
    root = tmp_path / "library"
    root.mkdir()
    cfg.data_root = root
    cfg.documents_dir = root / "documents"
    cfg.data_dir = root / "data"
    cfg.lancedb_dir = root / "data" / "lancedb"
    cfg.linked_roots = {}
    monkeypatch.setenv("LILBEE_DATA", str(root))
    monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
    services = make_mock_services(store=Store(cfg))
    svc_mod.set_services(services)
    yield root
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _write(path: Path, text: str = "# Page\n\nA page about tide tables.\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def notes(tmp_path) -> Path:
    """``notes`` with one loose file and the folder ``work`` holding one."""
    base = tmp_path / "src" / "notes"
    _write(base / "loose.md")
    _write(base / "work" / "plan.md")
    return base


def _registry() -> dict[str, str]:
    return dict(settings.load(cfg.data_root).get("linked_roots") or {})


def _files(root: Path) -> dict[str, bytes]:
    """Every file below *root* with its bytes."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _other_case(path: Path) -> Path:
    """*path* with the letter case of its last part changed."""
    return path.with_name(path.name.swapcase())


def _one_entry(first: Path, second: Path) -> bool:
    """Whether this disk holds *first* and *second* as one entry."""
    return second.exists() and os.path.samefile(first, second)


class TestADataRootThatCannotLockFiles:
    """An add that the data root refuses writes nothing."""

    def test_the_refusal_leaves_the_library_as_it_was(self, library, notes):
        other = _write(notes.parent / "other" / "memo.md").parent
        register_sources([other])
        mark_removed(cfg.data_root, {"notes/loose.md": "hash of notes/loose.md"})
        before = _files(library)
        assert "config.toml" in before

        with (
            filesystem_without_locking(),
            pytest.raises(LockingUnsupportedError, match=_NO_LOCKING),
        ):
            register_sources([notes])

        assert _files(library) == before
        assert sorted(_registry()) == ["other"]
        assert sorted(cfg.linked_roots) == ["other"]
        assert load_skip_kinds(cfg.data_root) == {"notes/loose.md": SkipKind.REMOVED}

    def test_the_same_add_registers_on_a_data_root_that_locks(self, library, notes):
        """The control for the test above: the add registers and drops the removal record."""
        mark_removed(cfg.data_root, {"notes/loose.md": "hash of notes/loose.md"})

        assert register_sources([notes]).registered == ["notes"]

        assert sorted(_registry()) == ["notes"]
        assert load_skip_kinds(cfg.data_root) == {}

    def test_a_data_root_that_does_not_exist_yet_is_refused_too(self, library, notes, tmp_path):
        cfg.data_root = tmp_path / "fresh"

        with (
            filesystem_without_locking(),
            pytest.raises(LockingUnsupportedError, match=_NO_LOCKING),
        ):
            register_sources([notes])

        assert _files(cfg.data_root) == {}

    def test_a_probe_that_fails_another_way_does_not_refuse_the_add(self, library, notes):
        """A sync runs unmarked on such a data root, so the add before it registers."""
        failure = OSError(5, "Input/output error")
        with mock.patch("lilbee.runtime.lock._require_locking", side_effect=failure) as probe:
            result = register_sources([notes])

        probe.assert_called_once_with(cfg.data_root)
        assert result.registered == ["notes"]
        assert sorted(_registry()) == ["notes"]

    def test_an_add_of_no_path_probes_nothing(self, library):
        with mock.patch("lilbee.runtime.lock._require_locking") as probe:
            assert register_sources([]).registered == []

        probe.assert_not_called()

    @pytest.mark.parametrize("flags", [[], ["--json"]], ids=["human", "json"])
    def test_the_command_reports_the_refusal_and_registers_nothing(self, library, notes, flags):
        with filesystem_without_locking():
            outcome = CliRunner().invoke(app, [*flags, "add", str(notes)])

        assert outcome.exit_code == 1
        assert _NO_LOCKING in outcome.output.replace("\n", " ")
        assert "Registered" not in outcome.output
        if flags:
            assert set(json.loads(outcome.output)) == {"error"}
        assert _registry() == {}


class TestOnePathInAnotherLetterCase:
    """The disk decides whether two paths are one entry; their spelling does not.

    Each test asks this disk first, so it asserts one source where the disk
    ignores letter case and two where it does not.
    """

    def test_a_folder_inside_a_source_is_not_a_second_source(self, library, notes):
        register_sources([notes])
        typed = _other_case(notes) / "work"
        one_entry = _one_entry(notes / "work", typed)
        if not one_entry:
            _write(typed / "plan.md")

        result = register_sources([typed])

        if one_entry:
            assert result.registered == [] and result.overlapping == ["work"]
            assert _registry() == {"notes": str(notes.resolve())}
            assert sorted(discover_files()) == ["notes/loose.md", "notes/work/plan.md"]
        else:
            assert result.registered == ["work"] and result.overlapping == []
            assert sorted(_registry()) == ["notes", "work"]

    def test_the_folder_of_a_source_is_that_source(self, library, notes):
        register_sources([notes])
        typed = _other_case(notes)
        one_entry = _one_entry(notes, typed)
        if not one_entry:
            _write(typed / "loose.md")

        result = register_sources([typed])

        if one_entry:
            assert result.tracked == ["notes"] and result.registered == []
            assert _registry() == {"notes": str(notes.resolve())}
        else:
            assert result.registered == [typed.name]

    def test_a_new_source_is_stored_as_the_disk_spells_it(self, library, notes):
        typed = _other_case(notes)
        one_entry = _one_entry(notes, typed)
        if not one_entry:
            _write(typed / "loose.md")

        result = register_sources([typed])

        stored = notes if one_entry else typed
        assert result.registered == [stored.name]
        assert _registry() == {stored.name: str(stored.resolve())}

    def test_a_parent_takes_in_the_source_below_it(self, library, notes):
        register_sources([notes / "work"])
        typed = _other_case(notes.parent) / "notes"
        one_entry = _one_entry(notes, typed)
        if not one_entry:
            _write(typed / "loose.md")

        result = register_sources([typed])

        assert result.registered == ["notes"]
        if one_entry:
            assert result.absorbed_into == {"notes": ["work"]}
            assert _registry() == {"notes": str(notes.resolve())}
        else:
            assert result.absorbed_into == {}
            assert sorted(_registry()) == ["notes", "work"]

    def test_a_name_that_is_not_ascii_is_one_source(self, library, tmp_path):
        folder = _write(tmp_path / "src" / "Ärger Übung" / "page.md").parent
        register_sources([folder])
        typed = _other_case(folder)
        one_entry = _one_entry(folder, typed)
        if not one_entry:
            _write(typed / "page.md")

        result = register_sources([typed])

        if one_entry:
            assert result.tracked == [folder.name] and result.registered == []
        else:
            assert result.registered == [typed.name]

    def test_a_path_that_does_not_exist_below_a_source_overlaps_it(self, library, notes):
        register_sources([notes])
        typed = _other_case(notes) / "missing"
        one_entry = _one_entry(notes, typed.parent)

        result = register_sources([typed])

        if one_entry:
            assert result.overlapping == ["missing"] and result.registered == []
        else:
            assert result.registered == ["missing"]

    def test_a_link_to_a_source_is_that_source(self, library, notes, tmp_path):
        register_sources([notes])
        link = tmp_path / "LINK"
        link.symlink_to(notes, target_is_directory=True)

        result = register_sources([link])

        assert result.tracked == ["notes"] and result.registered == []
        assert _registry() == {"notes": str(notes.resolve())}

    def test_a_trailing_separator_names_the_same_source(self, library, notes):
        register_sources([notes])

        result = register_sources([Path(f"{notes}{os.sep}")])

        assert result.tracked == ["notes"] and result.registered == []

    def test_two_folders_the_disk_keeps_apart_are_two_sources(self, library, notes):
        typed = _other_case(notes)
        asked: contextlib.AbstractContextManager[object] = contextlib.nullcontext()
        if _one_entry(notes, typed):
            # This disk cannot hold both, so it answers as one that can.
            asked = mock.patch.object(
                DiskSpelling, "of", lambda _self, path: Path(os.path.abspath(path))
            )
        else:
            _write(typed / "loose.md")

        with asked:
            register_sources([notes])
            result = register_sources([typed])

        assert result.registered == [typed.name] and result.tracked == []
        assert _registry() == {"notes": str(notes), typed.name: str(typed)}

    def test_a_folder_around_the_documents_folder_is_refused(self, library):
        typed = _other_case(library)
        one_entry = _one_entry(library, typed)
        if not one_entry:
            _write(typed / "page.md")

        result = register_sources([typed])

        if one_entry:
            assert result.containing == [typed.name] and result.registered == []
        else:
            assert result.containing == [] and result.registered == [typed.name]

    def test_a_folder_inside_the_documents_folder_is_tracked(self, library):
        owned = _write(cfg.documents_dir / "owned" / "page.md").parent
        typed = _other_case(library) / "documents" / "owned"
        one_entry = _one_entry(owned, typed)
        if not one_entry:
            _write(typed / "page.md")

        result = register_sources([typed])

        if one_entry:
            assert result.tracked == ["owned"] and result.name_taken == []
        else:
            # A second folder: its name is the name of an entry the documents folder holds.
            assert result.tracked == [] and result.name_taken == ["owned"]
        assert result.registered == [] and _registry() == {}

    def test_a_documents_folder_configured_in_another_case_is_that_folder(self, library):
        owned = _write(cfg.documents_dir / "owned" / "page.md").parent
        typed = _other_case(library) / "documents"
        one_entry = _one_entry(cfg.documents_dir, typed)
        cfg.documents_dir = typed

        outside = names_outside_corpus([owned])
        result = register_sources([owned])

        if one_entry:
            assert outside == []
            assert result.tracked == ["owned"] and result.registered == []
        else:
            assert outside == ["owned"]
            assert result.tracked == [] and result.registered == ["owned"]

    def test_a_parent_takes_in_a_source_stored_in_another_case(self, library, notes):
        typed = _other_case(notes) / "work"
        one_entry = _one_entry(notes / "work", typed)
        if not one_entry:
            _write(typed / "plan.md")
        settings.set_value(cfg.data_root, "linked_roots", {"work": str(typed)})

        result = register_sources([notes, notes / "work"])

        assert result.registered == ["notes"]
        assert result.overlapping == ["work"] and result.tracked == []
        if one_entry:
            assert result.absorbed_into == {"notes": ["work"]}
            assert _registry() == {"notes": str(notes.resolve())}
        else:
            assert result.absorbed_into == {}
            assert _registry() == {"notes": str(notes.resolve()), "work": str(typed)}

    def test_a_source_stored_in_another_case_is_still_that_source(self, library, notes):
        typed = _other_case(notes)
        one_entry = _one_entry(notes, typed)
        if not one_entry:
            _write(typed / "loose.md")
        settings.set_value(cfg.data_root, "linked_roots", {"notes": str(typed)})

        inside = register_sources([notes / "work"])
        same = register_sources([notes])

        if one_entry:
            assert inside.overlapping == ["work"] and inside.registered == []
            assert same.tracked == ["notes"] and same.registered == []
            assert _registry() == {"notes": str(typed)}
        else:
            assert inside.registered == ["work"]
            assert same.name_taken == ["notes"]

    @pytest.mark.parametrize(
        "stored_as_typed", [False, True], ids=["stored-as-listed", "stored-as-typed"]
    )
    def test_naming_the_folder_brings_back_a_file_removed_from_it(
        self, library, notes, stored_as_typed
    ):
        typed = _other_case(notes)
        one_entry = _one_entry(notes, typed)
        if not one_entry:
            _write(typed / "loose.md")
        stored, named = (typed, notes) if stored_as_typed else (notes, typed)
        settings.set_value(cfg.data_root, "linked_roots", {"notes": str(stored)})
        cfg.linked_roots = {"notes": str(stored)}
        mark_removed(cfg.data_root, {"notes/loose.md": "0" * 64})

        register_sources([named])

        held = {} if one_entry else {"notes/loose.md": SkipKind.REMOVED}
        assert load_skip_kinds(cfg.data_root) == held

    def test_the_overwrite_question_is_not_asked_for_the_same_folder(self, library, notes):
        register_sources([notes])
        typed = _other_case(notes.parent) / "notes"
        one_entry = _one_entry(notes, typed)
        if not one_entry:
            _write(typed / "loose.md")

        assert source_label_taken("notes", typed) is not one_entry
        assert source_label_taken("notes", notes) is False
        assert source_label_taken("notes", notes.parent / "elsewhere" / "notes") is True

    @pytest.mark.parametrize("flags", [[], ["--json"]], ids=["human", "json"])
    def test_the_command_adds_no_second_source(self, library, notes, flags):
        register_sources([notes])
        typed = _other_case(notes) / "work"
        one_entry = _one_entry(notes / "work", typed)
        if not one_entry:
            _write(typed / "plan.md")

        outcome = CliRunner().invoke(app, [*flags, "add", str(typed)])

        assert outcome.exit_code == 0
        assert sorted(_registry()) == (["notes"] if one_entry else ["notes", "work"])

    def test_an_add_reads_a_folder_as_often_with_many_sources_as_with_one(self, library, tmp_path):
        shelf = Path(os.path.realpath(tmp_path)) / "shelf"
        folders = [_write(shelf / f"source{n}" / "page.md").parent for n in range(9)]
        read: list[str] = []
        listdir = os.listdir

        def _listing(folder: str) -> list[str]:
            read.append(folder)
            return listdir(folder)

        register_sources(folders[:1])
        with mock.patch.object(os, "listdir", _listing):
            register_sources(folders[1:2])
            with_one_source = read.count(str(shelf))
            register_sources(folders[2:8])
            read.clear()
            register_sources(folders[8:])

        assert sorted(_registry()) == [f"source{n}" for n in range(9)]
        assert with_one_source > 0
        assert read.count(str(shelf)) == with_one_source
