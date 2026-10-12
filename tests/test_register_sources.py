"""What an add writes when the data root refuses it."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

import lilbee.app.services as svc_mod
from lilbee.app.ingest import register_sources
from lilbee.cli.app import app
from lilbee.core import settings
from lilbee.core.config import cfg
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
