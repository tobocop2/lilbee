"""A rebuild is reported only when the index holds chunks, never on a fresh install."""

from __future__ import annotations

from unittest import mock

import pytest

from lilbee.app import profiles
from lilbee.app.profiles import DiffRow, ProfileDiff, ProfileEffect, RebuildReason
from lilbee.app.services import peek_services, set_services
from lilbee.app.settings import apply_settings_update, index_has_chunks
from lilbee.cli.tui.screens.profile_dialogs import switch_plan
from lilbee.core.config import cfg
from lilbee.core.config.enums import SettingSource
from lilbee.core.profile_files import ProfileFile
from lilbee.data.store import Store
from tests._profile_fixtures import (
    gate_releases_at_once,  # noqa: F401 -- autouse fixture, applied by import
    isolated_cfg,  # noqa: F401 -- autouse fixture, applied by import
)
from tests.conftest import make_mock_services


@pytest.fixture(autouse=True)
def real_store():
    set_services(make_mock_services(store=Store(cfg)))
    yield
    set_services(None)


def test_a_fresh_install_has_no_chunks_and_no_library_is_made():
    assert not cfg.lancedb_dir.exists()
    assert index_has_chunks() is False
    assert not cfg.lancedb_dir.exists()


@pytest.mark.parametrize("holds", [True, False])
def test_the_check_with_no_container_builds_no_provider(
    holds: bool, monkeypatch: pytest.MonkeyPatch
):
    set_services(None)
    monkeypatch.setattr(Store, "has_chunks", lambda self: holds)
    with mock.patch("lilbee.providers.factory.create_provider") as create_provider:
        assert index_has_chunks() is holds
    create_provider.assert_not_called()
    assert peek_services() is None


def test_held_records_on_a_fresh_install_are_empty_and_make_no_library():
    assert Store(cfg).held_records() == {}
    assert not cfg.lancedb_dir.exists()


def test_held_records_of_an_older_record_file_are_imported_on_a_data_dir_with_no_library():
    (cfg.data_root / "skipped_sources.json").write_text('{"scan.pdf": "h1"}', encoding="utf-8")
    assert not cfg.lancedb_dir.exists()
    assert set(Store(cfg).held_records()) == {"scan.pdf"}


def test_a_fresh_install_opens_no_table_to_answer():
    store = Store(cfg)
    with mock.patch.object(Store, "open_table") as open_table:
        assert store.has_chunks() is False
        assert store.held_records() == {}
    assert open_table.call_count == 0


def test_a_library_folder_that_exists_is_opened_to_answer():
    cfg.lancedb_dir.mkdir(parents=True)
    store = Store(cfg)
    with mock.patch.object(Store, "open_table", return_value=None) as open_table:
        assert store.has_chunks() is False
        assert store.held_records() == {}
    assert open_table.call_count == 2


def test_an_open_library_is_asked_to_answer():
    store = Store(cfg)
    store.get_db()
    assert cfg.lancedb_dir.exists()
    with mock.patch.object(Store, "open_table", return_value=None) as open_table:
        assert store.has_chunks() is False
    assert open_table.call_count == 1


def test_a_setting_write_on_a_fresh_install_reports_no_rebuild_and_makes_no_library():
    assert apply_settings_update({"chunk_size": 900}).reindex_required is False
    assert not cfg.lancedb_dir.exists()


def test_rebuild_reason_on_a_fresh_install_is_none_and_makes_no_library():
    assert profiles.rebuild_reason(("chunk_size",), True) is None
    assert not cfg.lancedb_dir.exists()


def test_a_setting_write_on_an_empty_library_reports_no_rebuild(monkeypatch):
    monkeypatch.setattr(Store, "has_chunks", lambda self: False)
    assert apply_settings_update({"chunk_size": 900}).reindex_required is False


def test_a_setting_write_on_an_index_with_chunks_reports_a_rebuild(monkeypatch):
    monkeypatch.setattr(Store, "has_chunks", lambda self: True)
    assert apply_settings_update({"chunk_size": 900}).reindex_required is True


def test_rebuild_reason_on_an_empty_index_still_retries_a_held_out_file(monkeypatch):
    monkeypatch.setattr(Store, "has_chunks", lambda self: False)
    with mock.patch.object(profiles, "held_out_sources", return_value=([], 1)):
        assert profiles.rebuild_reason(("chunk_size",), True) is RebuildReason.HELD_OUT


def test_rebuild_reason_on_an_index_with_chunks_names_the_change(monkeypatch):
    monkeypatch.setattr(Store, "has_chunks", lambda self: True)
    assert profiles.rebuild_reason(("chunk_size",), True) is RebuildReason.CHANGES
    assert profiles.rebuild_reason(("chunk_size",), False) is RebuildReason.SETS_REINDEX


def _reindex_plan():
    row = DiffRow("chunk_size", 512, SettingSource.BUILT_IN, 900, ProfileEffect.REINDEX)
    diff = ProfileDiff("big", (row,), (), 0)
    profile = ProfileFile(
        name="big",
        description=None,
        authors=(),
        tested_on=None,
        format=1,
        min_lilbee=None,
        evidence=None,
        values={"chunk_size": 900},
    )
    return switch_plan(profile, diff)


def test_the_apply_dialog_plan_on_a_fresh_install_offers_no_rebuild_and_makes_no_library():
    plan = _reindex_plan()
    assert (plan.rebuild, plan.file_count) == (None, 0)
    assert not cfg.lancedb_dir.exists()


def test_the_apply_dialog_plan_on_an_index_with_chunks_counts_the_files(monkeypatch):
    monkeypatch.setattr(Store, "has_chunks", lambda self: True)
    monkeypatch.setattr(
        Store, "get_sources", lambda self, **kw: [{"filename": "a"}, {"filename": "b"}]
    )
    plan = _reindex_plan()
    assert (plan.rebuild, plan.file_count) == (RebuildReason.CHANGES, 2)
