"""An in-place write to config.toml under a second old is not taken; an atomic replace is."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from lilbee.app import settings as app_settings
from lilbee.app.services import set_services
from lilbee.core import settings as persistent_settings
from lilbee.core.config import CONFIG_FILE_NAME, cfg
from tests.conftest import make_mock_services

_MTIME_NS = 1_700_000_000_000_000_000
_SECOND_NS = 1_000_000_000


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    root = tmp_path / "project"
    root.mkdir()
    cfg.data_root = root
    cfg.data_dir = root / "data"
    cfg.documents_dir = root / "documents"
    cfg.lancedb_dir = cfg.data_dir / "lancedb"
    monkeypatch.setenv("LILBEE_DATA", str(root))
    monkeypatch.delenv("LILBEE_SKIP_TOML_CONFIG", raising=False)
    set_services(make_mock_services())
    yield root
    set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _config_path() -> Path:
    return cfg.data_root / CONFIG_FILE_NAME


def _stamp_file(tick: int) -> None:
    stamp = _MTIME_NS + tick * _SECOND_NS
    os.utime(_config_path(), ns=(stamp, stamp))


def _write(key: str, value: Any, tick: int) -> None:
    """An atomic write, as lilbee and `lilbee set` make it: a new inode."""
    persistent_settings.set_value(cfg.data_root, key, value)
    _stamp_file(tick)


def _truncate_in_place(tick: int) -> None:
    """An editor's truncate: the same inode, no keys, its rewrite not landed."""
    _config_path().write_text("", encoding="utf-8")
    _stamp_file(tick)


def _clock_at(monkeypatch: pytest.MonkeyPatch, tick: int, offset_ns: int) -> None:
    """Set the clock *offset_ns* after the mtime a write at *tick* gets."""
    now = _MTIME_NS + tick * _SECOND_NS + offset_ns
    monkeypatch.setattr(persistent_settings, "_now_ns", lambda: now, raising=False)


def test_a_truncated_file_mid_rewrite_keeps_the_values(monkeypatch):
    _write("top_k", 21, tick=1)
    reading = app_settings.read_config_state()
    cfg.top_k = 21
    _truncate_in_place(tick=2)
    _clock_at(monkeypatch, 2, _SECOND_NS // 10)
    after, result = app_settings.adopt_external_changes(reading)
    assert cfg.top_k == 21
    assert result is None
    assert after.stamp == reading.stamp  # the next check reads the file again


def test_an_atomic_replace_under_a_second_old_is_taken_at_once(monkeypatch):
    _write("top_k", 21, tick=1)
    reading = app_settings.read_config_state()
    cfg.top_k = 21
    _write("top_k", 13, tick=2)
    _clock_at(monkeypatch, 2, 0)
    app_settings.adopt_external_changes(reading)
    assert cfg.top_k == 13


def test_the_same_file_is_taken_once_it_is_a_second_old(monkeypatch):
    _write("top_k", 21, tick=1)
    reading = app_settings.read_config_state()
    cfg.top_k = 21
    _truncate_in_place(tick=2)
    _clock_at(monkeypatch, 2, _SECOND_NS // 10)
    reading, _ = app_settings.adopt_external_changes(reading)
    assert cfg.top_k == 21
    _clock_at(monkeypatch, 2, _SECOND_NS)
    app_settings.adopt_external_changes(reading)
    assert cfg.top_k != 21


def test_a_finished_in_place_rewrite_is_taken_after_the_window(monkeypatch):
    _write("top_k", 21, tick=1)
    reading = app_settings.read_config_state()
    cfg.top_k = 21
    _config_path().write_text("top_k = 13\n", encoding="utf-8")
    _stamp_file(2)
    _clock_at(monkeypatch, 2, 2 * _SECOND_NS)
    app_settings.adopt_external_changes(reading)
    assert cfg.top_k == 13


def test_read_settled_with_no_last_stamp_reads_a_young_file(monkeypatch):
    _write("top_k", 21, tick=1)
    _clock_at(monkeypatch, 1, 0)
    stamp, values = persistent_settings.read_settled(cfg.data_root, None)
    assert stamp is not None
    assert values is not None
    assert values["top_k"] == 21


def test_read_settled_skips_a_young_file_of_the_same_inode(monkeypatch):
    _write("top_k", 21, tick=1)
    last = persistent_settings.overlay_stamp(cfg.data_root)
    _truncate_in_place(tick=2)
    _clock_at(monkeypatch, 2, _SECOND_NS - 1)
    assert persistent_settings.read_settled(cfg.data_root, last) == (None, None)


def test_read_settled_reads_a_young_file_of_a_new_inode(monkeypatch):
    _write("top_k", 21, tick=1)
    last = persistent_settings.overlay_stamp(cfg.data_root)
    _write("top_k", 13, tick=2)
    _clock_at(monkeypatch, 2, 0)
    stamp, values = persistent_settings.read_settled(cfg.data_root, last)
    assert stamp is not None
    assert values is not None
    assert values["top_k"] == 13


def test_read_settled_reads_a_file_dated_in_the_future(monkeypatch):
    _write("top_k", 21, tick=1)
    last = persistent_settings.overlay_stamp(cfg.data_root)
    _truncate_in_place(tick=2)
    _clock_at(monkeypatch, 2, -_SECOND_NS)
    stamp, _values = persistent_settings.read_settled(cfg.data_root, last)
    assert stamp is not None


def test_read_settled_without_a_file_reads_defaults(monkeypatch):
    _clock_at(monkeypatch, 1, 0)
    stamp, values = persistent_settings.read_settled(cfg.data_root, None)
    assert stamp is None
    assert values is not None


def test_the_startup_overlay_takes_a_file_written_just_now(monkeypatch):
    _write("top_k", 21, tick=1)
    _clock_at(monkeypatch, 1, 0)
    persistent_settings.overlay_persisted_settings(cfg.data_root)
    assert cfg.top_k == 21
