"""Tests for the generic DRM fdinfo utilization reader.

The fdinfo fixtures below mirror the kernel's drm-usage-stats format. Live capture
on a modern-kernel box (kernel 6.2+, e.g. an amdgpu or i915 GPU under load)
confirms the real counters; these tests pin the parsing and delta math.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.providers.fleet.gpu_backends import fdinfo

_S = 1_000_000_000  # one second in nanoseconds


def _write_fdinfo(proc: Path, pid: int, fd: int, body: str) -> None:
    d = proc / str(pid) / "fdinfo"
    d.mkdir(parents=True, exist_ok=True)
    (d / str(fd)).write_text(body)


# ---------------------------------------------------------------------------
# read_drm_util: delta math (snapshots stubbed)
# ---------------------------------------------------------------------------


def _stub_snapshots(monkeypatch: pytest.MonkeyPatch, snaps: list[object]) -> None:
    monkeypatch.setattr(fdinfo, "_snapshot", lambda *_a: snaps.pop(0))
    monkeypatch.setattr(fdinfo.time, "sleep", lambda _s: None)


def test_read_drm_util_busiest_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_snapshots(
        monkeypatch,
        [
            ({"render": 1000, "video": 500}, _S),
            ({"render": 1000 + 850_000_000, "video": 500}, 2 * _S),
        ],
    )
    assert fdinfo.read_drm_util("i915") == 85


def test_read_drm_util_caps_at_100_and_ignores_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    # render counter dropped (client left) -> ignored; copy exceeds wall -> capped.
    _stub_snapshots(
        monkeypatch,
        [({"render": 9_000, "copy": 0}, _S), ({"render": 10, "copy": 3 * _S}, 2 * _S)],
    )
    assert fdinfo.read_drm_util("i915") == 100


def test_read_drm_util_none_when_no_first_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_snapshots(monkeypatch, [None])
    assert fdinfo.read_drm_util("i915") is None


def test_read_drm_util_none_when_no_second_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_snapshots(monkeypatch, [({"render": 1}, _S), None])
    assert fdinfo.read_drm_util("i915") is None


def test_read_drm_util_none_when_no_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_snapshots(monkeypatch, [({"render": 1}, _S), ({"render": 999}, _S)])
    assert fdinfo.read_drm_util("i915") is None


# ---------------------------------------------------------------------------
# _snapshot: /proc scanning (tmp_path fixtures)
# ---------------------------------------------------------------------------

_I915_FD = "drm-driver:\ti915\ndrm-engine-render:\t500 ns\ndrm-engine-video:\t0 ns\n"


def test_snapshot_sums_engines_across_clients(tmp_path: Path) -> None:
    _write_fdinfo(tmp_path, 100, 3, _I915_FD)
    _write_fdinfo(tmp_path, 200, 4, "drm-driver:\ti915\ndrm-engine-render:\t250 ns\n")
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None
    totals, wall = result
    assert totals["render"] == 750
    assert isinstance(wall, int)


def test_snapshot_skips_other_driver(tmp_path: Path) -> None:
    _write_fdinfo(tmp_path, 100, 3, "drm-driver:\tamdgpu\ndrm-engine-gfx:\t500 ns\n")
    assert fdinfo._snapshot("i915", tmp_path) is None


def test_snapshot_none_when_no_engine_lines(tmp_path: Path) -> None:
    _write_fdinfo(tmp_path, 100, 3, "drm-driver:\ti915\ndrm-pdev:\t0000:00:02.0\n")
    assert fdinfo._snapshot("i915", tmp_path) is None


def test_snapshot_none_when_proc_unreadable(tmp_path: Path) -> None:
    assert fdinfo._snapshot("i915", tmp_path / "does-not-exist") is None


def test_snapshot_skips_pid_without_fdinfo(tmp_path: Path) -> None:
    (tmp_path / "100").mkdir()  # a pid dir with no fdinfo subdir
    _write_fdinfo(tmp_path, 200, 4, _I915_FD)
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None and result[0]["render"] == 500


def test_snapshot_skips_non_pid_entries(tmp_path: Path) -> None:
    (tmp_path / "cpuinfo").write_text("x")  # non-numeric /proc entry
    _write_fdinfo(tmp_path, 200, 4, _I915_FD)
    assert fdinfo._snapshot("i915", tmp_path) is not None


def test_snapshot_skips_unreadable_entry(tmp_path: Path) -> None:
    # An fdinfo entry that is a directory -> read_text raises OSError -> skipped.
    (tmp_path / "100" / "fdinfo" / "3").mkdir(parents=True)
    _write_fdinfo(tmp_path, 200, 4, _I915_FD)
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None and result[0]["render"] == 500


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_driver_matches() -> None:
    assert fdinfo._driver_matches("drm-driver:\ti915\n", "i915") is True
    assert fdinfo._driver_matches("drm-driver:\tamdgpu\n", "i915") is False
    assert fdinfo._driver_matches("pos:\t0\n", "i915") is False


def test_parse_engine_line() -> None:
    assert fdinfo._parse_engine_line("drm-engine-render:\t123 ns") == ("render", 123)
    assert fdinfo._parse_engine_line("drm-engine-x:\tabc ns") == ("x", None)
    assert fdinfo._parse_engine_line("drm-engine-y:") == ("y", None)
