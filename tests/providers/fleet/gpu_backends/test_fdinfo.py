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
_MS = 1_000_000
_PDEV_A = "0000:00:02.0"
_PDEV_B = "0000:03:00.0"


def _write_fdinfo(proc: Path, pid: int, fd: int, body: str) -> None:
    d = proc / str(pid) / "fdinfo"
    d.mkdir(parents=True, exist_ok=True)
    (d / str(fd)).write_text(body)


def _body(client_id: int | None, pdev: str | None, render_ns: int) -> str:
    lines = ["pos:\t0", "drm-driver:\ti915"]
    if client_id is not None:
        lines.append(f"drm-client-id:\t{client_id}")
    if pdev is not None:
        lines.append(f"drm-pdev:\t{pdev}")
    lines.append(f"drm-engine-render:\t{render_ns} ns")
    return "\n".join(lines) + "\n"


# A client fd: (pid, fd, client id, pdev, render ns before the window, render ns after).
_Fd = tuple[int, int, int | None, str | None, int, int]


def _utilization(
    monkeypatch: pytest.MonkeyPatch, proc: Path, fds: list[_Fd], pdev: str | None = None
) -> int | None:
    """Run the public reader over a fake /proc whose counters move during the window."""

    def _write(phase: int) -> None:
        for pid, fd, client_id, dev, before, after in fds:
            _write_fdinfo(proc, pid, fd, _body(client_id, dev, (before, after)[phase]))

    clock = iter([_S, _S + 200 * _MS])
    monkeypatch.setattr(fdinfo.time, "monotonic_ns", lambda: next(clock))
    monkeypatch.setattr(fdinfo.time, "sleep", lambda _s: _write(1))
    _write(0)
    return fdinfo.read_drm_util("i915", proc=proc, pdev=pdev)


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
            ({None: {"render": 1000, "video": 500}}, _S),
            ({None: {"render": 1000 + 850_000_000, "video": 500}}, 2 * _S),
        ],
    )
    assert fdinfo.read_drm_util("i915") == 85


def test_read_drm_util_caps_at_100_and_ignores_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    # render counter dropped (client left) -> ignored; copy exceeds wall -> capped.
    _stub_snapshots(
        monkeypatch,
        [
            ({None: {"render": 9_000, "copy": 0}}, _S),
            ({None: {"render": 10, "copy": 3 * _S}}, 2 * _S),
        ],
    )
    assert fdinfo.read_drm_util("i915") == 100


def test_read_drm_util_none_when_no_first_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_snapshots(monkeypatch, [None])
    assert fdinfo.read_drm_util("i915") is None


def test_read_drm_util_none_when_no_second_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_snapshots(monkeypatch, [({None: {"render": 1}}, _S), None])
    assert fdinfo.read_drm_util("i915") is None


def test_read_drm_util_none_when_no_elapsed_time(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_snapshots(monkeypatch, [({None: {"render": 1}}, _S), ({None: {"render": 999}}, _S)])
    assert fdinfo.read_drm_util("i915") is None


def test_read_drm_util_none_when_device_leaves_between_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_snapshots(
        monkeypatch, [({_PDEV_A: {"render": 1}}, _S), ({_PDEV_B: {"render": 9}}, 2 * _S)]
    )
    assert fdinfo.read_drm_util("i915", pdev=_PDEV_A) is None


def test_read_drm_util_none_for_two_devices_without_pdev(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fds: list[_Fd] = [
        (100, 3, 1, _PDEV_A, 0, 120 * _MS),
        (200, 3, 2, _PDEV_B, 0, 40 * _MS),
    ]
    assert _utilization(monkeypatch, tmp_path, fds) is None


def test_read_drm_util_none_when_second_snapshot_has_two_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_snapshots(
        monkeypatch,
        [
            ({_PDEV_A: {"render": 1}}, _S),
            ({_PDEV_A: {"render": 9}, _PDEV_B: {"render": 9}}, 2 * _S),
        ],
    )
    assert fdinfo.read_drm_util("i915") is None


def test_read_drm_util_with_pdev_reads_one_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fds: list[_Fd] = [
        (100, 3, 1, _PDEV_A, 0, 120 * _MS),
        (200, 3, 2, _PDEV_B, 0, 40 * _MS),
    ]
    assert _utilization(monkeypatch, tmp_path, fds, pdev=_PDEV_A) == 60


def test_read_drm_util_unknown_pdev_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fds: list[_Fd] = [(100, 3, 1, _PDEV_A, 0, 120 * _MS)]
    assert _utilization(monkeypatch, tmp_path, fds, pdev=_PDEV_B) is None


# ---------------------------------------------------------------------------
# _snapshot: /proc scanning (tmp_path fixtures)
# ---------------------------------------------------------------------------

_I915_FD = "drm-driver:\ti915\ndrm-engine-render:\t500 ns\ndrm-engine-video:\t0 ns\n"


def test_snapshot_sums_engines_across_clients(tmp_path: Path) -> None:
    _write_fdinfo(tmp_path, 100, 3, _I915_FD)
    _write_fdinfo(tmp_path, 200, 4, "drm-driver:\ti915\ndrm-engine-render:\t250 ns\n")
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None
    devices, wall = result
    assert devices[None]["render"] == 750
    assert isinstance(wall, int)


def test_snapshot_counts_a_duplicated_fd_once(tmp_path: Path) -> None:
    _write_fdinfo(tmp_path, 100, 3, _body(7, _PDEV_A, 500))
    _write_fdinfo(tmp_path, 100, 4, _body(7, _PDEV_A, 500))
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None and result[0] == {_PDEV_A: {"render": 500}}


def test_snapshot_counts_an_inherited_fd_once_across_pids(tmp_path: Path) -> None:
    _write_fdinfo(tmp_path, 100, 3, _body(7, _PDEV_A, 500))
    _write_fdinfo(tmp_path, 200, 5, _body(7, _PDEV_A, 500))
    _write_fdinfo(tmp_path, 300, 3, _body(8, _PDEV_A, 100))
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None and result[0] == {_PDEV_A: {"render": 600}}


def test_snapshot_counts_a_client_without_an_id_once_per_fd(tmp_path: Path) -> None:
    _write_fdinfo(tmp_path, 100, 3, _body(None, _PDEV_A, 500))
    _write_fdinfo(tmp_path, 100, 4, _body(None, _PDEV_A, 500))
    _write_fdinfo(tmp_path, 200, 3, _body(None, _PDEV_A, 500))
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None and result[0] == {_PDEV_A: {"render": 1500}}


def test_same_client_id_on_two_devices_is_two_clients(tmp_path: Path) -> None:
    _write_fdinfo(tmp_path, 100, 3, _body(1, _PDEV_A, 500))
    _write_fdinfo(tmp_path, 200, 3, _body(1, _PDEV_B, 200))
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None
    assert result[0] == {_PDEV_A: {"render": 500}, _PDEV_B: {"render": 200}}


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
    assert result is not None and result[0][None]["render"] == 500


def test_snapshot_skips_non_pid_entries(tmp_path: Path) -> None:
    (tmp_path / "cpuinfo").write_text("x")  # non-numeric /proc entry
    _write_fdinfo(tmp_path, 200, 4, _I915_FD)
    assert fdinfo._snapshot("i915", tmp_path) is not None


def test_snapshot_skips_unreadable_entry(tmp_path: Path) -> None:
    # An fdinfo entry that is a directory -> read_text raises OSError -> skipped.
    (tmp_path / "100" / "fdinfo" / "3").mkdir(parents=True)
    _write_fdinfo(tmp_path, 200, 4, _I915_FD)
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None and result[0][None]["render"] == 500


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_driver_matches() -> None:
    assert fdinfo._driver_matches("drm-driver:\ti915\n", "i915") is True
    assert fdinfo._driver_matches("drm-driver:\tamdgpu\n", "i915") is False
    assert fdinfo._driver_matches("pos:\t0\n", "i915") is False


def test_value_reads_the_named_line_only() -> None:
    text = "drm-client-idx:\t9\ndrm-client-id:\t 42 \n"
    assert fdinfo._value(text, "drm-client-id") == "42"
    assert fdinfo._value(text, "drm-pdev") is None


def test_parse_engine_line() -> None:
    assert fdinfo._parse_engine_line("drm-engine-render:\t123 ns") == ("render", 123)
    assert fdinfo._parse_engine_line("drm-engine-x:\tabc ns") == ("x", None)
    assert fdinfo._parse_engine_line("drm-engine-y:") == ("y", None)


def test_snapshot_skips_an_engine_line_with_no_number(tmp_path: Path) -> None:
    _write_fdinfo(
        tmp_path, 100, 3, "drm-driver:\ti915\ndrm-engine-copy:\tabc ns\ndrm-engine-render:\t5 ns\n"
    )
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None and result[0] == {None: {"render": 5}}


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(("dup-first", "dup-second", "other"), id="duplicate-before-distinct"),
        pytest.param(("other", "dup-first", "dup-second"), id="distinct-before-duplicate"),
    ],
)
def test_snapshot_keeps_reading_a_pid_after_a_duplicate_fd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, order: tuple[str, ...]
) -> None:
    clients = {
        "dup-first": fdinfo._Client(_PDEV_A, "id:7", {"render": 500}),
        "dup-second": fdinfo._Client(_PDEV_A, "id:7", {"render": 500}),
        "other": fdinfo._Client(_PDEV_A, "id:8", {"render": 100}),
    }
    (tmp_path / "100").mkdir()
    monkeypatch.setattr(fdinfo, "_pid_clients", lambda _pid, _driver: [clients[n] for n in order])
    result = fdinfo._snapshot("i915", tmp_path)
    assert result is not None and result[0] == {_PDEV_A: {"render": 600}}
