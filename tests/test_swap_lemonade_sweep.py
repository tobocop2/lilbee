"""R10: the orphan sweep finds a Lemonade child by its role folder, not by a member port."""

from __future__ import annotations

from pathlib import Path

import psutil
import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import swap_manager as sm
from lilbee.providers.fleet.engines import lemonade
from lilbee.providers.roles import WorkerRole
from tests._fake_process import FakeProcess, UnreadableCmdline, process_table


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    return lemonade.role_dir(WorkerRole.CHAT, 0)


def _table(monkeypatch: pytest.MonkeyPatch, procs: list[FakeProcess]) -> None:
    monkeypatch.setattr(sm.psutil, "process_iter", process_table(procs))


def test_r10_a_lemond_that_runs_from_a_role_folder_is_found(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = FakeProcess(
        "lemond", ["lemond", "--port", "5800", str(root / "cache"), str(root / "config")]
    )
    other = FakeProcess("lemond", ["lemond", "--port", "8001", "/home/user/.cache/lemonade"])
    _table(monkeypatch, [parent, other])
    assert sm._find_orphan_servers((5800,)) == [parent]


def test_r10_the_data_folder_itself_is_found(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orphan = FakeProcess("lemond", ["lemond", str(lemonade.data_root())])
    _table(monkeypatch, [orphan])
    assert sm._find_orphan_servers((5800,)) == [orphan]


def test_r10_a_sibling_folder_that_shares_the_prefix_is_spared(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sibling = Path(f"{lemonade.data_root()}-other") / "cache"
    _table(monkeypatch, [FakeProcess("lemond", ["lemond", str(sibling)])])
    assert sm._find_orphan_servers((5800,)) == []


def test_r10_a_lemonade_process_below_a_live_swap_is_spared(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = FakeProcess("lemond", ["lemond", str(root / "cache")], ["llama-swap"])
    _table(monkeypatch, [live])
    assert sm._find_orphan_servers((5800,)) == []


def test_r10_a_process_that_cannot_be_read_is_skipped(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _table(monkeypatch, [UnreadableCmdline("lemond", [])])
    assert sm._find_orphan_servers((5800,)) == []


def test_r10_an_unreadable_parent_chain_does_not_spare_the_process(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orphan = FakeProcess("lemond", ["lemond", str(root / "cache")], [psutil.NoSuchProcess(1)])
    _table(monkeypatch, [orphan])
    assert sm._find_orphan_servers((5800,)) == [orphan]


def test_r10_a_state_with_no_recorded_ports_scans_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the process table must not be scanned")

    monkeypatch.setattr(sm.psutil, "process_iter", _forbidden)
    assert sm._find_orphan_servers(()) == []
