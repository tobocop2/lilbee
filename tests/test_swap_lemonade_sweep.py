"""R10: the orphan sweep finds a Lemonade child by its role folder, not by a member port."""

from __future__ import annotations

from pathlib import Path

import psutil
import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import swap_manager as sm
from lilbee.providers.fleet.engines import lemonade
from lilbee.providers.roles import WorkerRole


class _Proc:
    def __init__(self, name: str, cmdline: list[str], parents: list[str] | None = None) -> None:
        self._name = name
        self._cmdline = cmdline
        self._parents = parents or []

    def name(self) -> str:
        return self._name

    def cmdline(self) -> list[str]:
        return self._cmdline

    def parents(self) -> list[_Proc]:
        return [_Proc(name, []) for name in self._parents]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    return lemonade.role_dir(WorkerRole.CHAT, 0)


def _table(monkeypatch: pytest.MonkeyPatch, procs: list[_Proc]) -> None:
    monkeypatch.setattr(
        sm, "_processes_named", lambda needle: (p for p in procs if needle in p.name())
    )


def test_r10_a_lemonade_child_on_a_port_the_fleet_did_not_pick_is_found(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = _Proc(
        "llama-server", ["llama-server", "-m", str(root / "view/chat/m.gguf"), "--port", "8001"]
    )
    parent = _Proc(
        "lemond", ["lemond", "--port", "5800", str(root / "cache"), str(root / "config")]
    )
    other = _Proc("llama-server", ["llama-server", "-m", "/x.gguf", "--port", "8001"])
    _table(monkeypatch, [child, parent, other])
    assert sm._find_orphan_servers((5800,)) == [parent, child]


def test_r10_a_lemonade_process_below_a_live_swap_is_spared(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = _Proc("lemond", ["lemond", str(root / "cache")], parents=["lemond", "llama-swap"])
    _table(monkeypatch, [live])
    assert sm._find_orphan_servers(()) == []


def test_r10_a_process_that_cannot_be_read_is_skipped(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Gone(_Proc):
        def cmdline(self) -> list[str]:
            raise psutil.AccessDenied()

    _table(monkeypatch, [_Gone("lemond", [])])
    assert sm._find_orphan_servers(()) == []


def test_r10_an_unreadable_parent_chain_does_not_spare_the_process(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Hidden(_Proc):
        def parents(self) -> list[_Proc]:
            raise psutil.NoSuchProcess(1)

    orphan = _Hidden("lemond", ["lemond", str(root / "cache")])
    _table(monkeypatch, [orphan])
    assert sm._find_orphan_servers(()) == [orphan]
