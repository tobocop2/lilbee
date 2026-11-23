"""The explanation of a missing server reads installed state: it installs and downloads nothing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.app.engine_choice import BookReading, EngineBook
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import binary, first_use
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.provider import FleetProvider
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import engines_for_all_roles
from tests.conformance.fake_adapter import FakeAdapter

_FIRST_USE_TOOL = first_use.first_use_tool


class _Calls:
    """What a host with no llama-server was asked to do: installs of the pack and PATH scans."""

    def __init__(self) -> None:
        self.installs: list[str] = []
        self.scans: list[str] = []


@pytest.fixture
def no_server(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    """A host with no llama-server anywhere, that counts each install and each PATH scan."""
    calls = _Calls()
    monkeypatch.setattr(cfg, "llama_server_path", "")
    monkeypatch.setattr(binary, "_bundled_tool", lambda tool: None)
    monkeypatch.setattr(first_use, "installed_tool", lambda tool: None)
    monkeypatch.setattr(first_use, "_refusals", first_use.RefusedInstalls())
    monkeypatch.setattr(first_use, "pack_lock", lambda: (SimpleNamespace(version="1"), "cell"))

    def scan(tool: str, *_a: object, **_k: object) -> None:
        calls.scans.append(tool)

    def install(tool: str) -> None:
        calls.installs.append(tool)

    monkeypatch.setattr("shutil.which", scan)
    monkeypatch.setattr(first_use, "first_use_tool", install)
    monkeypatch.setattr(first_use, "install_pack", lambda: calls.installs.append("pack"))
    return calls


def _book(monkeypatch: pytest.MonkeyPatch, adapters: dict[WorkerRole, object]) -> None:
    monkeypatch.setattr(EngineBook, "read", lambda self: BookReading(adapters, {}))  # type: ignore[arg-type]


@pytest.fixture(params=["llama only", "llama beside another engine"])
def book(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    adapters: dict[WorkerRole, object] = dict(engines_for_all_roles(LLAMA))
    if request.param != "llama only":
        adapters[WorkerRole.CHAT] = FakeAdapter()
    _book(monkeypatch, adapters)
    return str(request.param)


@pytest.fixture
def pack_only(no_server: _Calls, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Calls:
    """The host's only llama-server is the installed pack."""
    server = tmp_path / "llama-server"
    server.write_bytes(b"")
    monkeypatch.setattr(first_use, "installed_tool", lambda tool: tmp_path / tool)
    return no_server


@pytest.fixture
def disk_full(no_server: _Calls, monkeypatch: pytest.MonkeyPatch) -> _Calls:
    """A host whose first-use install of the pack was refused because the disk is full."""
    monkeypatch.setattr(first_use, "first_use_tool", _FIRST_USE_TOOL)
    key = (str(first_use._store().root), "1", "cell")
    first_use._refusals.remember(key, OSError("No space left on device"))
    return no_server


class TestTheExplanationInstallsNothing:
    def test_the_reason_names_the_missing_server_and_installs_nothing(
        self, no_server: _Calls, book: str
    ) -> None:
        reason = FleetProvider()._unusable_engine_reason()

        assert reason is not None
        assert "llama-server binary not found" in reason
        assert no_server.installs == []

    @pytest.mark.parametrize("role", list(WorkerRole))
    def test_the_error_of_a_role_with_no_server_installs_nothing(
        self, no_server: _Calls, book: str, role: WorkerRole
    ) -> None:
        with pytest.raises(ProviderError, match="llama-server binary not found"):
            FleetProvider()._raise_no_clients(role)

        assert no_server.installs == []

    def test_the_path_is_scanned_once_for_the_server_the_reason_names(
        self, no_server: _Calls, book: str
    ) -> None:
        FleetProvider()._unusable_engine_reason()

        assert no_server.scans == ["llama-server"]


class TestTheExplanationReadsThePack:
    def test_a_host_with_only_the_pack_has_no_engine_problem(
        self, pack_only: _Calls, book: str
    ) -> None:
        assert FleetProvider()._unusable_engine_reason() is None
        assert pack_only.installs == []

    @pytest.mark.parametrize("role", list(WorkerRole))
    def test_the_error_of_a_role_does_not_say_the_pack_server_is_missing(
        self, pack_only: _Calls, book: str, role: WorkerRole
    ) -> None:
        with pytest.raises(ProviderError) as error:
            FleetProvider()._raise_no_clients(role)

        assert "binary not found" not in str(error.value)

    def test_the_pack_server_is_ready(self, pack_only: _Calls) -> None:
        assert binary.llama_server_ready()


class TestARememberedRefusal:
    def test_the_reason_names_the_refusal_and_installs_nothing(
        self, disk_full: _Calls, book: str
    ) -> None:
        reason = FleetProvider()._unusable_engine_reason()

        assert reason is not None
        assert "llama-server binary not found" in reason
        assert "No space left on device" in reason
        assert disk_full.installs == []

    def test_the_server_is_not_ready(self, disk_full: _Calls) -> None:
        assert not binary.llama_server_ready()
        assert disk_full.installs == []

    def test_a_launch_names_the_same_refusal(self, disk_full: _Calls) -> None:
        with pytest.raises(ProviderError, match="No space left on device"):
            binary.resolve_llama_server()

        assert disk_full.installs == []
