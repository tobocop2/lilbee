"""A remove that cannot take the store lock or rename the build is a refusal on every surface."""

from __future__ import annotations

import errno
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock
from typer.testing import CliRunner

from lilbee.app import engine_service
from lilbee.app.engine_service import EngineRow, EngineState
from lilbee.cli import app as cli_app
from lilbee.cli.tui.engine_commands import run_engine_command
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog
from lilbee.core.security import FileLockTimeoutError, FileLockUnavailableError
from lilbee.providers.fleet import engine_store
from lilbee.providers.fleet.engine_store import EngineStore, engines_root
from tests._async_wait import wait_until
from tests.cli.test_engine_builds import install, models_dir  # noqa: F401
from tests.test_tui_settings_engines import (
    _config,  # noqa: F401
    _open_engines,
    _Rows,
    _select_row,
    _SettingsApp,
    _toasts,
)

BUSY = "Another lilbee process is using the engine store. Run the command again."
REFUSED_LOCK = "the lock on fx cannot be taken (Permission denied)"
MOVE = "lilbee cannot move the engine folder"
_real_rename = Path.rename


def _lock_timeout(monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the store lock time out, as it does while another lilbee holds it."""

    _fail_lock(monkeypatch, FileLockTimeoutError("another lilbee process held the lock"))
    return BUSY


def _fail_lock(monkeypatch: pytest.MonkeyPatch, error: FileLockUnavailableError) -> None:
    @contextmanager
    def failing(_path: Path, _timeout_s: float) -> Iterator[None]:
        raise error
        yield

    monkeypatch.setattr(engine_store, "file_lock", failing)


def _rename_fails(monkeypatch: pytest.MonkeyPatch) -> str:
    """Make the rename of the fx build fail with a fault that is no denied write."""

    def rename(self: Path, target: Any) -> Path:
        if self.parent.name == "fx" and self.name == "1-cpu":
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        return _real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    return MOVE


FAILURES = [pytest.param(_lock_timeout, id="lock"), pytest.param(_rename_fails, id="rename")]
Inject = Callable[[pytest.MonkeyPatch], str]


def _kept(models_dir: Path) -> None:  # noqa: F811
    assert len(EngineStore(engines_root(models_dir)).installed()) == 1


@pytest.mark.parametrize("inject", FAILURES)
def test_the_service_refuses_instead_of_raising_the_os_error(
    models_dir: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    inject: Inject,
) -> None:
    install(models_dir)
    text = inject(monkeypatch)
    with pytest.raises(engine_service.EngineRefusedError) as raised:
        engine_service.remove_build("fx", "1", "cpu")
    assert text in str(raised.value)
    _kept(models_dir)


def test_a_lock_the_system_refuses_shows_its_own_text_not_the_busy_text(
    models_dir: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(models_dir)
    _fail_lock(monkeypatch, FileLockUnavailableError(REFUSED_LOCK))
    with pytest.raises(engine_service.EngineRefusedError) as raised:
        engine_service.remove_build("fx", "1", "cpu")
    assert str(raised.value) == REFUSED_LOCK
    _kept(models_dir)


def test_a_lock_held_by_another_holder_past_the_wait_shows_the_busy_text(
    models_dir: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(models_dir)
    real = engine_store.file_lock
    monkeypatch.setattr(engine_store, "file_lock", lambda path, _wait: real(path, 0.2))
    with FileLock(engines_root(models_dir) / engine_store.LOCK_DIR / "fx.lock"):
        with pytest.raises(engine_service.EngineRefusedError) as raised:
            engine_service.remove_build("fx", "1", "cpu")
    assert str(raised.value) == BUSY
    _kept(models_dir)


@pytest.mark.parametrize("inject", FAILURES)
def test_the_cli_prints_the_refusal_and_exits_one(
    models_dir: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    inject: Inject,
) -> None:
    install(models_dir)
    text = inject(monkeypatch)
    result = CliRunner().invoke(cli_app, ["--json", "engine", "remove", "fx", "1", "cpu"])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert text in json.loads(result.output.strip().splitlines()[-1])["error"]
    _kept(models_dir)


@pytest.mark.parametrize("inject", FAILURES)
def test_the_slash_command_returns_the_refusal_as_its_text(
    models_dir: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    inject: Inject,
) -> None:
    install(models_dir)
    text = inject(monkeypatch)
    result = run_engine_command("remove fx 1 cpu")
    assert not result.ok
    assert text in result.text
    _kept(models_dir)


@pytest.mark.parametrize("inject", FAILURES)
async def test_the_tab_shows_the_refusal_as_a_toast_and_keeps_running(
    models_dir: Path,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    inject: Inject,
) -> None:
    install(models_dir)
    row = EngineRow("fx", EngineState.INSTALLED, "1", "cpu")
    monkeypatch.setattr(engine_service, "engine_rows", _Rows([row]))
    monkeypatch.setattr(engine_service, "offered_engines", lambda: {})
    text = inject(monkeypatch)
    app = _SettingsApp()
    async with app.run_test(size=(140, 40)) as pilot:
        table = await _open_engines(app, pilot, rows=1)
        await _select_row(pilot, table, "fx")
        assert await wait_until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
        await pilot.press("y")
        assert await wait_until(pilot, lambda: any(text in toast for toast in _toasts()))
        assert app.is_running
    _kept(models_dir)
