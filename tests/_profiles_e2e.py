"""Isolated worlds, thin surface drivers and the end state for the profiles end-to-end suites."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from litestar.testing import TestClient
from mcp.types import CallToolResult
from pydantic_core import to_jsonable_python
from textual.app import ComposeResult
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Footer
from typer.testing import CliRunner, Result

from lilbee.app.services import set_services
from lilbee.app.settings import setting_sources
from lilbee.cli.app import app as cli_app
from lilbee.cli.app import clear_overrides
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.core.config import Config, cfg
from lilbee.core.config.resolve import PROFILE_FIELDS
from lilbee.core.profile_files import PROFILE_SUFFIX, PROFILES_DIRNAME, profile_folders
from lilbee.core.project_state import STATE_FILE_NAME, read_state
from lilbee.core.system import default_data_dir
from lilbee.data.types import SyncResult
from lilbee.mcp_server import build_mcp_server
from lilbee.server import auth as auth_mod
from lilbee.server.app import create_app
from tests._async_wait import press_widget, wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conftest import REAL_GLOBAL_ROOT, make_mock_services, redirect_global_root

COURT = "Court filings"
COURT_TEXT = (
    '[profile]\nname = "Court filings"\ndescription = "Scanned court PDFs."\n'
    'authors = [{ name = "Jane Doe", github = "janedoe" }]\n'
    "[values]\nchunk_size = 768\nchunk_overlap = 50\ntable_extraction = true\n"
    "enable_ocr = false\n"
)
GERMAN = "Die Bundesregierung hat heute beschlossen, dass die neuen Regeln für alle gelten. " * 8
TRACEBACK = "Traceback (most recent call last)"
PAUSES = 300
# Files that carry a clock or a lock; the end state reads the project state through read_state.
_VOLATILE_NAMES = frozenset({STATE_FILE_NAME})
_VOLATILE_SUFFIXES = (".lock",)
_VOLATILE_DIRS = frozenset({"data", "lancedb", "documents"})

runner = CliRunner()


@dataclass(frozen=True)
class World:
    """One isolated lilbee install: its own home, data root and scratch folders."""

    base: Path

    @property
    def root(self) -> Path:
        return self.base / "project"

    @property
    def home(self) -> Path:
        return self.base / "home"

    @property
    def outbox(self) -> Path:
        return self.base / "outbox"

    @property
    def inbox(self) -> Path:
        return self.base / "inbox"

    @property
    def notes(self) -> Path:
        return self.base / "notes"

    def global_profiles(self) -> Path:
        return default_data_dir() / PROFILES_DIRNAME


def enter(world: World, monkeypatch: pytest.MonkeyPatch, pristine: Config) -> None:
    """Make *world* the live install: cfg from *pristine*, then its root, home and services."""
    for name in Config.model_fields:
        setattr(cfg, name, getattr(pristine, name))
    for folder in (world.root, world.home, world.outbox, world.inbox, world.notes):
        folder.mkdir(parents=True, exist_ok=True)
    redirect_global_root(monkeypatch, world.home)
    monkeypatch.setenv("LILBEE_DATA", str(world.root))
    cfg.data_root = world.root
    cfg.data_dir = world.root / "data"
    cfg.documents_dir = world.root / "documents"
    cfg.lancedb_dir = world.root / "data" / "lancedb"
    cfg.mcp_profiles_enabled = True
    clear_overrides()
    set_services(make_mock_services())
    escaped = [p for _, p in profile_folders(cfg.data_root) if not _under(p, world.base)]
    builtin = [p for p in escaped if p.name == "builtin"]
    assert escaped == builtin, f"a writable profile folder escapes the world: {escaped}"


def _under(path: Path, base: Path) -> bool:
    return path.resolve().is_relative_to(base.resolve())


def _volatile(relative: Path) -> bool:
    parts = relative.parts
    return (
        relative.name in _VOLATILE_NAMES
        or relative.name.endswith(_VOLATILE_SUFFIXES)
        or (len(parts) > 1 and parts[0] == "project" and parts[1] in _VOLATILE_DIRS)
    )


def written_files(world: World) -> dict[str, str]:
    """Every durable file in *world*, by its path under the world, with its text."""
    return {
        path.relative_to(world.base).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(world.base.rglob("*"))
        if path.is_file() and not _volatile(path.relative_to(world.base))
    }


def end_state(world: World, rebuilds: int, answer: Any) -> dict[str, Any]:
    """What a scenario leaves behind: files, cfg, sources, project state, rebuilds, its answer."""
    sources = setting_sources()
    state = read_state(world.root)
    return {
        "files": written_files(world),
        "cfg": {key: to_jsonable_python(getattr(cfg, key)) for key in PROFILE_FIELDS},
        "sources": {key: sources[key].value for key in PROFILE_FIELDS},
        "project_state": {
            "analyzed": state.analyzed_at is not None,
            "tip_dismissed": state.tip_dismissed,
        },
        "rebuilds": rebuilds,
        "answer": answer,
    }


def real_root_listing() -> dict[str, tuple[int, int]]:
    """Size and mtime of every TOML file in the real global root and its profiles folder.

    Lists without reading: the real root holds the owner's own settings.
    """
    if not REAL_GLOBAL_ROOT.is_dir():
        return {}
    found = [
        *REAL_GLOBAL_ROOT.glob(f"*{PROFILE_SUFFIX}"),
        *(REAL_GLOBAL_ROOT / PROFILES_DIRNAME).rglob("*"),
    ]
    return {
        path.relative_to(REAL_GLOBAL_ROOT).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(found)
        if path.is_file()
    }


def record_rebuilds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for the sync every rebuild reaches; each forced sync is recorded, none runs."""
    started: list[str] = []

    async def _sync(force_rebuild: bool = False, quiet: bool = False, **_: Any) -> SyncResult:
        if force_rebuild:
            started.append("sync")
        return SyncResult()

    monkeypatch.setattr("lilbee.data.ingest.sync", _sync)
    return started


def cli(world: World, args: list[str], *, json_mode: bool) -> Result:
    """Run ``lilbee [--json] <args>`` in process, with ``--data-dir`` before any ``--``."""
    prefix = ["--json"] if json_mode else []
    cut = args.index("--") if "--" in args else len(args)
    data_dir = ["--data-dir", str(world.root)]
    result = runner.invoke(cli_app, [*prefix, *args[:cut], *data_dir, *args[cut:]])
    assert TRACEBACK not in result.output, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    return result


def json_of(result: Result) -> Any:
    return json.loads(result.output)


@contextmanager
def http_client() -> Iterator[TestClient[Any]]:
    """A test client for the real server app with the session check off."""
    auth_mod.session_manager.disable()
    try:
        yield TestClient(create_app())
    finally:
        auth_mod.session_manager.cleanup()


def sse_events(body: str) -> list[tuple[str, Any]]:
    """The ``(event, data)`` pairs of an SSE body."""
    events: list[tuple[str, Any]] = []
    for block in body.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields.get("data", "null"))))
    return events


async def mcp_call(tool: str, arguments: dict[str, Any]) -> Any:
    """Call *tool* on a freshly built MCP server and return its result payload."""
    result = await build_mcp_server().call_tool(tool, arguments)
    # call_tool also answers InputRequiredResult, which no lilbee tool asks for
    assert isinstance(result, CallToolResult), result
    assert not result.is_error, result.content
    payload = result.structured_content
    assert payload is not None, result.content
    return payload["result"] if set(payload) == {"result"} else payload


async def in_thread(fn: Callable[[], Any]) -> Any:
    """Run a blocking driver off the test's event loop."""
    return await asyncio.to_thread(fn)


class SettingsHost(LilbeeAppHost):
    """The app host with the Settings screen open."""

    def compose(self) -> ComposeResult:
        yield Footer()

    async def on_mount(self) -> None:
        await self.push_screen(SettingsScreen())


async def until(pilot: Pilot, predicate: Callable[[], bool]) -> bool:
    return await wait_until(pilot, predicate, max_pauses=PAUSES)


async def press(pilot: Pilot, widget: Widget, key: str = "enter") -> None:
    await press_widget(pilot, widget, key, max_pauses=PAUSES)
