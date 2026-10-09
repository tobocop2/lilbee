"""Every profiles and analyze scenario on every surface, checked against the app layer.

Each test runs one scenario twice from the same seed: once through the app layer, once
through a surface. The end states must match, and so must what the surface reports.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any
from unittest import mock
from urllib.parse import quote

import pytest
from litestar.testing import TestClient
from pydantic_core import to_jsonable_python
from textual.notifications import Notification
from textual.pilot import Pilot
from textual.widgets import DataTable, Input, OptionList, Select, Static
from typer.testing import Result

from lilbee.app import analyze as analyze_mod
from lilbee.app import profiles
from lilbee.app.analyze import (
    NOTES_AND_MARKDOWN,
    AnalyzeReport,
    AnalyzeRequest,
    derived_name,
    hide_tip,
    run_analysis,
    tip_state,
)
from lilbee.app.profiles import ProfileEffect
from lilbee.app.settings import apply_settings_update, reset_settings, setting_sources
from lilbee.cli.commands._shared import REBUILD_HINT
from lilbee.cli.commands.analyze import TIP_HIDDEN_MESSAGE
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import LilbeeApp
from lilbee.cli.tui.screens.analyze_report import AnalyzeReportScreen
from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.screens.profile_dialogs import (
    ApplyProfileDialog,
    ProfilePathDialog,
    SaveProfileDialog,
    value_text,
)
from lilbee.cli.tui.screens.profile_library import LibraryAction, ProfileLibrary
from lilbee.cli.tui.screens.profile_tab import ProfileAction as TabAction
from lilbee.cli.tui.screens.profile_tab import ProfileTab
from lilbee.cli.tui.screens.settings import SettingsScreen
from lilbee.cli.tui.screens.settings_widgets import (
    ROW_ID_PREFIX,
    active_profile_name,
    source_pill,
)
from lilbee.cli.tui.widgets.chat_input import ChatInput
from lilbee.cli.tui.widgets.confirm_dialog import ConfirmDialog, ConfirmPill
from lilbee.cli.tui.widgets.model_bar import ModelBar
from lilbee.core.config import cfg
from lilbee.core.config.enums import SettingSource
from lilbee.core.profile_files import PROFILES_DIRNAME, ProfileFolder, ProfileStore
from lilbee.mcp_server import ProfileAction as McpAction
from lilbee.mcp_server import profile_apply as mcp_profile_apply
from lilbee.server.routes.profiles import profiles_apply_route
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import await_chat, ready_services
from tests._profiles_e2e import (
    COURT,
    COURT_TEXT,
    GERMAN,
    SettingsHost,
    World,
    cli,
    end_state,
    enter,
    http_client,
    in_thread,
    json_of,
    mcp_call,
    press,
    real_root_listing,
    record_rebuilds,
    sse_events,
    until,
)


class Scenario(StrEnum):
    NEW = "new"
    SAVE_AS = "save_as"
    SAVE_AS_PROJECT = "save_as_project"
    APPLY = "apply"
    APPLY_REINDEX = "apply_reindex"
    UPDATE = "update"
    DISCARD = "discard"
    DUPLICATE = "duplicate"
    RENAME = "rename"
    DELETE = "delete"
    EXPORT = "export"
    IMPORT = "import"
    VALIDATE = "validate"
    SHOW = "show"
    LIST = "list"
    DIFF = "diff"
    ANALYZE_REPORT = "analyze_report"
    ANALYZE_SAVE = "analyze_save"
    ANALYZE_SAVE_GLOBAL = "analyze_save_global"
    ANALYZE_APPLY = "analyze_apply"
    DISMISS_TIP = "dismiss_tip"
    RESET = "reset"
    SOURCES = "sources"


class Surface(StrEnum):
    CLI = "cli"
    CLI_JSON = "cli_json"
    HTTP = "http"
    MCP = "mcp"
    TUI = "tui"


class Aspect(StrEnum):
    """What a cell compares: what the scenario leaves behind, or what the surface reports."""

    END_STATE = "end_state"
    REPORT = "report"


@dataclass(frozen=True)
class Outcome:
    """A scenario's answer (for a query) and report (reindex flag and warnings, for a change)."""

    answer: Any = None
    report: Any = None


NOT_OFFERED: dict[tuple[Surface, Scenario], str] = {
    (Surface.TUI, Scenario.NEW): "neither the Profile tab nor the library writes a template",
    (Surface.TUI, Scenario.VALIDATE): "the TUI has no validate action; import refuses instead",
    (Surface.TUI, Scenario.ANALYZE_SAVE_GLOBAL): "the report's Save saves to the default folder",
}
# Apply with reindex is one call on the CLI and the TUI, and two calls over HTTP and MCP.
TWO_STEP: dict[Surface, str] = {
    Surface.HTTP: "POST /api/profiles/{name}/apply, then POST /api/sync with force_rebuild",
    Surface.MCP: "profile_apply, then sync(force_rebuild=True)",
}
REPORTED = frozenset(
    {
        Scenario.APPLY,
        Scenario.APPLY_REINDEX,
        Scenario.DISCARD,
        Scenario.RESET,
        Scenario.ANALYZE_APPLY,
    }
)
COPY = "Court copy"
RENAMED = "Court renamed"
SAVED = "My set"
FRESH = "Fresh start"
VISION = "acme/vision-model"
SHARED_NAME = "Shared set"
SHARED_FILE = "shared-set.toml"
SHARED_TEXT = f'[profile]\nname = "{SHARED_NAME}"\n[values]\ntop_k = 12\nhyde = true\n'
BROKEN_FILE = "broken.toml"
BROKEN_TEXT = '[profile]\nname = "Broken one"\nfoo = 1\n[values]\nchat_model = "x"\ntop_kk = 3\n'
RECOMMENDED = derived_name(NOTES_AND_MARKDOWN, "project")
SOURCE_KEYS = ("chunk_size", "chunk_overlap", "top_k")
RESET_KEYS = ("chunk_size", "enable_ocr")
# What the text CLI prints when a change needs a rebuild; analyze prints its changes table.
TEXT_REINDEX_MARK: dict[Scenario, str] = {
    Scenario.APPLY: REBUILD_HINT,
    Scenario.APPLY_REINDEX: "Rebuilt:",
    Scenario.DISCARD: REBUILD_HINT,
    Scenario.RESET: REBUILD_HINT,
    Scenario.ANALYZE_APPLY: ProfileEffect.REINDEX.value,
}
_TUI_SIZE = (120, 40)
# An offer that is coming is pushed within a few message-loop ticks of the reset.
_OFFER_PAUSES = 30
_OTHER = "other"
# POST /api/sync answers 201; POST /api/analyze answers 200.
SYNC_STATUS = 201
# The sources a Settings row pills; built-in and derived values show no pill.
_PILLED_SOURCES = (SettingSource.USER, SettingSource.ENV, SettingSource.PROFILE)
_LIBRARY_KEYS = {
    LibraryAction.DUPLICATE: "d",
    LibraryAction.RENAME: "r",
    LibraryAction.DELETE: "x",
    LibraryAction.EXPORT: "e",
    LibraryAction.IMPORT: "i",
}
TuiDriver = Callable[[LilbeeApp, Pilot, World], Awaitable[Outcome]]
ChatDriver = Callable[[LilbeeApp, Pilot, ChatScreen, World], Awaitable[Outcome]]


def _cells() -> list[Any]:
    cells: list[Any] = []
    for scenario in Scenario:
        for surface in Surface:
            if (surface, scenario) in NOT_OFFERED:
                continue
            aspects = [Aspect.END_STATE, *([Aspect.REPORT] if scenario in REPORTED else [])]
            for aspect in aspects:
                cell_id = f"{surface}-{scenario}-{aspect}"
                cells.append(pytest.param(surface, scenario, aspect, id=cell_id))
    return cells


CELLS = _cells()


@pytest.fixture
def rebuilds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record each rebuild a surface starts instead of running one."""
    started = record_rebuilds(monkeypatch)
    monkeypatch.setattr(LilbeeApp, "start_rebuild", lambda _self: started.append("tui"))
    monkeypatch.setattr(analyze_mod, "ocr_language_supported", lambda code: code == "eng")
    return started


def _store() -> ProfileStore:
    return ProfileStore()


def _write(folder: Path, stem: str, text: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.toml").write_text(text, encoding="utf-8")


def _seed_court(world: World) -> None:
    _write(world.global_profiles(), "court-filings", COURT_TEXT)


def _seed_applied(world: World) -> None:
    _seed_court(world)
    profiles.apply(_store(), COURT)


def _seed_listing(world: World) -> None:
    _seed_court(world)
    _write(world.global_profiles(), "broken", BROKEN_TEXT)
    _write(world.root / PROFILES_DIRNAME, "court-filings", COURT_TEXT)


def _seed_inbox(world: World) -> None:
    (world.inbox / SHARED_FILE).write_text(SHARED_TEXT, encoding="utf-8")
    (world.inbox / BROKEN_FILE).write_text(BROKEN_TEXT, encoding="utf-8")


def _seed_notes(world: World) -> None:
    for i in range(3):
        (world.notes / f"note{i}.md").write_text(f"# Notiz {i}\n\n{GERMAN}", encoding="utf-8")


def _seeded(*steps: Callable[[World], None], **yours: Any) -> Callable[[World], None]:
    def _seed(world: World) -> None:
        for step in steps:
            step(world)
        if yours:
            apply_settings_update(yours)

    return _seed


SEEDS: dict[Scenario, Callable[[World], None]] = {
    Scenario.NEW: _seeded(_seed_court),
    Scenario.SAVE_AS: _seeded(_seed_applied, chunk_size=384, top_k=9),
    Scenario.SAVE_AS_PROJECT: _seeded(_seed_applied, chunk_size=384, top_k=9),
    Scenario.APPLY: _seeded(_seed_court, chunk_overlap=60, vision_model=VISION),
    Scenario.APPLY_REINDEX: _seeded(_seed_court, chunk_overlap=60, vision_model=VISION),
    Scenario.UPDATE: _seeded(_seed_applied, chunk_size=900),
    Scenario.DISCARD: _seeded(
        _seed_applied, chunk_size=900, top_k=9, enable_ocr=True, vision_model=VISION
    ),
    Scenario.DUPLICATE: _seeded(_seed_court),
    Scenario.RENAME: _seeded(_seed_applied),
    Scenario.DELETE: _seeded(_seed_court),
    Scenario.EXPORT: _seeded(_seed_court),
    Scenario.IMPORT: _seeded(_seed_inbox),
    Scenario.VALIDATE: _seeded(_seed_inbox),
    Scenario.SHOW: _seeded(_seed_court),
    Scenario.LIST: _seeded(_seed_listing),
    Scenario.DIFF: _seeded(_seed_court, chunk_overlap=60),
    Scenario.ANALYZE_REPORT: _seeded(_seed_notes),
    Scenario.ANALYZE_SAVE: _seeded(_seed_notes),
    Scenario.ANALYZE_SAVE_GLOBAL: _seeded(_seed_notes),
    Scenario.ANALYZE_APPLY: _seeded(_seed_notes, vision_model=VISION),
    Scenario.DISMISS_TIP: _seeded(),
    Scenario.RESET: _seeded(_seed_applied, chunk_size=900, enable_ocr=True, vision_model=VISION),
    Scenario.SOURCES: _seeded(_seed_applied, chunk_overlap=60),
}


def _validation(shown: Any) -> dict[str, Any]:
    return {"name": shown["name"], "valid": shown["valid"], "problems": list(shown["problems"])}


def _sources(values: dict[str, Any]) -> dict[str, str]:
    return {key: str(values[key]) for key in SOURCE_KEYS}


def _entry(shown: Any) -> dict[str, Any]:
    return {"name": shown["name"], "folder": shown["folder"], "description": shown["description"]}


def _listing(shown: Any) -> list[dict[str, Any]]:
    return [{"name": e["name"], "folder": e["folder"], "valid": e["valid"]} for e in shown]


def _changes(rows: Any) -> list[list[Any]]:
    return [[row["key"], row["new"], row["effect"]] for row in rows]


def _diff(shown: Any) -> dict[str, Any]:
    return {"changes": _changes(shown["changes"]), "kept": list(shown["kept"])}


def _recommendation(shown: Any) -> dict[str, Any]:
    rec = shown["recommendation"]
    return {"builtin": rec["builtin"], "name": rec["name"], "changes": _changes(rec["changes"])}


def _change_report(shown: Any) -> dict[str, Any]:
    return {"reindex_required": shown["reindex_required"], "warnings": list(shown["warnings"])}


def _analyze_report(shown: Any) -> dict[str, Any]:
    effects = {row["effect"] for row in shown["recommendation"]["changes"]}
    saved = shown["saved"]
    return {
        "reindex_required": ProfileEffect.REINDEX.value in effects,
        "warnings": list(saved["warnings"]) if saved else [],
    }


# Each reads a surface's JSON payload, shared by the CLI's --json, HTTP and MCP.
ANSWER_OF: dict[Scenario, Callable[[Any], Any]] = {
    Scenario.VALIDATE: _validation,
    Scenario.SHOW: _entry,
    Scenario.LIST: _listing,
    Scenario.DIFF: _diff,
    Scenario.ANALYZE_REPORT: _recommendation,
    Scenario.DISMISS_TIP: lambda shown: {"tip_dismissed": shown["tip_dismissed"]},
}
REPORT_OF: dict[Scenario, Callable[[Any], Any]] = {
    Scenario.APPLY: _change_report,
    Scenario.APPLY_REINDEX: _change_report,
    Scenario.DISCARD: _change_report,
    Scenario.RESET: _change_report,
    Scenario.ANALYZE_APPLY: _analyze_report,
}


def _as_json(value: Any) -> Any:
    return to_jsonable_python(value)


def _apply_and_rebuild(store: ProfileStore, rebuilds: list[str]) -> Any:
    result = profiles.apply(store, COURT)
    rebuilds.append("app")
    return result


async def _oracle(scenario: Scenario, world: World, rebuilds: list[str]) -> Outcome:
    """Run *scenario* through the app layer."""
    store = _store()
    notes = world.notes
    writes: dict[Scenario, Callable[[], object]] = {
        Scenario.NEW: lambda: profiles.new(store, FRESH, ProfileFolder.GLOBAL, from_name=COURT),
        Scenario.SAVE_AS: lambda: profiles.save_as(SAVED, ProfileFolder.GLOBAL),
        Scenario.SAVE_AS_PROJECT: lambda: profiles.save_as(SAVED, ProfileFolder.PROJECT),
        Scenario.UPDATE: lambda: profiles.update(store),
        Scenario.DUPLICATE: lambda: profiles.duplicate(store, COURT, COPY, ProfileFolder.GLOBAL),
        Scenario.RENAME: lambda: profiles.rename(store, COURT, RENAMED),
        Scenario.DELETE: lambda: profiles.delete(store, COURT),
        Scenario.EXPORT: lambda: profiles.export(store, COURT, world.outbox),
        Scenario.IMPORT: lambda: profiles.import_profile(
            store, world.inbox / SHARED_FILE, ProfileFolder.GLOBAL
        ),
    }
    if scenario in writes:
        writes[scenario]()
        return Outcome()
    changes: dict[Scenario, Callable[[], Any]] = {
        Scenario.APPLY: lambda: profiles.apply(store, COURT),
        Scenario.APPLY_REINDEX: lambda: _apply_and_rebuild(store, rebuilds),
        Scenario.DISCARD: profiles.discard,
        Scenario.RESET: lambda: reset_settings(list(RESET_KEYS)),
    }
    if scenario in changes:
        return Outcome(report=_change_report(_as_json(changes[scenario]())))
    queries: dict[Scenario, Callable[[], Any]] = {
        Scenario.VALIDATE: lambda: profiles.validate(
            world.inbox / BROKEN_FILE, ProfileFolder.GLOBAL
        ),
        Scenario.SHOW: lambda: profiles.show(store, COURT),
        Scenario.LIST: lambda: profiles.list_profiles(store).entries,
        Scenario.DIFF: lambda: profiles.diff(store, COURT),
    }
    if scenario in queries:
        return Outcome(answer=_oracle_answer(scenario, queries[scenario]()))
    if scenario is Scenario.SOURCES:
        return Outcome(answer=_sources({k: s.value for k, s in setting_sources().items()}))
    if scenario is Scenario.DISMISS_TIP:
        hide_tip(cfg.data_root)
        return Outcome(answer={"tip_dismissed": tip_state(cfg.data_root).tip_dismissed})
    requests = {
        Scenario.ANALYZE_REPORT: AnalyzeRequest(directory=notes),
        Scenario.ANALYZE_SAVE: AnalyzeRequest(directory=notes, save=RECOMMENDED),
        Scenario.ANALYZE_SAVE_GLOBAL: AnalyzeRequest(
            directory=notes, save=RECOMMENDED, target=ProfileFolder.GLOBAL
        ),
        Scenario.ANALYZE_APPLY: AnalyzeRequest(directory=notes, apply=True),
    }
    report = await run_analysis(store, requests[scenario])
    return _oracle_analyze(scenario, report)


def _oracle_answer(scenario: Scenario, found: Any) -> Any:
    """A query's answer in the JSON shape the surfaces share."""
    if scenario is Scenario.VALIDATE:
        return {"name": found.name, "valid": found.valid, "problems": list(found.problems)}
    if scenario is Scenario.SHOW:
        return {"name": found.name, "folder": found.folder.value, "description": _about(found)}
    if scenario is Scenario.LIST:
        return [
            {"name": e.name, "folder": e.folder.value, "valid": e.file is not None} for e in found
        ]
    rows = [[row.key, _as_json(row.new), row.effect.value] for row in found.changes]
    return {"changes": rows, "kept": list(found.kept)}


def _about(entry: Any) -> str | None:
    return entry.file.description if entry.file is not None else None


def _oracle_analyze(scenario: Scenario, report: AnalyzeReport) -> Outcome:
    rec = report.recommendation
    rows = [[row.key, _as_json(row.new), row.effect.value] for row in rec.changes]
    if scenario is Scenario.ANALYZE_REPORT:
        return Outcome(answer={"builtin": rec.builtin, "name": rec.name, "changes": rows})
    if scenario in (Scenario.ANALYZE_SAVE, Scenario.ANALYZE_SAVE_GLOBAL):
        return Outcome()
    reindex = any(row.effect is ProfileEffect.REINDEX for row in rec.changes)
    warnings = list(report.saved.warnings) if report.saved is not None else []
    return Outcome(report={"reindex_required": reindex, "warnings": warnings})


def _cli_steps(scenario: Scenario, world: World) -> list[list[str]]:
    shared = str(world.inbox / SHARED_FILE)
    notes = str(world.notes)
    steps: dict[Scenario, list[list[str]]] = {
        Scenario.NEW: [["profile", "new", FRESH, "--from", COURT, "--target", "global"]],
        Scenario.SAVE_AS: [["profile", "save", SAVED, "--target", "global"]],
        Scenario.SAVE_AS_PROJECT: [["profile", "save", SAVED, "--target", "project"]],
        Scenario.APPLY: [["profile", "apply", COURT]],
        Scenario.APPLY_REINDEX: [["profile", "apply", COURT, "--reindex"]],
        Scenario.UPDATE: [["profile", "update"]],
        Scenario.DISCARD: [["profile", "discard"]],
        Scenario.DUPLICATE: [["profile", "duplicate", COURT, COPY, "--target", "global"]],
        Scenario.RENAME: [["profile", "rename", COURT, RENAMED]],
        Scenario.DELETE: [["profile", "delete", COURT]],
        Scenario.EXPORT: [["profile", "export", COURT, str(world.outbox)]],
        Scenario.IMPORT: [["profile", "import", shared, "--target", "global"]],
        Scenario.VALIDATE: [["profile", "validate", str(world.inbox / BROKEN_FILE)]],
        Scenario.SHOW: [["profile", "show", COURT]],
        Scenario.LIST: [["profile", "list"]],
        Scenario.DIFF: [["profile", "diff", COURT]],
        Scenario.ANALYZE_REPORT: [["analyze", notes]],
        Scenario.ANALYZE_SAVE: [["analyze", notes, "--save", RECOMMENDED]],
        Scenario.ANALYZE_SAVE_GLOBAL: [
            ["analyze", notes, "--save", RECOMMENDED, "--target", "global"]
        ],
        Scenario.ANALYZE_APPLY: [["analyze", notes, "--apply"]],
        Scenario.DISMISS_TIP: [["analyze", "--off"]],
        Scenario.RESET: [["settings", "unset", *RESET_KEYS]],
        Scenario.SOURCES: [["settings", "get", key] for key in SOURCE_KEYS],
    }
    return steps[scenario]


def _cli_validation(result: Result) -> dict[str, Any]:
    head, *rest = result.output.splitlines()
    name = head.removesuffix(" is not a valid profile:")
    problems = [line.removeprefix("  ") for line in rest if line]
    return {"name": name, "valid": False, "problems": problems}


def _cli_source(result: Result, json_mode: bool) -> str:
    if json_mode:
        return str(json_of(result)["source"])
    shown = next(line for line in result.output.splitlines() if line.startswith("source: "))
    return shown.removeprefix("source: ").replace(" ", "_")


def _cli_json_outcome(scenario: Scenario, results: list[Result]) -> Outcome:
    if scenario is Scenario.SOURCES:
        return Outcome(
            answer={k: _cli_source(r, True) for k, r in zip(SOURCE_KEYS, results, strict=True)}
        )
    if scenario in ANSWER_OF:
        shown = json_of(results[0])
        shown = shown["profiles"] if scenario is Scenario.LIST else shown
        return Outcome(answer=ANSWER_OF[scenario](shown))
    if scenario in REPORT_OF:
        return Outcome(report=REPORT_OF[scenario](json_of(results[0])))
    return Outcome()


def _cli_text_outcome(scenario: Scenario, results: list[Result]) -> Outcome:
    """The text CLI's output; the comparison reads it for what the oracle's answer names."""
    if scenario is Scenario.SOURCES:
        return Outcome(
            answer={k: _cli_source(r, False) for k, r in zip(SOURCE_KEYS, results, strict=True)}
        )
    if scenario is Scenario.VALIDATE:
        return Outcome(answer=_cli_validation(results[0]))
    output = results[0].output
    return Outcome(
        answer=output if scenario in ANSWER_OF else None,
        report=output if scenario in REPORT_OF else None,
    )


async def _drive_cli(scenario: Scenario, world: World, json_mode: bool) -> Outcome:
    results: list[Result] = []
    for args in _cli_steps(scenario, world):
        results.append(await in_thread(partial(cli, world, args, json_mode=json_mode)))
    expected_exit = 1 if scenario is Scenario.VALIDATE else 0
    for result in results:
        assert result.exit_code == expected_exit, result.output
    if json_mode:
        return _cli_json_outcome(scenario, results)
    return _cli_text_outcome(scenario, results)


def _url(name: str, suffix: str = "") -> str:
    return f"/api/profiles/{quote(name, safe='')}{suffix}"


def _ok(response: Any, status: int = 200) -> Any:
    assert response.status_code == status, response.text
    return response


def _json(response: Any) -> Any:
    return _ok(response).json()


def _done(response: Any, status: int = 200) -> Any:
    """The ``done`` payload of an SSE response answered with *status*."""
    events = sse_events(_ok(response, status).text)
    assert events and events[-1][0] == "done", events
    return events[-1][1]


def _http_export(client: TestClient[Any], world: World) -> None:
    response = _ok(client.get(_url(COURT, "/export")))
    filename = response.headers["content-disposition"].split('filename="')[1].split('"')[0]
    (world.outbox / filename).write_text(response.text, encoding="utf-8")


def _http_apply_then_rebuild(client: TestClient[Any], _world: World) -> Any:
    applied = _json(client.post(_url(COURT, "/apply")))
    _done(client.post("/api/sync", json={"force_rebuild": True}), SYNC_STATUS)
    return applied


def _http_analyze(**body: Any) -> Callable[[TestClient[Any], World], Any]:
    def _call(client: TestClient[Any], world: World) -> Any:
        return _done(client.post("/api/analyze", json={"directory": str(world.notes), **body}))

    return _call


HTTP_CALLS: dict[Scenario, Callable[[TestClient[Any], World], Any]] = {
    Scenario.NEW: lambda c, _w: _json(
        c.post("/api/profiles/new", json={"name": FRESH, "from_profile": COURT})
    ),
    Scenario.SAVE_AS: lambda c, _w: _json(
        c.post("/api/profiles", json={"name": SAVED, "target": "global"})
    ),
    Scenario.SAVE_AS_PROJECT: lambda c, _w: _json(
        c.post("/api/profiles", json={"name": SAVED, "target": "project"})
    ),
    Scenario.APPLY: lambda c, _w: _json(c.post(_url(COURT, "/apply"))),
    Scenario.APPLY_REINDEX: _http_apply_then_rebuild,
    Scenario.UPDATE: lambda c, _w: _json(c.put(_url(COURT))),
    Scenario.DISCARD: lambda c, _w: _json(c.post("/api/profiles/discard")),
    Scenario.DUPLICATE: lambda c, _w: _json(
        c.post(_url(COURT, "/duplicate"), json={"new_name": COPY, "target": "global"})
    ),
    Scenario.RENAME: lambda c, _w: _json(c.patch(_url(COURT), json={"new_name": RENAMED})),
    Scenario.DELETE: lambda c, _w: _json(c.delete(_url(COURT))),
    Scenario.EXPORT: _http_export,
    Scenario.IMPORT: lambda c, _w: _json(
        c.post(
            "/api/profiles/import",
            json={"content": SHARED_TEXT, "filename": SHARED_FILE, "target": "global"},
        )
    ),
    Scenario.VALIDATE: lambda c, _w: _json(
        c.post(
            "/api/profiles/validate",
            json={"content": BROKEN_TEXT, "filename": BROKEN_FILE, "folder": "global"},
        )
    ),
    Scenario.SHOW: lambda c, _w: _json(c.get(_url(COURT))),
    Scenario.LIST: lambda c, _w: _json(c.get("/api/profiles"))["profiles"],
    Scenario.DIFF: lambda c, _w: _json(c.get(_url(COURT, "/diff"))),
    Scenario.ANALYZE_REPORT: _http_analyze(),
    Scenario.ANALYZE_SAVE: _http_analyze(save=RECOMMENDED),
    Scenario.ANALYZE_SAVE_GLOBAL: _http_analyze(save=RECOMMENDED, target="global"),
    Scenario.ANALYZE_APPLY: _http_analyze(apply=True),
    Scenario.DISMISS_TIP: lambda c, _w: _json(c.post("/api/analyze/dismiss")),
    Scenario.RESET: lambda c, _w: _json(
        c.post("/api/config/reset", json={"keys": list(RESET_KEYS)})
    ),
    Scenario.SOURCES: lambda c, _w: _sources(_json(c.get("/api/config/sources"))["sources"]),
}


def _payload_outcome(scenario: Scenario, payload: Any) -> Outcome:
    """Read a JSON payload the same way whichever surface returned it."""
    if scenario is Scenario.SOURCES:
        return Outcome(answer=payload)
    if scenario in ANSWER_OF:
        return Outcome(answer=ANSWER_OF[scenario](payload))
    if scenario in REPORT_OF:
        return Outcome(report=REPORT_OF[scenario](payload))
    return Outcome()


async def _drive_http(scenario: Scenario, world: World) -> Outcome:
    def _call() -> Any:
        with http_client() as client:
            return HTTP_CALLS[scenario](client, world)

    return _payload_outcome(scenario, await in_thread(_call))


def _manage(action: McpAction, **arguments: Any) -> tuple[str, dict[str, Any]]:
    return "profile_manage", {"action": action.value, **arguments}


def _mcp_analyze(**arguments: Any) -> Callable[[World], list[tuple[str, dict[str, Any]]]]:
    return lambda w: [("analyze", {"directory": str(w.notes), **arguments})]


MCP_CALLS: dict[Scenario, Callable[[World], list[tuple[str, dict[str, Any]]]]] = {
    Scenario.NEW: lambda _w: [_manage(McpAction.NEW, name=FRESH, from_profile=COURT)],
    Scenario.SAVE_AS: lambda _w: [_manage(McpAction.SAVE, name=SAVED, folder="global")],
    Scenario.SAVE_AS_PROJECT: lambda _w: [_manage(McpAction.SAVE, name=SAVED, folder="project")],
    Scenario.APPLY: lambda _w: [("profile_apply", {"name": COURT})],
    Scenario.APPLY_REINDEX: lambda _w: [
        ("profile_apply", {"name": COURT}),
        ("sync", {"force_rebuild": True}),
    ],
    Scenario.UPDATE: lambda _w: [_manage(McpAction.UPDATE)],
    Scenario.DISCARD: lambda _w: [_manage(McpAction.DISCARD)],
    Scenario.DUPLICATE: lambda _w: [_manage(McpAction.DUPLICATE, name=COURT, new_name=COPY)],
    Scenario.RENAME: lambda _w: [_manage(McpAction.RENAME, name=COURT, new_name=RENAMED)],
    Scenario.DELETE: lambda _w: [_manage(McpAction.DELETE, name=COURT)],
    Scenario.EXPORT: lambda _w: [_manage(McpAction.EXPORT, name=COURT)],
    Scenario.IMPORT: lambda _w: [
        _manage(McpAction.IMPORT, content=SHARED_TEXT, filename=SHARED_FILE, folder="global")
    ],
    Scenario.VALIDATE: lambda _w: [
        _manage(McpAction.VALIDATE, content=BROKEN_TEXT, filename=BROKEN_FILE)
    ],
    Scenario.SHOW: lambda _w: [("profile_show", {"name": COURT})],
    Scenario.LIST: lambda _w: [("profile_list", {})],
    Scenario.DIFF: lambda _w: [("profile_show", {"name": COURT})],
    Scenario.ANALYZE_REPORT: _mcp_analyze(),
    Scenario.ANALYZE_SAVE: _mcp_analyze(save=RECOMMENDED),
    Scenario.ANALYZE_SAVE_GLOBAL: _mcp_analyze(save=RECOMMENDED, target="global"),
    Scenario.ANALYZE_APPLY: _mcp_analyze(apply=True),
    Scenario.DISMISS_TIP: lambda _w: [("analyze_dismiss", {})],
    Scenario.RESET: lambda _w: [("settings_reset", {"keys": list(RESET_KEYS)})],
    Scenario.SOURCES: lambda _w: [("settings_get", {"key": key}) for key in SOURCE_KEYS],
}
# The part of an MCP result the other surfaces return on their own.
MCP_PAYLOAD: dict[Scenario, Callable[[list[Any]], Any]] = {
    Scenario.SHOW: lambda results: results[0]["profile"],
    Scenario.LIST: lambda results: results[0]["profiles"],
    Scenario.DIFF: lambda results: results[0]["diff"],
    Scenario.SOURCES: lambda results: _sources(
        {r["setting"]["key"]: r["setting"]["source"] for r in results}
    ),
}


async def _drive_mcp(scenario: Scenario, world: World) -> Outcome:
    results = [await mcp_call(tool, args) for tool, args in MCP_CALLS[scenario](world)]
    for result in results:
        assert "error" not in result, result
    if scenario is Scenario.EXPORT:
        (world.outbox / results[0]["filename"]).write_text(results[0]["content"], encoding="utf-8")
    payload = MCP_PAYLOAD.get(scenario, lambda found: found[0])(results)
    return _payload_outcome(scenario, payload)


def _seen(app: LilbeeApp) -> set[str]:
    return {notification.identity for notification in app._notifications}


def _new_notices(app: LilbeeApp, seen: set[str]) -> list[Notification]:
    return [n for n in app._notifications if n.identity not in seen]


def _warnings(app: LilbeeApp, seen: set[str]) -> list[str]:
    return [n.message for n in _new_notices(app, seen) if n.severity == "warning"]


async def _notified(app: LilbeeApp, pilot: Pilot, seen: set[str]) -> None:
    """Wait until a new toast appears; the end state judges the outcome."""
    await until(pilot, lambda: bool(_new_notices(app, seen)))


def _settings(app: LilbeeApp) -> SettingsScreen:
    return next(s for s in app.screen_stack if isinstance(s, SettingsScreen))


async def _tab(app: LilbeeApp, pilot: Pilot) -> SettingsScreen:
    """Wait until the Profile tab shows this project's profile."""
    active = profiles.active(_store()).name

    def _ready() -> bool:
        screens = [s for s in app.screen_stack if isinstance(s, SettingsScreen)]
        if not screens or not screens[0].query(ProfileTab):
            return False
        line = screens[0].query_one("#profile-line-name", Static)
        return str(line.render()) == active

    assert await until(pilot, _ready), active
    return _settings(app)


async def _top(app: LilbeeApp, pilot: Pilot, kind: type) -> Any:
    """Wait until a *kind* screen is on top and holds focus."""
    assert await until(
        pilot,
        lambda: (
            isinstance(app.screen, kind)
            and app.focused is not None
            and app.focused.screen is app.screen
        ),
    ), kind
    return app.screen


async def _pill(app: LilbeeApp, pilot: Pilot, pill_id: str) -> None:
    await press(pilot, app.screen.query_one(f"#{pill_id}", ConfirmPill))


async def _pill_then_toast(app: LilbeeApp, pilot: Pilot, pill_id: str) -> set[str]:
    """Press *pill_id*, wait for the toast it causes, and return the toasts seen before it."""
    seen = _seen(app)
    await _pill(app, pilot, pill_id)
    await _notified(app, pilot, seen)
    return seen


async def _name_and_save(
    app: LilbeeApp, pilot: Pilot, name: str, folder: ProfileFolder | None = None
) -> None:
    dialog = await _top(app, pilot, SaveProfileDialog)
    dialog.query_one("#save-name", Input).value = name
    if folder is not None:
        dialog.query_one("#save-folder", Select).value = folder
    await _pill_then_toast(app, pilot, "save-save")


async def _path_and_ok(app: LilbeeApp, pilot: Pilot, path: Path) -> None:
    dialog = await _top(app, pilot, ProfilePathDialog)
    dialog.query_one("#path-input", Input).value = str(path)
    await _pill_then_toast(app, pilot, "path-ok")


async def _tab_pill(app: LilbeeApp, pilot: Pilot, action: TabAction) -> None:
    screen = await _tab(app, pilot)
    await press(pilot, screen.query_one(f"#profile-{action.value}", ConfirmPill))


async def _tui_save_as(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    await _tab_pill(app, pilot, TabAction.SAVE_AS)
    await _name_and_save(app, pilot, SAVED)
    return Outcome()


async def _tui_save_as_project(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    await _tab_pill(app, pilot, TabAction.SAVE_AS)
    await _name_and_save(app, pilot, SAVED, ProfileFolder.PROJECT)
    return Outcome()


async def _tui_update(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    seen = _seen(app)
    await _tab_pill(app, pilot, TabAction.UPDATE)
    await _notified(app, pilot, seen)
    return Outcome()


async def _tui_discard(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    seen = _seen(app)
    await _tab_pill(app, pilot, TabAction.DISCARD)
    offered = await until(pilot, lambda: isinstance(app.screen, ConfirmDialog))
    report = {"reindex_required": offered, "warnings": _warnings(app, seen)}
    if offered:
        await _top(app, pilot, ConfirmDialog)
        await _pill(app, pilot, "confirm-no")
        await until(pilot, lambda: isinstance(app.screen, SettingsScreen))
    return Outcome(report=report)


async def _apply_in_dialog(app: LilbeeApp, pilot: Pilot, pill_id: str) -> dict[str, Any]:
    """Read the Apply dialog's reindex offer, press *pill_id*, and read the warnings."""
    dialog = await _top(app, pilot, ApplyProfileDialog)
    reindex = bool(dialog.query("#apply-reindex"))
    seen = await _pill_then_toast(app, pilot, pill_id)
    return {"reindex_required": reindex, "warnings": _warnings(app, seen)}


async def _tui_apply_reindex(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    screen = await _tab(app, pilot)
    screen.query_one("#profile-select", Select).value = COURT
    return Outcome(report=await _apply_in_dialog(app, pilot, "apply-reindex"))


async def _library(app: LilbeeApp, pilot: Pilot) -> ProfileLibrary:
    """Open the library from the Profile tab and wait for its list."""
    await _tab_pill(app, pilot, TabAction.MANAGE)
    library: ProfileLibrary = await _top(app, pilot, ProfileLibrary)
    listing = library.query_one("#library-list", OptionList)
    assert await until(pilot, lambda: listing.option_count > 0)
    return library


def _rows(library: ProfileLibrary) -> list[str]:
    return [str(option.prompt) for option in library.query_one(OptionList).options]


async def _library_on(app: LilbeeApp, pilot: Pilot, name: str) -> ProfileLibrary:
    """Open the library and highlight the row for *name*."""
    library = await _library(app, pilot)
    listing = library.query_one("#library-list", OptionList)
    listing.highlighted = next(
        i for i, row in enumerate(_rows(library)) if row.startswith(f"{name}  ")
    )
    shown = library.query_one("#library-name", Static)
    assert await until(pilot, lambda: str(shown.render()) == name)
    return library


async def _library_key(app: LilbeeApp, pilot: Pilot, name: str, action: LibraryAction) -> None:
    library = await _library_on(app, pilot, name)
    await press(pilot, library.query_one("#library-list", OptionList), _LIBRARY_KEYS[action])


async def _tui_duplicate(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    await _library_key(app, pilot, COURT, LibraryAction.DUPLICATE)
    await _name_and_save(app, pilot, COPY)
    return Outcome()


async def _tui_rename(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    await _library_key(app, pilot, COURT, LibraryAction.RENAME)
    await _name_and_save(app, pilot, RENAMED)
    return Outcome()


async def _tui_delete(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    await _library_key(app, pilot, COURT, LibraryAction.DELETE)
    await _top(app, pilot, ConfirmDialog)
    await _pill_then_toast(app, pilot, "confirm-yes")
    return Outcome()


async def _tui_export(app: LilbeeApp, pilot: Pilot, world: World) -> Outcome:
    await _library_key(app, pilot, COURT, LibraryAction.EXPORT)
    await _path_and_ok(app, pilot, world.outbox)
    return Outcome()


async def _tui_import(app: LilbeeApp, pilot: Pilot, world: World) -> Outcome:
    await _library_key(app, pilot, "Default", LibraryAction.IMPORT)
    await _path_and_ok(app, pilot, world.inbox / SHARED_FILE)
    return Outcome()


def _text(widget: Static) -> str | None:
    return str(widget.render()) if widget.display else None


async def _tui_show(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    library = await _library_on(app, pilot, COURT)
    shown = {
        field: _text(library.query_one(f"#library-{field}", Static))
        for field in ("name", "folder", "description")
    }
    return Outcome(answer=shown)


async def _tui_list(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    library = await _library(app, pilot)
    return Outcome(answer=_rows(library))


def _table_changes(table: DataTable[Any]) -> list[list[str]]:
    """Each row's setting, its new value and its cost, as the table shows them."""
    rows = [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]
    return [[row[0], row[2], row[3].strip()] for row in rows]


async def _tui_diff(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    library = await _library_on(app, pilot, COURT)
    table = library.query_one("#library-changes", DataTable)
    assert await until(pilot, lambda: table.display and table.row_count > 0)
    return Outcome(answer={"changes": _table_changes(table)})


def _editor(screen: SettingsScreen, key: str) -> Any:
    return screen.query_one(f"#ed-{key}")


async def _all_panes(app: LilbeeApp, pilot: Pilot, keys: tuple[str, ...]) -> SettingsScreen:
    screen = await _tab(app, pilot)
    screen.populate_all_panes()
    assert await until(pilot, lambda: all(screen.query(f"#ed-{k}") for k in keys))
    return screen


def _not_yours(key: str) -> bool:
    return setting_sources()[key] is not SettingSource.USER


async def _declined_rebuild(app: LilbeeApp, pilot: Pilot) -> bool:
    """Answer No to a rebuild offer if one comes; whether it came."""
    if not await wait_until(
        pilot, lambda: isinstance(app.screen, ConfirmDialog), max_pauses=_OFFER_PAUSES
    ):
        return False
    await _pill(app, pilot, "confirm-no")
    assert await until(pilot, lambda: isinstance(app.screen, SettingsScreen))
    return True


async def _tui_reset(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    screen = await _all_panes(app, pilot, RESET_KEYS)
    offered = False
    warnings: list[str] = []
    for key in RESET_KEYS:
        seen = _seen(app)
        await press(pilot, _editor(screen, key), "ctrl+r")
        assert await until(pilot, partial(_not_yours, key)), key
        # Read before the wait for an offer: a toast expires, and that wait runs
        # its whole budget when no offer comes.
        warnings += _warnings(app, seen)
        offered = await _declined_rebuild(app, pilot) or offered
    return Outcome(report={"reindex_required": offered, "warnings": warnings})


def _shown_source(screen: SettingsScreen, key: str, profile_name: str | None) -> str:
    """The source a row's title shows: its pill's source, or other when it shows none."""
    title = str(screen.query_one(f"#{ROW_ID_PREFIX}{key} .setting-title", Static).render())
    for source in _PILLED_SOURCES:
        pill = source_pill(key, source, profile_name)
        if pill is not None and pill.plain in title:
            return source.value
    return _OTHER


async def _tui_sources(app: LilbeeApp, pilot: Pilot, _world: World) -> Outcome:
    screen = await _all_panes(app, pilot, SOURCE_KEYS)
    profile_name = active_profile_name()
    return Outcome(answer={key: _shown_source(screen, key, profile_name) for key in SOURCE_KEYS})


SETTINGS_DRIVERS: dict[Scenario, TuiDriver] = {
    Scenario.SAVE_AS: _tui_save_as,
    Scenario.SAVE_AS_PROJECT: _tui_save_as_project,
    Scenario.APPLY_REINDEX: _tui_apply_reindex,
    Scenario.UPDATE: _tui_update,
    Scenario.DISCARD: _tui_discard,
    Scenario.DUPLICATE: _tui_duplicate,
    Scenario.RENAME: _tui_rename,
    Scenario.DELETE: _tui_delete,
    Scenario.EXPORT: _tui_export,
    Scenario.IMPORT: _tui_import,
    Scenario.SHOW: _tui_show,
    Scenario.LIST: _tui_list,
    Scenario.DIFF: _tui_diff,
    Scenario.RESET: _tui_reset,
    Scenario.SOURCES: _tui_sources,
}


async def _submit(pilot: Pilot, chat: ChatScreen, line: str) -> None:
    field = chat.query_one("#chat-input", ChatInput)
    field.value = line
    await press(pilot, field)


async def _report_screen(
    app: LilbeeApp, pilot: Pilot, chat: ChatScreen, world: World
) -> AnalyzeReportScreen:
    """Run /analyze on the notes, then open the finished report."""
    await _submit(pilot, chat, f"/analyze {world.notes}")
    assert await until(pilot, lambda: app.last_analysis is not None)
    await _submit(pilot, chat, "/analyze report")
    report: AnalyzeReportScreen = await _top(app, pilot, AnalyzeReportScreen)
    return report


async def _tui_slash_apply(
    app: LilbeeApp, pilot: Pilot, chat: ChatScreen, _world: World
) -> Outcome:
    await _submit(pilot, chat, f"/profile {COURT}")
    return Outcome(report=await _apply_in_dialog(app, pilot, "apply-apply"))


async def _tui_analyze_report(
    app: LilbeeApp, pilot: Pilot, chat: ChatScreen, world: World
) -> Outcome:
    report = await _report_screen(app, pilot, chat, world)
    title = next(
        str(s.render())
        for s in report.query(".analyze-heading").results(Static)
        if str(s.render()).startswith(msg.ANALYZE_RECOMMEND_TITLE.format(name=""))
    )
    changes = _table_changes(report.query_one("#analyze-changes", DataTable))
    return Outcome(answer={"title": title, "changes": changes})


async def _tui_analyze_save(
    app: LilbeeApp, pilot: Pilot, chat: ChatScreen, world: World
) -> Outcome:
    await _report_screen(app, pilot, chat, world)
    await _pill_then_toast(app, pilot, "analyze-save")
    return Outcome()


async def _tui_analyze_apply(
    app: LilbeeApp, pilot: Pilot, chat: ChatScreen, world: World
) -> Outcome:
    await _report_screen(app, pilot, chat, world)
    await _pill(app, pilot, "analyze-apply")
    return Outcome(report=await _apply_in_dialog(app, pilot, "apply-apply"))


async def _tui_dismiss(app: LilbeeApp, pilot: Pilot, chat: ChatScreen, _world: World) -> Outcome:
    seen = _seen(app)
    await _submit(pilot, chat, "/analyze off")
    await _notified(app, pilot, seen)
    hidden = any(n.message == msg.ANALYZE_TIP_HIDDEN for n in _new_notices(app, seen))
    return Outcome(answer={"tip_dismissed": hidden})


CHAT_DRIVERS: dict[Scenario, ChatDriver] = {
    Scenario.APPLY: _tui_slash_apply,
    Scenario.ANALYZE_REPORT: _tui_analyze_report,
    Scenario.ANALYZE_SAVE: _tui_analyze_save,
    Scenario.ANALYZE_APPLY: _tui_analyze_apply,
    Scenario.DISMISS_TIP: _tui_dismiss,
}


async def _drive_tui(scenario: Scenario, world: World) -> Outcome:
    if scenario in CHAT_DRIVERS:
        # the host skips the ChatScreen install, so slash commands run on the real app
        with (
            ready_services(),
            mock.patch.object(ChatScreen, "_embedding_ready", return_value=True),
            mock.patch.object(ModelBar, "_scan_models"),
        ):
            app = LilbeeApp()
            async with app.run_test(size=_TUI_SIZE) as pilot:
                chat = await await_chat(app, pilot)
                assert isinstance(chat, ChatScreen)
                return await CHAT_DRIVERS[scenario](app, pilot, chat, world)
    with ready_services():
        host = SettingsHost()
        async with host.run_test(size=_TUI_SIZE) as pilot:
            return await SETTINGS_DRIVERS[scenario](host, pilot, world)


async def _drive(surface: Surface, scenario: Scenario, world: World) -> Outcome:
    if surface in (Surface.CLI, Surface.CLI_JSON):
        return await _drive_cli(scenario, world, json_mode=surface is Surface.CLI_JSON)
    if surface is Surface.HTTP:
        return await _drive_http(scenario, world)
    if surface is Surface.MCP:
        return await _drive_mcp(scenario, world)
    return await _drive_tui(scenario, world)


def _changes_as_shown(rows: list[list[Any]]) -> list[list[str]]:
    return [
        [key, value_text(new), msg.PROFILE_EFFECT_TEXT[ProfileEffect(e)]] for key, new, e in rows
    ]


def _label(entry: dict[str, Any]) -> str:
    folder = msg.PROFILE_FOLDER_TAG[ProfileFolder(entry["folder"])]
    return f"{entry['name']}  {folder}"


def _tui_answer(scenario: Scenario, answer: Any) -> Any:
    """*answer* as the TUI shows it: source pills, folder text, value text and effect labels."""
    if scenario is Scenario.SOURCES:
        visible = {source.value for source in _PILLED_SOURCES}
        return {k: v if v in visible else _OTHER for k, v in answer.items()}
    if scenario is Scenario.SHOW:
        folder = msg.PROFILE_FOLDER_TEXT[ProfileFolder(answer["folder"])]
        return {**answer, "folder": folder}
    if scenario is Scenario.LIST:
        return [_label(entry) for entry in answer]
    if scenario is Scenario.DIFF:
        return {"changes": _changes_as_shown(answer["changes"])}
    if scenario is Scenario.ANALYZE_REPORT:
        title = msg.ANALYZE_RECOMMEND_TITLE.format(name=answer["name"] or answer["builtin"])
        return {"title": title, "changes": _changes_as_shown(answer["changes"])}
    return answer


def _row_heads(actual: list[str]) -> list[str]:
    """Each library row up to its first tag: the name and the folder it is in."""
    return [row.split(",")[0] for row in actual]


def _mentions(scenario: Scenario, answer: Any) -> list[str]:
    """What the text CLI must print for *answer*."""
    if scenario is Scenario.SHOW:
        return [answer["name"], f"{answer['folder']} profile", answer["description"]]
    if scenario is Scenario.LIST:
        return [entry["name"] for entry in answer]
    if scenario is Scenario.DIFF:
        return [row[0] for row in answer["changes"]] + list(answer["kept"])
    if scenario is Scenario.ANALYZE_REPORT:
        return [answer["name"] or answer["builtin"], *(row[0] for row in answer["changes"])]
    return [TIP_HIDDEN_MESSAGE] if answer["tip_dismissed"] else []


def _text_report(scenario: Scenario, expected: dict[str, Any], output: str) -> dict[str, Any]:
    """The text CLI's report: its rebuild mark, and the expected warnings it prints."""
    return {
        "reindex_required": TEXT_REINDEX_MARK[scenario] in output,
        "warnings": [w for w in expected["warnings"] if w in output],
    }


def _comparable(
    surface: Surface, scenario: Scenario, aspect: Aspect, expected: Any, actual: Any
) -> tuple[Any, Any]:
    """The oracle's value and the surface's value, each as the surface can show it."""
    if aspect is Aspect.REPORT:
        if surface is Surface.CLI:
            return expected, _text_report(scenario, expected, actual)
        return expected, actual
    text_answer = surface is Surface.CLI and isinstance(actual["answer"], str)
    if text_answer:
        wanted = _mentions(scenario, expected["answer"])
        shown = {token: token in actual["answer"] for token in wanted}
        return {**expected, "answer": dict.fromkeys(wanted, True)}, {**actual, "answer": shown}
    if surface is Surface.TUI:
        answer = actual["answer"]
        if scenario is Scenario.LIST:
            answer = _row_heads(answer)
        shown = {**expected, "answer": _tui_answer(scenario, expected["answer"])}
        return shown, {**actual, "answer": answer}
    return expected, actual


@pytest.mark.parametrize(("surface", "scenario", "aspect"), CELLS)
async def test_surface_matches_the_app_layer(
    surface: Surface,
    scenario: Scenario,
    aspect: Aspect,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rebuilds: list[str],
) -> None:
    pristine = cfg.model_copy()
    real_before = real_root_listing()
    oracle = World(tmp_path / "oracle")
    enter(oracle, monkeypatch, pristine)
    SEEDS[scenario](oracle)
    planned = await _oracle(scenario, oracle, rebuilds)
    expected = end_state(oracle, len(rebuilds), planned.answer)
    rebuilds.clear()
    world = World(tmp_path / surface.value)
    enter(world, monkeypatch, pristine)
    SEEDS[scenario](world)
    outcome = await _drive(surface, scenario, world)
    if aspect is Aspect.END_STATE:
        pair = (expected, end_state(world, len(rebuilds), outcome.answer))
    else:
        pair = (planned.report, outcome.report)
    want, got = _comparable(surface, scenario, aspect, *pair)
    assert got == want, f"{surface} diverges on {scenario} ({aspect})"
    assert real_root_listing() == real_before


async def test_every_scenario_changes_or_answers_something(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rebuilds: list[str]
) -> None:
    """The oracle is not vacuous: a seed alone never equals the scenario's end state."""
    pristine = cfg.model_copy()
    for scenario in Scenario:
        world = World(tmp_path / scenario.value)
        enter(world, monkeypatch, pristine)
        SEEDS[scenario](world)
        seeded = end_state(world, 0, None)
        outcome = await _oracle(scenario, world, rebuilds)
        assert end_state(world, len(rebuilds), outcome.answer) != seeded, scenario
        assert (outcome.report is not None) is (scenario in REPORTED), scenario
        rebuilds.clear()


async def test_the_reported_scenarios_carry_a_rebuild_and_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rebuilds: list[str]
) -> None:
    """The report cells have power: each seed makes a change that needs a rebuild and warns."""
    pristine = cfg.model_copy()
    for scenario in sorted(REPORTED):
        world = World(tmp_path / scenario.value)
        enter(world, monkeypatch, pristine)
        SEEDS[scenario](world)
        report = (await _oracle(scenario, world, rebuilds)).report
        assert report["reindex_required"] is True, scenario
        assert report["warnings"], scenario
        rebuilds.clear()


def test_the_cells_cover_every_offered_pair() -> None:
    offered = len(Scenario) * len(Surface) - len(NOT_OFFERED)
    reported = len(REPORTED) * len(Surface)
    assert len(CELLS) == offered + reported


def test_the_not_offered_cells_are_absent_from_their_surface() -> None:
    tab_and_library = {a.value for a in TabAction} | {a.value for a in LibraryAction}
    assert Scenario.NEW.value not in tab_and_library
    assert Scenario.VALIDATE.value not in tab_and_library
    assert "default_save_folder()" in inspect.getsource(AnalyzeReportScreen.action_save)


def test_apply_takes_one_call_to_reindex_only_on_the_cli_and_tui() -> None:
    """HTTP and MCP apply take only a name, so a rebuild is a second call; see TWO_STEP."""
    assert set(TWO_STEP) == {Surface.HTTP, Surface.MCP}
    assert list(inspect.signature(mcp_profile_apply).parameters) == ["name"]
    assert set(profiles_apply_route.fn.__annotations__) == {"name", "return"}
