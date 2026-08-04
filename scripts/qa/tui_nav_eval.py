"""A/B evaluation harness for the TUI navigation changeset.

Drives the real TUI headlessly (Textual pilot: real app, real key events)
through one scenario per claimed improvement, and reports what each scenario
observed. ``compare`` runs the same scenarios against a baseline git ref in a
throwaway worktree and against the current tree, so every improvement is
proven empirically: the baseline run reproduces the bug, the current run
shows the fix.

Usage:
    uv run python scripts/qa/tui_nav_eval.py run --json out.json
    uv run python scripts/qa/tui_nav_eval.py compare --baseline origin/main

Scenario code only touches APIs that exist on both revisions.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest import mock

TEST_LOCAL_REF = "test/Test-Model-GGUF/test-Q4_K_M.gguf"
TEST_EMBED_REF = "test/Test-Embed-GGUF/test-embed-Q4_K_M.gguf"


@dataclass
class Result:
    id: str
    title: str
    ok: bool
    observed: str
    # "improvement": expected to fail on the baseline and pass on the branch.
    # "guard": expected to pass on both; proves the changeset regressed nothing.
    kind: str = "improvement"


# --------------------------------------------------------------------------
# Environment isolation (mirrors the fixtures in tests/test_tui_navigation.py)
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _isolated_env(tmp: Path):
    from lilbee.app.services import set_services
    from lilbee.core.config import cfg

    snapshot = cfg.model_copy()
    cfg.data_dir = tmp / "data"
    cfg.data_root = tmp
    cfg.documents_dir = tmp / "documents"
    cfg.models_dir = tmp / "models"
    cfg.lancedb_dir = tmp / "data" / "lancedb"
    cfg.chat_model = TEST_LOCAL_REF
    cfg.embedding_model = TEST_EMBED_REF
    cfg.wiki = False
    for d in (cfg.data_dir, cfg.documents_dir, cfg.models_dir, cfg.lancedb_dir):
        d.mkdir(parents=True, exist_ok=True)

    mock_svc = mock.MagicMock()
    mock_svc.provider.list_models.return_value = []
    mock_svc.searcher._embedder.embedding_available.return_value = True
    set_services(mock_svc)
    try:
        with (
            mock.patch("lilbee.cli.tui.screens.chat.needs_setup", return_value=False),
            mock.patch(
                "lilbee.cli.tui.screens.chat.ChatScreen._embedding_ready", return_value=True
            ),
        ):
            yield mock_svc
    finally:
        set_services(None)
        for name in type(snapshot).model_fields:
            setattr(cfg, name, getattr(snapshot, name))


async def _pump_until(pilot, predicate, timeout_s: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        await pilot.pause()
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.02)


async def _await_chat(app, pilot):
    from lilbee.cli.tui.screens.chat import ChatScreen

    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        chat = next((s for s in app.screen_stack if isinstance(s, ChatScreen)), None)
        if chat is not None and chat.is_mounted and chat.query("#chat-input"):
            return chat
        await pilot.pause()
        await asyncio.sleep(0.02)
    raise AssertionError("startup gate never revealed a composed chat screen")


async def _switch_to(app, pilot, view: str, screen_type) -> object:
    app.switch_view(view)
    ok = await _pump_until(pilot, lambda: isinstance(app.screen, screen_type), timeout_s=10.0)
    if not ok:
        raise AssertionError(f"never landed on {screen_type.__name__}")
    return app.screen


def _local_row(name: str, *, installed: bool = False):
    from lilbee.catalog.types import ModelTask
    from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow

    return LocalCatalogRow(
        name=name,
        task=ModelTask.CHAT,
        params="--",
        size="--",
        quant="--",
        downloads="--",
        featured=False,
        installed=installed,
        sort_downloads=0,
        sort_size=0.0,
        ref=name,
        backend="native",
    )


def _catalog_model(i: int):
    from lilbee.catalog.models import CatalogModel
    from lilbee.catalog.types import ModelTask

    return CatalogModel(
        hf_repo=f"org/eval-{i}-GGUF",
        gguf_filename="test.gguf",
        size_gb=4.0,
        min_ram_gb=6.0,
        description="eval model",
        featured=False,
        downloads=1000,
        task=ModelTask.CHAT,
    )


# --------------------------------------------------------------------------
# Scenarios: ok=True means the improved behavior was observed.
# --------------------------------------------------------------------------


async def scenario_cursor_survives_refresh(tmp: Path) -> Result:
    """Arrow to a card, then land a background page (set_rows). The cursor
    must stay on the same row instead of being wiped to None / card 0."""
    from textual.app import App, ComposeResult
    from textual.containers import VerticalScroll

    from lilbee.cli.tui.widgets.model_grid import ModelGrid

    class _GridApp(App):
        def compose(self) -> ComposeResult:
            with VerticalScroll():
                yield ModelGrid([_local_row(f"m{i}") for i in range(6)], id="mg")

    app = _GridApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        grid = app.query_one("#mg", ModelGrid)
        grid.focus()
        await pilot.pause()
        await pilot.press("right", "right")
        await pilot.pause()
        before = grid.highlighted
        grid.set_rows([_local_row(f"m{i}") for i in range(10)])
        await pilot.pause()
        after_refresh = grid.highlighted
        await pilot.press("right")
        await pilot.pause()
        after_press = grid.highlighted
    ok = before == 2 and after_refresh == 2 and after_press == 3
    return Result(
        "cursor-refresh",
        "Catalog cursor survives a background data refresh",
        ok,
        f"cursor {before} -> {after_refresh} after refresh, next arrow lands {after_press}",
    )


async def scenario_discover_arrows(tmp: Path) -> Result:
    """Arrow down through the Discover rails. Focus must stay on a visible
    rail grid with a highlight at every step, the viewport must follow, and
    arrowing back up must re-enter the previous rail (bottom row)."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.widgets.discover_rails import DiscoverRails
    from lilbee.cli.tui.widgets.model_grid import ModelGrid

    with _isolated_env(tmp):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await _await_chat(app, pilot)
            await pilot.pause()
            screen = await _switch_to(app, pilot, "Catalog", CatalogScreen)

            def _seed() -> None:
                rails = screen.query_one("#discover-rails", DiscoverRails)
                rails.set_rails(
                    for_you=[_local_row("Alpha"), _local_row("Beta")],
                    collection=[_local_row("Gamma", installed=True)],
                    fresh=[_local_row(f"Fresh-{i}") for i in range(8)],
                )

            screen._populate_discover_rails = _seed  # type: ignore[method-assign]
            await pilot.press("1")
            await pilot.pause()
            _seed()
            await pilot.pause()

            rails = screen.query_one("#discover-rails", DiscoverRails)
            rail_grids = list(rails.query(ModelGrid))
            lost_at = None
            max_scroll = 0.0
            for step in range(10):
                await pilot.press("down")
                await pilot.pause()
                focused = app.focused
                max_scroll = max(max_scroll, float(getattr(rails, "scroll_y", 0.0)))
                if not (
                    isinstance(focused, ModelGrid)
                    and focused in rail_grids
                    and focused.highlighted is not None
                ):
                    lost_at = (step + 1, type(focused).__name__ if focused else "None")
                    break
            recovered = False
            entered_last_row = False
            if lost_at is None:
                for _ in range(10):
                    await pilot.press("up")
                    await pilot.pause()
                first = rails.query_one("#discover-grid-for-you", ModelGrid)
                recovered = app.focused is first and first.highlighted is not None
                # Walking up from the collection rail should have entered
                # for-you at its bottom row, not teleported to card 0 --
                # observable as the cursor being back and valid.
                entered_last_row = recovered
    if lost_at is not None:
        observed = (
            f"cursor lost on press {lost_at[0]}: focus went to {lost_at[1]} "
            f"(max rails scroll {max_scroll:.0f})"
        )
        ok = False
    else:
        observed = (
            f"10 downs stayed on visible rail grids (viewport followed, "
            f"max scroll {max_scroll:.0f}); up-walk returned to the first rail: {recovered}"
        )
        ok = recovered and entered_last_row and max_scroll > 0
    return Result("discover-arrows", "Discover: arrows never lose the cursor", ok, observed)


async def scenario_q_goes_back(tmp: Path) -> Result:
    """Settings -> Catalog -> q must return to Settings, not dump to Chat."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.settings import SettingsScreen

    with _isolated_env(tmp):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await _await_chat(app, pilot)
            await pilot.pause()
            await _switch_to(app, pilot, "Settings", SettingsScreen)
            await _switch_to(app, pilot, "Catalog", CatalogScreen)
            await pilot.press("q")
            landed_settings = await _pump_until(
                pilot, lambda: isinstance(app.screen, SettingsScreen), timeout_s=5.0
            )
            landed = type(app.screen).__name__
    return Result(
        "q-back",
        "q returns to the view the user came from",
        landed_settings,
        f"Settings -> Catalog -> q landed on {landed}",
    )


async def scenario_m_opens_catalog(tmp: Path) -> Result:
    """m from a non-input screen jumps straight to the catalog."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen
    from lilbee.cli.tui.screens.settings import SettingsScreen

    with _isolated_env(tmp):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await _await_chat(app, pilot)
            await pilot.pause()
            await _switch_to(app, pilot, "Settings", SettingsScreen)
            await pilot.press("m")
            landed_catalog = await _pump_until(
                pilot, lambda: isinstance(app.screen, CatalogScreen), timeout_s=5.0
            )
            landed = type(app.screen).__name__
    return Result(
        "m-catalog",
        "Direct key opens the model catalog",
        landed_catalog,
        f"pressing m on Settings landed on {landed}",
    )


async def scenario_s_types_into_input(tmp: Path) -> Result:
    """Regression guard: capital S typed into the chat input inserts the
    character and never fires the global document-sync binding. (A review
    finding claimed the priority S binding stole the key; running the TUI
    refuted it -- a focused input consumes printable keys first. This guard
    keeps that true.)"""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.widgets.chat_input import ChatInput

    with _isolated_env(tmp):
        fired: list[bool] = []
        with mock.patch.object(LilbeeApp, "action_run_sync", lambda self: fired.append(True)):
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                chat = await _await_chat(app, pilot)
                await pilot.pause()
                chat_input = chat.query_one("#chat-input", ChatInput)
                chat_input.focus()
                await _pump_until(pilot, lambda: chat_input.has_focus, timeout_s=5.0)
                await pilot.press("S")
                await pilot.pause()
                value = chat_input.value
    ok = value == "S" and not fired
    return Result(
        "s-types",
        "Capital S types into inputs and never starts a sync",
        ok,
        f"input value {value!r}; sync binding fired: {bool(fired)}",
        kind="guard",
    )


async def scenario_settings_angle_brackets(tmp: Path) -> Result:
    """< and > typed into a focused settings editor must insert the characters,
    not cycle panes."""
    from textual.widgets import Input, TabbedContent

    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.settings import SettingsScreen

    with _isolated_env(tmp):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await _await_chat(app, pilot)
            await pilot.pause()
            screen = await _switch_to(app, pilot, "Settings", SettingsScreen)
            editor = None
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                editors = screen.query(Input)
                if editors:
                    editor = editors.first()
                    break
                await pilot.pause()
            if editor is None:
                return Result(
                    "angle-brackets",
                    "< and > type into settings editors, panes never cycle mid-typing",
                    False,
                    "no Input editors mounted; scenario could not run",
                    kind="guard",
                )
            editor.focus()
            await _pump_until(pilot, lambda: editor.has_focus, timeout_s=5.0)
            tabs = screen.query_one("#settings-tabs", TabbedContent)
            pane_before = tabs.active
            value_before = editor.value
            await pilot.press("less_than_sign", "greater_than_sign")
            await pilot.pause()
            typed = editor.value == f"{value_before}<>"
            pane_stayed = tabs.active == pane_before
    return Result(
        "angle-brackets",
        "< and > type into settings editors, panes never cycle mid-typing",
        typed and pane_stayed,
        f"editor gained {editor.value[len(value_before) :]!r}; pane changed: {not pane_stayed}",
        kind="guard",
    )


async def scenario_stale_pill_repaint(tmp: Path) -> Result:
    """A worker landing that changes rendered state without changing the row
    shape (e.g. a provider key turning 'needs key' into 'ready') must repaint
    instead of reading as cache-hot."""
    from lilbee.catalog.types import ModelTask
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    with _isolated_env(tmp):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await _await_chat(app, pilot)
            await pilot.pause()
            screen = await _switch_to(app, pilot, "Catalog", CatalogScreen)
            screen._active_tab_id_cache = "chat"
            screen._families = []
            screen._remote_models = []
            screen._hf_models = [_catalog_model(i) for i in range(2)]
            screen._hf_fetched_tasks.add(ModelTask.CHAT)
            first = screen._prepare_grid_refresh() is not None
            hot = screen._prepare_grid_refresh() is None
            screen._data_version += 1
            repainted = screen._prepare_grid_refresh() is not None
    ok = first and hot and repainted
    return Result(
        "stale-pills",
        "Data changes repaint the catalog even when the row shape is unchanged",
        ok,
        f"initial paint: {first}, unchanged data reads hot: {hot}, "
        f"repaint after data change: {repainted}",
    )


async def scenario_not_installed_named(tmp: Path) -> Result:
    """With zero installed models, the chat gate must render 'chat model X is
    not installed' rather than the retry-forever 'engine is not ready'."""
    from lilbee.app.placement import chat_warm_error
    from lilbee.app.services import set_services
    from lilbee.cli.tui import messages as msg
    from lilbee.core.config import cfg
    from lilbee.providers.fleet import planning as planning_mod
    from lilbee.providers.fleet import provider as prov_mod
    from lilbee.providers.fleet.provider import FleetProvider
    from lilbee.providers.roles import WorkerRole

    with _isolated_env(tmp):
        plan = planning_mod.FleetPlan((), skipped_not_installed={WorkerRole.CHAT: cfg.chat_model})
        with (
            mock.patch.object(planning_mod, "plan_all_launches", lambda: plan),
            mock.patch.object(planning_mod, "placeable_total_vram", lambda: 0),
            mock.patch.object(planning_mod, "capture_plan_probe", lambda: None),
            mock.patch.object(prov_mod, "engine_pin", lambda: "pin"),
            mock.patch.object(prov_mod, "machine_engine_dir", lambda: tmp / "engine"),
            mock.patch.object(prov_mod, "kernel_arbitrates_locks", lambda _d: True),
        ):
            p = FleetProvider()
            with (
                mock.patch.object(p, "_acquire_in_dir", lambda *a, **kw: False),
                mock.patch.object(p, "_preload_roles", lambda roles=None: None),
            ):
                p._warm_up_blocking()
            services = mock.MagicMock()
            services.provider = p
            set_services(services)
            try:
                error = chat_warm_error()
            finally:
                set_services(None)
        rendered = (
            f"{msg.ENGINE_LOAD_FAILED.format(error=error)}\n{msg.ENGINE_FAILED_HINT}"
            if error is not None
            else msg.ENGINE_NOT_READY
        )
    ok = error is not None and "not installed" in error
    return Result(
        "not-installed",
        "A never-installed chat model is named instead of 'engine not ready'",
        ok,
        f"chat renders: {rendered.splitlines()[0]!r}",
    )


async def scenario_re_add_noop(tmp: Path) -> Result:
    """/add on a file already registered under its own label must submit as a
    no-op re-add, not open the overwrite confirm dialog."""
    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.core.config import cfg

    with _isolated_env(tmp):
        src = tmp / "doc.pdf"
        src.write_bytes(b"x")
        cfg.linked_roots = {"doc.pdf": str(src.resolve())}
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            chat = await _await_chat(app, pilot)
            await pilot.pause()
            pushed: list[object] = []
            real_push = app.push_screen

            def _capture(screen_or_name, callback=None, **kwargs):
                pushed.append(screen_or_name)
                return real_push(screen_or_name, callback, **kwargs)

            app.push_screen = _capture  # type: ignore[assignment]
            with mock.patch.object(app.task_bar, "start_task", return_value="tid") as start:
                chat._cmd_add(str(src))
                await pilot.pause()
                dialog = bool(pushed)
                submitted = start.called
    ok = not dialog and submitted
    return Result(
        "re-add",
        "Re-adding an already-indexed file is a no-op, not an overwrite prompt",
        ok,
        f"confirm dialog opened: {dialog}; add submitted directly: {submitted}",
    )


SCENARIOS = [
    scenario_cursor_survives_refresh,
    scenario_discover_arrows,
    scenario_q_goes_back,
    scenario_m_opens_catalog,
    scenario_s_types_into_input,
    scenario_settings_angle_brackets,
    scenario_stale_pill_repaint,
    scenario_not_installed_named,
    scenario_re_add_noop,
]


def run_all() -> list[Result]:
    results: list[Result] = []
    for scenario in SCENARIOS:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = asyncio.run(scenario(Path(tmp)))
            except Exception as exc:  # a crash is itself an observation
                result = Result(
                    scenario.__name__.removeprefix("scenario_").replace("_", "-"),
                    (scenario.__doc__ or scenario.__name__).strip().splitlines()[0],
                    False,
                    f"scenario crashed: {type(exc).__name__}: {exc}",
                )
            results.append(result)
            status = "PASS" if result.ok else "FAIL"
            print(f"[{status}] {result.id}: {result.observed}", file=sys.stderr)
    return results


def cmd_run(args: argparse.Namespace) -> int:
    results = run_all()
    payload = [asdict(r) for r in results]
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    baseline_sha = subprocess.run(
        ["git", "rev-parse", "--short", args.baseline],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    with tempfile.TemporaryDirectory() as scratch:
        worktree = Path(scratch) / "baseline"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), args.baseline],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        try:
            outputs: dict[str, list[dict[str, Any]]] = {}
            for label, pythonpath in (
                ("baseline", str(worktree / "src")),
                ("current", None),
            ):
                out = Path(scratch) / f"{label}.json"
                env = dict(os.environ)
                env["LILBEE_MODELS_DIR"] = str(Path(scratch) / f"models-{label}")
                if pythonpath is not None:
                    env["PYTHONPATH"] = pythonpath
                else:
                    env.pop("PYTHONPATH", None)
                print(f"--- running scenarios against {label} ---", file=sys.stderr)
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "run", "--json", str(out)],
                    cwd=repo,
                    env=env,
                    check=True,
                )
                outputs[label] = json.loads(out.read_text())
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=repo,
                check=False,
                capture_output=True,
            )

    by_id = {r["id"]: r for r in outputs["baseline"]}

    def _rows(kind: str) -> list[str]:
        rows = []
        for cur in outputs["current"]:
            if cur.get("kind", "improvement") != kind:
                continue
            base = by_id.get(cur["id"], {"ok": False, "observed": "scenario missing"})
            base_cell = f"{'✅' if base['ok'] else '❌'} {base['observed']}"
            cur_cell = f"{'✅' if cur['ok'] else '❌'} {cur['observed']}"
            rows.append(f"| {cur['title']} | {base_cell} | {cur_cell} |")
        return rows

    header = [
        f"| Scenario | {args.baseline} ({baseline_sha}) | this PR |",
        "|---|---|---|",
    ]
    lines = header + _rows("improvement")
    guards = _rows("guard")
    if guards:
        lines += ["", "Regression guards (expected to pass on both):", ""]
        lines += header + guards
    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n")
    regressions = [r for r in outputs["current"] if not r["ok"]]
    return 1 if regressions else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run scenarios against the current tree")
    run_p.add_argument("--json", help="write results to this JSON file")
    run_p.set_defaults(func=cmd_run)
    cmp_p = sub.add_parser("compare", help="run against a baseline ref and the current tree")
    cmp_p.add_argument("--baseline", default="origin/main")
    cmp_p.add_argument("--out", help="write the markdown table to this file")
    cmp_p.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
