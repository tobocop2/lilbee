"""Press every binding on every surface and grade what it renders.

The fuzz harness next door drives random keys and asserts the app stays
structurally sound: no crash, non-empty screen stack, focus mounted, grids
highlighted. It says nothing about whether the result is legible, because a
screen made of partial-block borders and double-width glyphs is structurally
perfect and visually broken.

This sweep is the complement. It is exhaustive rather than random: every binding
each surface declares, pressed one at a time, and after each press the resulting
screen is graded for render defects. Terminal-independent by construction, so it
runs headless in CI rather than needing a real terminal.

It drives the real app rather than the test host, because the host skips the
on_mount that installs the chat screen and every nav key that switches to chat
then throws, which reads as nineteen defects that are all the harness.

Usage:
    uv run python scripts/qa/tui_render_sweep.py            # every surface
    uv run python scripts/qa/tui_render_sweep.py --surface chat
    uv run python scripts/qa/tui_render_sweep.py --svg out/  # save each state
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tui_nav_eval import _await_chat, _isolated_env

# Partial blocks do not tile in every terminal font; box-drawing always does.
PARTIAL_BLOCK_BORDERS = frozenset({"tall", "thick", "wide", "panel"})
BORDER_EDGES = ("border_top", "border_right", "border_bottom", "border_left")

# Keys that quit; the sweep must survive its own run.
EXCLUDED_KEYS = frozenset({"ctrl+c", "ctrl+q"})

SURFACES: dict[str, tuple[str, str]] = {
    "chat": ("lilbee.cli.tui.screens.chat", "ChatScreen"),
    "catalog": ("lilbee.cli.tui.screens.catalog", "CatalogScreen"),
    "settings": ("lilbee.cli.tui.screens.settings", "SettingsScreen"),
    "tasks": ("lilbee.cli.tui.screens.task_center", "TaskCenter"),
    "fleet": ("lilbee.cli.tui.screens.fleet", "FleetScreen"),
    "sessions": ("lilbee.cli.tui.screens.sessions", "SessionsScreen"),
    "memories": ("lilbee.cli.tui.screens.memories", "MemoriesScreen"),
    "status": ("lilbee.cli.tui.screens.status", "StatusScreen"),
    "wiki": ("lilbee.cli.tui.screens.wiki", "WikiScreen"),
    "wiki-drafts": ("lilbee.cli.tui.screens.wiki_drafts", "WikiDraftsScreen"),
    "startup-gate": ("lilbee.cli.tui.screens.startup_gate", "StartupGate"),
    "palette": ("lilbee.cli.tui.screens.command_palette", "LilbeeCommandPalette"),
    "confirm": ("lilbee.cli.tui.widgets.confirm_dialog", "ConfirmDialog"),
    "notice": ("lilbee.cli.tui.widgets.notice_dialog", "NoticeDialog"),
    "crawl": ("lilbee.cli.tui.widgets.crawl_dialog", "CrawlDialog"),
    "slash-catalog": ("lilbee.cli.tui.widgets.slash_command_catalog", "SlashCommandCatalog"),
    "model-info": ("lilbee.cli.tui.screens.model_info", "ModelInfoModal"),
    "model-picker": ("lilbee.cli.tui.screens.model_picker", "ModelPickerModal"),
}


def screen_kwargs(cls_name: str) -> dict:
    """Constructor arguments for the surfaces that take them."""
    from lilbee.catalog.types import ModelTask
    from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
    from lilbee.cli.tui.widgets.model_bar import ModelOption

    row = LocalCatalogRow(
        name="Qwen3 0.6B",
        task=ModelTask.CHAT,
        params="0.6B",
        size="0.5 GB",
        quant="Q4_K_M",
        downloads="--",
        featured=False,
        installed=True,
        sort_downloads=0,
        sort_size=0.5,
        ref="Qwen/Qwen3-0.6B-GGUF",
    )
    return {
        "ConfirmDialog": {"title": "Delete model", "message": "Removes 17.3 GB from disk."},
        "NoticeDialog": {"title": "Heads up", "message": "The engine is still warming."},
        "ModelInfoModal": {"row": row},
        "ModelPickerModal": {
            "scope": "chat",
            "options": [ModelOption("Qwen3 0.6B", "Qwen/Qwen3-0.6B-GGUF")],
        },
    }.get(cls_name, {})


def grade(screen) -> list[str]:
    """Render defects visible on the mounted screen."""
    defects: list[str] = []
    for widget in screen.query("*"):
        name = type(widget).__name__
        for edge in BORDER_EDGES:
            border = getattr(widget.styles, edge, None)
            if border and border[0] in PARTIAL_BLOCK_BORDERS:
                defects.append(f"partial-block border: {name}.{edge}={border[0]}")
        title = getattr(widget, "border_title", None)
        for text in (title or "",):
            for char in text:
                if unicodedata.east_asian_width(char) == "W" or 0x1F000 <= ord(char) <= 0x1FAFF:
                    defects.append(f"double-width glyph in border title: {name} {char!r}")
    return defects


def bindings_for(app, screen) -> list[str]:
    """Every key the surface or the app declares, minus the ones that quit."""
    keys: list[str] = []
    for source in (screen, app):
        for binding in getattr(source, "BINDINGS", []):
            key = getattr(binding, "key", None) or (
                binding[0] if isinstance(binding, tuple) else None
            )
            if key and key not in EXCLUDED_KEYS:
                keys.append(key)
    return sorted(set(keys))


async def sweep_surface(label: str, svg_dir: Path | None, tmp: Path) -> tuple[int, list[str]]:
    from lilbee.cli.tui.app import LilbeeApp

    module, cls_name = SURFACES[label]
    findings: list[str] = []

    async def mount(app, pilot):
        cls = getattr(importlib.import_module(module), cls_name)
        app.push_screen(cls(**screen_kwargs(cls_name)))
        for _ in range(4):
            await pilot.pause()

    # One mount per surface: a fresh app per key is correct but costs a couple of
    # seconds each, which is twenty minutes over the whole set. Escape between
    # keys returns from whatever a key opened, so states mostly do not compound;
    # a key that leaves the surface is reported and the sweep re-mounts.
    with _isolated_env(tmp):
        app = LilbeeApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await _await_chat(app, pilot)
            await mount(app, pilot)
            keys = bindings_for(app, app.screen)
            findings += [f"[on mount] {d}" for d in grade(app.screen)]

            for key in keys:
                findings += await _press_and_grade(app, pilot, key, label, cls_name, svg_dir)
                await _return_to_surface(app, pilot, cls_name, mount)
    return len(keys), findings


async def _press_and_grade(app, pilot, key, label, cls_name, svg_dir) -> list[str]:
    try:
        await pilot.press(key)
        for _ in range(2):
            await pilot.pause()
    except Exception as exc:
        return [f"[{key}] raised {type(exc).__name__}: {exc}"]
    if app._exception is not None:
        crash = f"[{key}] app crashed: {app._exception!r}"
        app._exception = None
        return [crash]
    if svg_dir:
        safe = key.replace("+", "-").replace("/", "slash")
        (svg_dir / f"{label}--{safe}.svg").write_text(app.export_screenshot(), encoding="utf-8")
    return [f"[{key}] {defect}" for defect in grade(app.screen)]


async def _return_to_surface(app, pilot, cls_name, mount) -> None:
    """Go back only if the key left the surface.

    Pressing Escape unconditionally eventually dismisses the surface itself and
    then pops an empty result-callback stack, which is the harness misbehaving
    rather than a defect in the screen.
    """
    if type(app.screen).__name__ == cls_name:
        return
    if len(app.screen_stack) > 1:
        await pilot.press("escape")
        await pilot.pause()
    if type(app.screen).__name__ != cls_name:
        await mount(app, pilot)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", action="append", choices=sorted(SURFACES))
    parser.add_argument("--svg", type=Path, help="write an SVG per state into this directory")
    args = parser.parse_args()

    if args.svg:
        args.svg.mkdir(parents=True, exist_ok=True)

    total_defects = 0
    tmp = tempfile.mkdtemp(prefix="lilbee-render-sweep-")
    for label in args.surface or sorted(SURFACES):
        try:
            n_keys, findings = await asyncio.wait_for(
                sweep_surface(label, args.svg, Path(tmp)), timeout=600
            )
        except Exception as exc:
            print(f"  {label:14} COULD NOT SWEEP: {type(exc).__name__}: {str(exc)[:70]}")
            continue
        total_defects += len(findings)
        status = "clean" if not findings else f"{len(findings)} DEFECTS"
        print(f"  {label:14} {n_keys:>3} bindings   {status}")
        for finding in findings[:6]:
            print(f"      {finding}")
        if len(findings) > 6:
            print(f"      ... and {len(findings) - 6} more")
    print(f"\ntotal render defects: {total_defects}")
    return 1 if total_defects else 0


if __name__ == "__main__":
    os.environ.setdefault("LILBEE_MODELS_DIR", "/tmp/lilbee-render-sweep-models")
    raise SystemExit(asyncio.run(main()))
