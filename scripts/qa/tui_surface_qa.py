#!/usr/bin/env python3
"""TUI surface QA: boot the real LilbeeApp headless and drive every screen.

Uses Textual's pilot (not a terminal) so it works reliably in CI/headless and
surfaces real bugs: exceptions on mount/navigation, error-level logs, and error
notifications. Run: uv run --no-sync python scripts/qa/tui_surface_qa.py
"""

from __future__ import annotations

import asyncio
import logging
import traceback

RESULTS: list[tuple[str, bool, str]] = []
LOG_RECORDS: list[logging.LogRecord] = []


def rec(cid: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((cid, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}" + (f"  -- {detail}" if detail else ""), flush=True)


class _Collector(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        LOG_RECORDS.append(record)


def _install_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(_Collector())


async def drive() -> None:
    from textual.widgets import Static

    from lilbee.cli.tui.app import LilbeeApp

    notes: list[tuple[str, str]] = []

    class _QAApp(LilbeeApp):
        def notify(self, message, *args, **kwargs):  # type: ignore[no-untyped-def, override]
            sev = str(kwargs.get("severity", "information"))
            notes.append((sev, str(message)))
            return super().notify(message, *args, **kwargs)

    app = _QAApp()
    try:
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            await pilot.pause()
            # 1. did the app mount a real screen with content?
            scr = app.screen
            widget_count = len(list(scr.query("*")))
            rec("TUI-boot", widget_count > 3, f"initial screen={type(scr).__name__} widgets={widget_count}")

            # 2. navigate every managed view and confirm it mounts
            from lilbee.cli.tui import messages as msg

            for view in [*msg.get_nav_views(), "Chat"]:
                before = len(notes)
                try:
                    app.switch_view(view)
                    await pilot.pause()
                    await pilot.pause()
                    scr = app.screen
                    n = len(list(scr.query("*")))
                    err_notes = [m for s, m in notes[before:] if s == "error"]
                    ok = n > 1 and not err_notes
                    rec(f"TUI-view:{view}", ok, f"{type(scr).__name__} widgets={n}" + (f" ERR={err_notes}" if err_notes else ""))
                except Exception:
                    rec(f"TUI-view:{view}", False, f"EXC: {traceback.format_exc().splitlines()[-1]}")

            # 3. on the chat screen, type into the prompt (no send -> no model load)
            try:
                app.switch_view("Chat")
                await pilot.pause()
                await pilot.press("h", "i")
                await pilot.pause()
                rec("TUI-input", True, "typed into prompt without crash")
            except Exception:
                rec("TUI-input", False, f"EXC: {traceback.format_exc().splitlines()[-1]}")

            # 4. error notifications raised during the whole run
            errs = [m for s, m in notes if s == "error"]
            rec("TUI-no-error-toasts", not errs, f"error toasts: {errs[:3]}" if errs else "none")
    except Exception:
        rec("TUI-run", False, f"app.run_test crashed: {traceback.format_exc().splitlines()[-1]}")


def main() -> int:
    _install_logging()
    try:
        asyncio.run(drive())
    except Exception:
        rec("TUI-asyncio", False, traceback.format_exc().splitlines()[-1])

    # report error/exception logs captured during the run
    errs = [r for r in LOG_RECORDS if r.levelno >= logging.ERROR]
    warns = [r for r in LOG_RECORDS if r.levelno == logging.WARNING]
    print(f"\n=== captured logs: {len(errs)} ERROR, {len(warns)} WARNING ===", flush=True)
    for r in errs[:15]:
        print(f"  ERROR {r.name}: {r.getMessage()[:200]}", flush=True)
        if r.exc_info:
            print("    " + "".join(traceback.format_exception(*r.exc_info)).splitlines()[-1], flush=True)
    seen: set[str] = set()
    for r in warns:
        key = f"{r.name}:{r.getMessage()[:80]}"
        if key in seen:
            continue
        seen.add(key)
        print(f"  WARN  {r.name}: {r.getMessage()[:160]}", flush=True)

    failed = [c for c, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
