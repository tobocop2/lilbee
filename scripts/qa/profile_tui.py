"""Profile TUI screens via Textual's pilot harness.

Runs the same gestures a manual tester would (open each screen, type
text, press chord keys, toggle views) and measures the latency of each
step end-to-end. The TUI is a real LilbeeApp running in the same
process, so the measurements include compose + on_mount + watcher
re-renders, just like a real session.

Output:
    [ms]   step
    -----  ----------------------------------
      4.2  open Chat (default)
     38.7  switch to Catalog
      0.4  press v (toggle to list)
    ...

Anything over the per-step budget is flagged with a >>>SLOW<<< marker
so a reader skimming the report sees the regressions immediately.

Usage::

    uv run python scripts/qa/profile_tui.py            # text report
    uv run python scripts/qa/profile_tui.py --json     # machine readable

Exit code is non-zero when at least one step exceeds its budget so the
script can gate CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

# Default per-step latency budget (ms). Steps marked with their own
# explicit budget below override this. Conservative on a dev box; CI
# would set tighter budgets via env once a baseline is established.
_DEFAULT_BUDGET_MS = 250.0

# Some steps legitimately take longer (first model-bar paint imports
# heavy provider modules). The budget reflects what's tolerable; it is
# NOT a measurement target, just a regression gate.
_BUDGETS_MS: dict[str, float] = {
    # Boot is heavy (services init, theme load, screen install).
    "boot LilbeeApp + open Chat": 1500.0,
    # Switch budgets target time-to-interactive, not "until every
    # worker settles." The earlier numbers buried the actual switch
    # cost under a fixed 500 ms settle wait that wasn't doing real
    # work, just blocking on pilot.pause(0.05) ten times.
    "switch to Catalog": 500.0,
    "switch to Settings": 600.0,
    "switch to Tasks": 350.0,
    "switch to Status": 350.0,
    "switch to Wiki": 500.0,
    "switch to Catalog (re-entry)": 250.0,
    "type 5 chars in chat input": 500.0,  # Textual TextArea ~75ms/char
    "type 5 chars in catalog search": 800.0,
    "press v (toggle to list view)": 700.0,
    "press v (toggle back to grid)": 600.0,
    "press [": 250.0,
    "press ]": 250.0,
    "press escape (chat normal mode)": 250.0,
    "press i (chat insert mode)": 250.0,
    # Stress budgets. ``type+clear long catalog filter`` types 28 chars
    # then deletes 28; debounce collapses it to two filter passes total.
    # The toggle storm is 8 keystrokes against the list cache. The
    # paging stress is 40 scroll keys with no remount cost.
    "stress: type+clear long catalog filter": 6000.0,
    "stress: 8x grid <-> list toggle": 5500.0,
    "stress: 40x pgdn/pgup in catalog": 6000.0,
}


@dataclass
class StepResult:
    name: str
    ms: float
    budget_ms: float
    over_budget: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ms": round(self.ms, 2),
            "budget_ms": self.budget_ms,
            "over_budget": self.over_budget,
            "error": self.error,
        }


@dataclass
class ProfileReport:
    steps: list[StepResult] = field(default_factory=list)

    def add(self, step: StepResult) -> None:
        self.steps.append(step)

    @property
    def any_over_budget(self) -> bool:
        return any(s.over_budget for s in self.steps)

    @property
    def total_ms(self) -> float:
        return sum(s.ms for s in self.steps)

    def render_text(self) -> str:
        out: list[str] = []
        out.append(f"{'ms':>10}  {'step':<60} {'budget':>8}")
        out.append("-" * 88)
        for s in self.steps:
            mark = " >>>SLOW<<<" if s.over_budget else ""
            err = f"  ERROR: {s.error}" if s.error else ""
            out.append(f"{s.ms:>10.2f}  {s.name:<60} {s.budget_ms:>8.0f}{mark}{err}")
        out.append("-" * 88)
        out.append(f"{self.total_ms:>10.2f}  total")
        return "\n".join(out)

    def to_json(self) -> str:
        return json.dumps(
            {
                "total_ms": round(self.total_ms, 2),
                "any_over_budget": self.any_over_budget,
                "steps": [s.to_dict() for s in self.steps],
            },
            indent=2,
        )


class _Profiler:
    """Async helper that times pilot operations against a LilbeeApp."""

    def __init__(self, report: ProfileReport) -> None:
        self.report = report

    async def step(self, name: str, fn, settle: bool = False) -> None:
        """Run *fn* and measure wall time.

        When ``settle`` is True a fixed post-step settle wait runs
        outside the measured window so background workers can land
        before the next step starts. Including the settle in the
        measurement inflates wall time (the pause(0.05) loop dominates)
        without measuring real work.
        """
        budget = _BUDGETS_MS.get(name, _DEFAULT_BUDGET_MS)
        t0 = time.perf_counter()
        err: str | None = None
        try:
            await fn()
        except Exception as exc:  # surface but keep going so other steps run
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        ms = (time.perf_counter() - t0) * 1000
        over = ms > budget or err is not None
        self.report.add(StepResult(name=name, ms=ms, budget_ms=budget, over_budget=over, error=err))


async def run_profile() -> ProfileReport:  # noqa: C901, PLR0915 -- linear screen-by-screen QA walk
    """Drive a LilbeeApp through every screen + a few interactions."""
    # Imports inside the runner so module-import time doesn't pollute
    # the boot measurement.
    from textual.widgets import Input

    from lilbee.cli.tui.app import LilbeeApp
    from lilbee.cli.tui.widgets.chat_input import ChatInput

    report = ProfileReport()
    profiler = _Profiler(report)

    app = LilbeeApp()

    async with app.run_test(size=(160, 48)) as pilot:

        async def boot() -> None:
            await pilot.pause()

        await profiler.step("boot LilbeeApp + open Chat", boot)

        # Chat input: typing latency.
        async def type_in_chat() -> None:
            inp = app.screen.query_one("#chat-input", ChatInput)
            inp.focus()
            await pilot.pause()
            for ch in "hello":
                await pilot.press(ch)
            await pilot.pause()

        await profiler.step("type 5 chars in chat input", type_in_chat)

        # Brackets must NOT navigate while input has focus (A1 contract).
        async def bracket_in_chat() -> None:
            await pilot.press("left_square_bracket")
            await pilot.pause()
            await pilot.press("right_square_bracket")
            await pilot.pause()

        await profiler.step("press [", bracket_in_chat)

        # Defocus input so brackets navigate at the screen level.
        async def escape_to_normal() -> None:
            await pilot.press("escape")
            await pilot.pause()

        await profiler.step("press escape (chat normal mode)", escape_to_normal)

        async def press_i() -> None:
            await pilot.press("i")
            await pilot.pause()

        await profiler.step("press i (chat insert mode)", press_i)

        # Now defocus and cycle screens.
        await pilot.press("escape")
        await pilot.pause()

        async def to_catalog() -> None:
            app.switch_view("Catalog")
            await pilot.pause()  # one tick: screen mounted + first paint

        async def settle_catalog() -> None:
            for _ in range(10):
                await pilot.pause(0.05)

        await profiler.step("switch to Catalog", to_catalog)
        await settle_catalog()

        async def type_in_catalog() -> None:
            search = app.screen.query_one("#catalog-search", Input)
            search.focus()
            await pilot.pause()
            for ch in "qwen3":
                await pilot.press(ch)
            await pilot.pause()

        await profiler.step("type 5 chars in catalog search", type_in_catalog)

        # Toggle catalog views; B1 ensures this no longer deadlocks.
        async def toggle_to_list() -> None:
            search = app.screen.query_one("#catalog-search", Input)
            search.value = ""
            await pilot.pause()
            # Defocus search so v reaches the screen binding.
            scroll = app.screen.query_one("#catalog-grid")
            scroll.focus()
            await pilot.pause()
            await pilot.press("v")
            await pilot.pause()

        await profiler.step("press v (toggle to list view)", toggle_to_list)

        async def toggle_back() -> None:
            await pilot.press("v")
            await pilot.pause()

        await profiler.step("press v (toggle back to grid)", toggle_back)

        # Settings.
        async def to_settings() -> None:
            app.switch_view("Settings")
            await pilot.pause()

        async def settle_settings() -> None:
            for _ in range(5):
                await pilot.pause(0.05)

        await profiler.step("switch to Settings", to_settings)
        await settle_settings()

        # Settings dropped its filter input -- tabs already group the
        # ~60 settings into 8 small chunks, so search added complexity
        # (debounce, full-DOM walk, populate-all-on-filter) for little
        # navigational value. No "type in settings search" step now.

        async def to_tasks() -> None:
            app.switch_view("Tasks")
            await pilot.pause()

        await profiler.step("switch to Tasks", to_tasks)

        async def to_status() -> None:
            app.switch_view("Status")
            await pilot.pause()

        await profiler.step("switch to Status", to_status)

        # Catalog re-entry should be fast on second visit (install_screen reuse).
        async def to_catalog_again() -> None:
            app.switch_view("Catalog")
            await pilot.pause()

        await profiler.step("switch to Catalog (re-entry)", to_catalog_again)
        await settle_catalog()

        # Stress steps: simulate the kind of heavy catalog navigation a
        # user does while picking a model -- typing a long query,
        # toggling grid <-> list repeatedly, scrolling the list, then
        # clearing and retyping the filter. These exercise paths that
        # were brittle (B1 deadlock, repeat remount on toggle) and
        # produce the "jittery / sluggish" symptoms in user reports.

        async def stress_long_filter() -> None:
            search = app.screen.query_one("#catalog-search", Input)
            search.focus()
            await pilot.pause()
            for ch in "qwen2-instruct-vision-large":
                await pilot.press(ch)
            await pilot.pause(0.2)  # past debounce, single filter pass
            for _ in range(28):
                await pilot.press("backspace")
            await pilot.pause(0.2)

        await profiler.step("stress: type+clear long catalog filter", stress_long_filter)

        async def stress_toggle_storm() -> None:
            scroll = app.screen.query_one("#catalog-grid")
            scroll.focus()
            await pilot.pause()
            for _ in range(8):  # 4 round trips
                await pilot.press("v")
                await pilot.pause()

        await profiler.step("stress: 8x grid <-> list toggle", stress_toggle_storm)

        async def stress_list_pagedown() -> None:
            # Make sure we're in list view, then scroll heavy.
            scroll_id = "#catalog-list" if not app.screen._grid_view else "#catalog-grid"
            scroll = app.screen.query_one(scroll_id)
            scroll.focus()
            await pilot.pause()
            for _ in range(20):
                await pilot.press("pagedown")
            for _ in range(20):
                await pilot.press("pageup")
            await pilot.pause()

        await profiler.step("stress: 40x pgdn/pgup in catalog", stress_list_pagedown)

    return report


async def _switch_and_settle(app, pilot, expected_type, view_name) -> None:
    app.switch_view(view_name)
    await pilot.pause()
    for _ in range(10):
        await pilot.pause(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    report = asyncio.run(run_profile())

    if args.json:
        print(report.to_json())
    else:
        print(report.render_text())

    return 1 if report.any_over_budget else 0


if __name__ == "__main__":
    sys.exit(main())
