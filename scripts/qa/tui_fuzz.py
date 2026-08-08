"""Seeded adversarial fuzz harness for the TUI.

Drives the real ``LilbeeApp`` headlessly (Textual pilot: real app, real key
and mouse events) through seeded-random adversarial sessions: random keys on
every screen, view-cycling bursts mid-fetch, resize storms, random clicks and
double-clicks, Escape/q spam through modals, and unicode paste into whatever
holds focus. Structural invariants are asserted after every step, not just at
the end; a violation aborts the session and prints the seed plus the full
action trace so the failure replays deterministically.

Findings are filed as beads with the replay command, not fixed inline.

Usage:
    uv run python scripts/qa/tui_fuzz.py smoke                 # fixed seeds, CI-sized
    uv run python scripts/qa/tui_fuzz.py long --minutes 15     # fresh seeds until time is up
    uv run python scripts/qa/tui_fuzz.py replay --seed 7 --steps 120
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import string
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from textual import events

sys.path.insert(0, str(Path(__file__).parent))
from tui_nav_eval import _await_chat, _isolated_env

# Keys that end the app; the session should survive everything else.
_EXCLUDED_KEYS = frozenset({"ctrl+c", "ctrl+q"})

_NAV_KEYS = ["up", "down", "left", "right", "tab", "shift+tab", "home", "end", "pageup", "pagedown"]
_VIEW_KEYS = ["m", "t", "q", "escape", "ctrl+g", "ctrl+o", "[", "]"]
_CATALOG_KEYS = ["1", "2", "3", "4", "5", "6", "/", "<", ">", "s", "d", "g", "G", "enter", "space"]
_LETTER_KEYS = [*string.ascii_lowercase, "S", "F5", "f1", "f2"]
_KEY_POOL = sorted(set(_NAV_KEYS + _VIEW_KEYS + _CATALOG_KEYS + _LETTER_KEYS) - _EXCLUDED_KEYS)

_PASTE_SAMPLES = [
    "héllo wörld 🐝🐝🐝",
    "日本語のテキストと絵文字🎌",
    "a" * 500,
    "line one\nline two\r\nline three\ttabbed",
    "\x1b[31mANSI red\x1b[0m",
    "'; DROP TABLE models; --",
    "runes ᛒᛖᛖ, zero-width ​‍ joins, ligature ﷽",
]

_RESPONSE_TIMEOUT_S = 10.0


@dataclass
class Violation:
    seed: int
    step: int
    action: str
    detail: str
    trace: list[str]


class _ErrorLogCapture(logging.Handler):
    """Collects ERROR+ records carrying a traceback from the app's loggers."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info:
            self.records.append(f"{record.name}: {record.getMessage()}")


def _check_invariants(app) -> str | None:
    """Return a violation description, or None when the app looks structurally sound."""
    from lilbee.cli.tui.widgets.grid_select import GridSelect
    from lilbee.cli.tui.widgets.model_grid import ModelGrid

    if app._exception is not None:
        tb = "".join(traceback.format_exception(app._exception))
        return f"app crashed: {app._exception!r}\n{tb}"
    if not app.screen_stack:
        return "screen stack is empty"
    focused = app.focused
    if focused is not None and not focused.is_mounted:
        return f"focused widget is unmounted: {focused!r}"
    if isinstance(focused, ModelGrid) and focused.rows and focused.highlighted is None:
        return f"focused ModelGrid has rows but no highlight: {focused!r}"
    if isinstance(focused, GridSelect) and focused.children and focused.highlighted is None:
        return f"focused GridSelect has children but no highlight: {focused!r}"
    return None


async def _pump(pilot) -> None:
    """One responsiveness probe: the app must settle within a bounded window."""
    await asyncio.wait_for(pilot.pause(), timeout=_RESPONSE_TIMEOUT_S)


async def _act_keys(pilot, rng: random.Random) -> str:
    keys = [rng.choice(_KEY_POOL) for _ in range(rng.randint(1, 5))]
    await pilot.press(*keys)
    return f"press {' '.join(keys)}"


async def _act_view_burst(pilot, rng: random.Random) -> str:
    keys = [rng.choice(_VIEW_KEYS) for _ in range(rng.randint(2, 6))]
    await pilot.press(*keys)
    return f"view-burst {' '.join(keys)}"


async def _act_escape_spam(pilot, rng: random.Random) -> str:
    key = rng.choice(["escape", "q"])
    count = rng.randint(3, 6)
    await pilot.press(*[key] * count)
    return f"spam {key} x{count}"


async def _act_resize(pilot, rng: random.Random) -> str:
    width = rng.choice([20, 40, 80, 120, 200])
    height = rng.choice([10, 24, 40, 60])
    await pilot.resize_terminal(width, height)
    return f"resize {width}x{height}"


async def _act_click(pilot, rng: random.Random) -> str:
    size = pilot.app.size
    x = rng.randrange(max(1, size.width))
    y = rng.randrange(max(1, size.height))
    times = rng.choice([1, 1, 2])
    await pilot.click(offset=(x, y), times=times)
    return f"click ({x},{y}) x{times}"


async def _act_paste(pilot, rng: random.Random) -> str:
    text = rng.choice(_PASTE_SAMPLES)
    pilot.app.post_message(events.Paste(text))
    return f"paste {text[:24]!r}..."


async def _act_type_garbage(pilot, rng: random.Random) -> str:
    chars = [rng.choice("abcxyz0189é漢🐝/<>[]{}%$") for _ in range(rng.randint(3, 12))]
    await pilot.press(*chars)
    return f"type {''.join(chars)!r}"


_ACTIONS = [
    (_act_keys, 5),
    (_act_view_burst, 2),
    (_act_escape_spam, 2),
    (_act_resize, 2),
    (_act_click, 3),
    (_act_paste, 1),
    (_act_type_garbage, 2),
]
_ACTION_FNS = [fn for fn, weight in _ACTIONS for _ in range(weight)]


async def _fuzz_session(seed: int, steps: int, tmp: Path) -> Violation | None:
    from lilbee.cli.tui.app import LilbeeApp

    rng = random.Random(seed)
    capture = _ErrorLogCapture()
    logging.getLogger("lilbee").addHandler(capture)
    trace: list[str] = []
    try:
        with _isolated_env(tmp):
            app = LilbeeApp()
            async with app.run_test(size=(120, 40)) as pilot:
                await _await_chat(app, pilot)
                await pilot.pause()
                for step in range(steps):
                    action_fn = rng.choice(_ACTION_FNS)
                    action = await action_fn(pilot, rng)
                    trace.append(action)
                    try:
                        await _pump(pilot)
                    except TimeoutError:
                        return Violation(seed, step, action, "app unresponsive", trace)
                    detail = _check_invariants(app)
                    if detail is None and capture.records:
                        detail = f"traceback logged: {capture.records[0]}"
                    if detail is not None:
                        return Violation(seed, step, action, detail, trace)
                # End the walk somewhere quittable so run_test can exit cleanly.
                await pilot.press("escape", "escape", "escape")
                await pilot.pause()
    except Exception as exc:  # a crash escaping the pilot is itself the finding
        detail = "".join(traceback.format_exception(exc))
        return Violation(seed, len(trace), trace[-1] if trace else "startup", detail, trace)
    finally:
        logging.getLogger("lilbee").removeHandler(capture)
    return None


def _run_sessions(seed_steps: list[tuple[int, int]]) -> int:
    failures = 0
    for seed, steps in seed_steps:
        started = time.monotonic()
        with tempfile.TemporaryDirectory() as tmp:
            violation = asyncio.run(_fuzz_session(seed, steps, Path(tmp)))
        elapsed = time.monotonic() - started
        if violation is None:
            print(f"seed {seed}: {steps} steps clean in {elapsed:.1f}s")
            continue
        failures += 1
        print(f"seed {seed}: VIOLATION at step {violation.step} after {violation.action!r}")
        print(f"  {violation.detail}")
        replay_cmd = f"uv run python scripts/qa/tui_fuzz.py replay --seed {seed} --steps {steps}"
        print(f"  replay: {replay_cmd}")
        print("  trace:")
        for i, action in enumerate(violation.trace):
            print(f"    {i:4d}  {action}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    smoke = sub.add_parser("smoke", help="fixed seeds, CI-sized (~2 min)")
    smoke.add_argument("--steps", type=int, default=40)
    long_p = sub.add_parser("long", help="fresh seeds until the time budget is spent")
    long_p.add_argument("--minutes", type=float, default=15.0)
    long_p.add_argument("--steps", type=int, default=150)
    replay = sub.add_parser("replay", help="re-run one seed deterministically")
    replay.add_argument("--seed", type=int, required=True)
    replay.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()

    if args.cmd == "smoke":
        failures = _run_sessions([(seed, args.steps) for seed in (1, 2, 3, 4, 5)])
    elif args.cmd == "replay":
        failures = _run_sessions([(args.seed, args.steps)])
    else:
        deadline = time.monotonic() + args.minutes * 60
        failures = 0
        while time.monotonic() < deadline:
            seed = random.SystemRandom().randrange(1, 2**31)
            failures += _run_sessions([(seed, args.steps)])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
