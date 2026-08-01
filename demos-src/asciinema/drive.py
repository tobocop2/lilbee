#!/usr/bin/env python3
"""Drive a lilbee reel: asciinema recording inside a sized tmux pane, keys via send-keys.

Replaces the VHS tape bodies. tmux supplies the PTY (asciinema attached read-only to an
existing pane fails with "terminal does not support clear"), asciinema logs the byte
stream with real timestamps, and agg renders offline from those timestamps.

Four primitives, each fixing a measured failure rather than a suspected one:

``type_text``   Sends a string keystroke by keystroke at a per-call rate, with a separate
                dwell before Enter. Rate is a directorial parameter, not a default: the
                retired tapes ran 35ms globally, 55ms on the deliberate ``/add`` paths and
                15ms on tui-chat's long question. Pasting a whole string produces one
                frame where the reel needs thirty-five.

``esc``         Sends Escape, then waits ESC_FLOOR before anything else. Textual's
                ESCAPE_DELAY is 100ms and two send-keys land 4-8ms apart, so ``Escape``
                followed by ``t`` arrives as ``alt+t``. ``t``/``i``/``v`` are non-priority
                bindings, so this fails silently and the take looks merely wrong rather
                than broken. ``]``/``[`` survive because those bindings are priority.

``wait_for``    Polls capture-pane for a pattern instead of sleeping a guess. Timings do
                not transfer between boxes, and every shipped tape is tuned to an M1.

``goto``        Navigates by reading the tab strip rather than counting ``]`` presses. The
                nav ring grew from five screens to seven or eight, so every counted walk
                in the old tapes now lands somewhere else.
"""
from __future__ import annotations

import dataclasses
import pathlib
import re
import subprocess
import time

ESC_FLOOR = 0.18          # > Textual's ESCAPE_DELAY of 0.1, with margin for send-keys jitter
DEFAULT_TYPE_RATE = 0.035
POLL_INTERVAL = 0.25


def _tmux(*args: str, check: bool = True) -> str:
    r = subprocess.run(["tmux", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"tmux {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


class Timeout(RuntimeError):
    pass


@dataclasses.dataclass
class Session:
    name: str
    cols: int
    rows: int
    cast: pathlib.Path
    target: str = dataclasses.field(init=False)
    marks: dict[str, float] = dataclasses.field(default_factory=dict, init=False)
    # Spans during which the driver itself was producing motion -- typing, or a burst of
    # repeated keys. Frame rate is only meaningful inside these: everywhere else the
    # cadence belongs to the app (a token stream, an unpack bar) and is not the
    # pipeline's to control.
    motion_spans: list[tuple[float, float]] = dataclasses.field(default_factory=list,
                                                                init=False)
    started_at: float = dataclasses.field(default=0.0, init=False)

    def __post_init__(self) -> None:
        # Resolved after start(): base-index and pane-base-index are user config and are
        # 1 on this box, so a hardcoded :0.0 silently addresses nothing.
        self.target = self.name

    # -- lifecycle ---------------------------------------------------------------

    def start(self, command: str, env: dict[str, str] | None = None) -> None:
        """Open a tmux pane at exact dims and record `command` inside it."""
        self.kill()
        self.cast.parent.mkdir(parents=True, exist_ok=True)
        self.cast.unlink(missing_ok=True)
        env = {
            "COLORTERM": "truecolor",
            "TERM": "xterm-256color",
            "LILBEE_THEME": "rose-pine",
            **(env or {}),
        }
        exports = " ".join(f"{k}={v}" for k, v in env.items())
        inner = f"{exports} asciinema rec --overwrite -c {command!r} {str(self.cast)!r}"
        _tmux("new-session", "-d", "-s", self.name,
              "-x", str(self.cols), "-y", str(self.rows), inner)
        # tmux clamps to the client size unless the window is detached-sized; force it.
        _tmux("set-option", "-t", self.name, "window-size", "manual", check=False)
        _tmux("resize-window", "-t", self.name, "-x", str(self.cols), "-y", str(self.rows),
              check=False)
        self.started_at = time.monotonic()
        pane = _tmux("list-panes", "-t", self.name, "-F", "#{pane_id}").split()
        if not pane:
            raise RuntimeError(f"session {self.name} started with no pane")
        self.target = pane[0]

    def kill(self) -> None:
        _tmux("kill-session", "-t", self.name, check=False)

    def alive(self) -> bool:
        return self.name in _tmux("list-sessions", "-F", "#{session_name}", check=False)

    # -- observation -------------------------------------------------------------

    def screen(self) -> str:
        return _tmux("capture-pane", "-p", "-t", self.target, check=False)

    def wait_for(self, pattern: str, timeout: float = 60.0, *, absent: bool = False) -> float:
        """Block until `pattern` appears (or disappears). Returns seconds waited."""
        rx = re.compile(pattern)
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            hit = bool(rx.search(self.screen()))
            if hit is not absent:
                return time.monotonic() - start
            time.sleep(POLL_INTERVAL)
        state = "vanish" if absent else "appear"
        raise Timeout(f"{pattern!r} did not {state} within {timeout}s\n--- screen ---\n"
                      f"{self.screen()}")

    def ask(self, question: str, *, rate: float = 0.045, dwell: float = 0.9) -> None:
        """Put the chat input in insert mode, type a question, and confirm it arrived.

        Does not wait for the INSERT chip. With the placement drawer open the header is
        narrowed and the chip is pushed off it, so waiting on the chip fails on a reel
        where the input is working perfectly well. What matters is whether the characters
        landed in the input, so that is what gets checked -- and if they did not, they
        went to the app as key bindings, which is worth failing the take over.
        """
        self.key("i", after=0.6)
        self.key("C-u", after=0.35)
        self.type_text(question, rate=rate)
        time.sleep(dwell)
        probe = re.escape(question[:24])
        if not re.search(probe, self.screen()):
            raise Timeout(f"question never reached the input: {question!r}\n"
                          f"--- screen ---\n{self.screen()}")
        self.key("Enter", after=0.8)

    def await_answer(self, timeout: float = 900.0) -> float:
        """Block until generation has finished, not until the answer looks likely.

        Waiting on a phrase from the expected answer is the trap this replaces. The
        question is on screen too, so ``wait_for("9C1")`` matched the text the driver had
        just typed and returned in 0.0s -- the crawl reel cut to black with the thinking
        spinner still running and every gate green, because the reel really did contain
        every string it was asked for.

        Two conditions, both about the app rather than the content: the citations block
        exists, and the footer has stopped offering to cancel the stream.
        """
        start = time.monotonic()
        self.wait_for(r"Sources:", timeout=timeout)
        self.wait_for(r"Cancel stream", absent=True,
                      timeout=max(5.0, timeout - (time.monotonic() - start)))
        return time.monotonic() - start

    def wait_settled(self, quiet: float = 1.0, timeout: float = 120.0) -> float:
        """Block until the screen stops changing for `quiet` seconds."""
        start = time.monotonic()
        last, stamp = self.screen(), time.monotonic()
        while time.monotonic() - start < timeout:
            time.sleep(POLL_INTERVAL)
            now = self.screen()
            if now != last:
                last, stamp = now, time.monotonic()
            elif time.monotonic() - stamp >= quiet:
                return time.monotonic() - start
        raise Timeout(f"screen never settled within {timeout}s")

    # -- input -------------------------------------------------------------------

    def key(self, *keys: str, after: float = 0.35) -> None:
        """Send keys `after` seconds apart, measured from send to send.

        The interval is corrected for the ~25ms cost of the send-keys call itself, for the
        same reason type_text corrects it: a burst of repeated keys is the only motion in
        a navigation reel, so the interval between them IS the frame rate agg renders.
        Uncorrected, a 90ms setting produced 100ms frames and a 10fps reel.
        """
        begin = time.monotonic()
        for k in keys:
            t0 = time.monotonic()
            _tmux("send-keys", "-t", self.target, k)
            time.sleep(max(0.0, after - (time.monotonic() - t0)))
        if len(keys) >= 4 and after <= 0.12:
            self._span(begin)

    def esc(self, times: int = 1) -> None:
        for _ in range(times):
            _tmux("send-keys", "-t", self.target, "Escape")
            time.sleep(ESC_FLOOR)

    def type_text(self, text: str, *, rate: float = DEFAULT_TYPE_RATE,
                  enter: bool = False, dwell: float = 0.0) -> None:
        """Type `text` one keystroke at a time so the reel shows typing, not a paste.

        The sleep is corrected for the cost of the send-keys call itself. Each spawn runs
        roughly 25ms, so sleeping the full rate on top of it produced a 60ms interval from
        a 35ms setting. That matters more than it looks: agg emits one frame per content
        change, so the typing interval IS the frame rate during motion, and the
        uncorrected version rendered every reel at 12-17fps against a 25fps target.
        """
        begin = time.monotonic()
        for ch in text:
            t0 = time.monotonic()
            _tmux("send-keys", "-t", self.target, "-l", ch)
            time.sleep(max(0.0, rate - (time.monotonic() - t0)))
        self._span(begin)
        if dwell:
            time.sleep(dwell)
        if enter:
            _tmux("send-keys", "-t", self.target, "Enter")

    # -- navigation --------------------------------------------------------------

    def goto(self, screen: str, *, forward: bool = True, limit: int = 10,
             marker: str | None = None) -> None:
        """Walk the screen ring until `screen` is showing, reading the strip each step.

        Counting bracket presses is what the old tapes did and it is now wrong; the ring
        grew and the count no longer lands where the tape assumed.
        """
        probe = marker or screen
        for _ in range(limit):
            current = self.screen()
            if re.search(probe, current, re.IGNORECASE):
                return
            # Close an overlay before walking. The Sessions and Fleet drawers sit on top
            # of whatever screen is showing and swallow the ring keys, so a walk that
            # starts under one presses its limit without moving and fails on a reel whose
            # navigation is otherwise correct.
            if re.search(r"Filter conversations|No saved conversations", current):
                self.key("C-o", after=0.6)
                continue
            if re.search(r"Placement", current):
                self.esc()
                time.sleep(0.3)
                continue
            self.key("]" if forward else "[")
        raise Timeout(f"never reached {screen!r} in {limit} steps\n--- screen ---\n"
                      f"{self.screen()}")

    # -- trim windows ------------------------------------------------------------

    def repaint(self) -> None:
        """Force the app to redraw the whole screen.

        Head-trimming a cast is only safe at a boundary where the next event paints
        everything; Textual otherwise emits cursor-addressed diffs against frames the
        trim just removed. Nudging the window size makes it repaint in full.
        """
        _tmux("resize-window", "-t", self.name, "-x", str(self.cols - 1), "-y",
              str(self.rows), check=False)
        time.sleep(0.4)
        _tmux("resize-window", "-t", self.name, "-x", str(self.cols), "-y",
              str(self.rows), check=False)
        time.sleep(0.8)

    def _span(self, begin: float) -> None:
        """Record a driver-driven motion window, in session-clock seconds."""
        self.motion_spans.append((begin - self.started_at,
                                  time.monotonic() - self.started_at))

    def mark(self, label: str) -> None:
        """Record a trim boundary as a timestamp, never as bytes sent to the app.

        An earlier version wrote an OSC sentinel with ``send-keys -H``, which delivers
        those bytes to the application as keystrokes rather than to the terminal as
        output. ``\\x1b]1337;`` arrives as Escape followed by ``]``, and ``]`` is the
        screen-ring binding, so every sentinel silently advanced a screen and the reel
        drifted onto Fleet several beats later. Offsets against the session clock are
        enough: the cast carries real timestamps, so post-trim can find the window
        without anything being written into the stream.
        """
        self.marks[label] = time.monotonic() - self.started_at


def calibrate(cols: int, rows: int) -> tuple[int, int]:
    """Render one frame at these dims and report the pixel size."""
    import tempfile
    from PIL import Image

    import agg_finish

    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        s = Session("reelcal", cols, rows, d / "cal.cast")
        s.start("printf x; sleep 1")
        time.sleep(2.5)
        s.kill()
        time.sleep(0.5)
        out = agg_finish.render(d / "cal.cast", d / "cal")
        return Image.open(out["gif"]).size
