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
    # Tracked, because the chip that shows it is not always on screen: the placement
    # drawer takes the room the tab strip renders it in. The driver sends every key, so
    # it follows the mode itself and reconciles against the chip whenever one is visible.
    _mode: str = dataclasses.field(default="INSERT", init=False)

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
        """Block until `pattern` appears (or disappears). Returns seconds waited.

        A note for anyone waiting on the chat screen: "personal encyclopedia" is the
        empty-state hint and is gone the moment a data root has any history, so it says
        "this is a fresh chat", not "chat is up". Reels that reuse a staged root failed
        on it after their first take. "Slash commands" is in the footer either way.
        """
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

    CHAT_MARKER = r"personal encyclopedia|Slash commands"

    def await_chat(self, timeout: float = 180.0) -> float:
        """Wait for the chat screen, dismissing the first-run wizard if it comes up first.

        A fresh data root opens on "Welcome to lilbee", which is a model picker, not chat.
        Only the placement reel handled it; everywhere else the boot wait sat looking for
        a chat marker that was never going to arrive until someone pressed Escape, and
        the reel died reporting a missing marker rather than the wizard it was staring at.

        Waits on either, so a root that skips the wizard costs nothing.
        """
        start = time.monotonic()
        rx = re.compile(rf"{self.CHAT_MARKER}|Welcome to lilbee")
        while time.monotonic() - start < timeout:
            screen = self.screen()
            if re.search(self.CHAT_MARKER, screen):
                return time.monotonic() - start
            if "Welcome to lilbee" in screen:
                self.esc()
                time.sleep(1.2)
            elif not rx.search(screen):
                time.sleep(POLL_INTERVAL)
                continue
            time.sleep(POLL_INTERVAL)
        raise Timeout(f"chat never appeared within {timeout}s\n--- screen ---\n{self.screen()}")

    def observed_mode(self) -> str | None:
        """NORMAL or INSERT as the tab strip currently reports it, if it is showing."""
        m = re.search(r"\b(INSERT|NORMAL)\b", self.screen())
        return m.group(1) if m else None

    def insert(self, *, clear: bool = True, timeout: float = 8.0) -> None:
        """Enter insert mode, whatever mode we are in now, without typing an ``i``.

        ``i`` is a mode switch only from NORMAL. The app inserts it as a character when
        already in INSERT (chat.on_key), and lilbee starts in INSERT -- which is where
        the stray ``i``, typed and then wiped by the following C-u, came from on every
        reel that asked a question. From NORMAL with focus inside a drawer the app
        swallows it instead and never switches at all, so pressing and hoping leaves the
        reel typing its question into key bindings.

        Neither case is detectable after the fact, so the mode is established before the
        keypress rather than corrected after it.
        """
        seen = self.observed_mode()
        if seen:
            self._mode = seen
        if self._mode != "INSERT":
            self.key("i", after=0.5)
            if seen:
                # A chip was on screen a moment ago, so one is expected after the switch.
                # Not arriving means the key went somewhere else -- focus inside a drawer,
                # where the app swallows i -- and typing a question into key bindings is
                # worth failing the take over.
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if self.observed_mode() == "INSERT":
                        break
                    time.sleep(POLL_INTERVAL)
                else:
                    raise Timeout(
                        "'i' did not reach insert mode within "
                        f"{timeout}s -- focus is most likely inside a drawer, where the "
                        f"app swallows it\n--- screen ---\n{self.screen()}")
            else:
                # No chip to confirm against: the placement drawer takes the space the
                # tab strip renders it in, which is a normal state for these reels rather
                # than a fault. Requiring one here failed five takes whose input was
                # working. Callers verify by effect instead -- ask() checks the question
                # reached the input, and the slash-command sites wait on what the command
                # opens -- so a swallowed keypress still surfaces, one step later.
                time.sleep(0.4)
            self._mode = "INSERT"
        if clear:
            self.key("C-u", after=0.3)

    def ask(self, question: str, *, rate: float = 0.045, dwell: float = 0.9) -> None:
        """Put the chat input in insert mode, type a question, and confirm it arrived."""
        self.insert()
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

    def await_task(self, kind: str = "add|crawl", *, appear: float = 90.0,
                   finish: float = 1800.0) -> float:
        """Wait for a background task to show up, and only then for it to finish.

        "all caught up" is what the Task Center displays when it holds *nothing*, so
        accepting it as a completion signal matches the moment before the task has been
        registered rather than the moment after it finished. A crawl reel read it 88
        seconds in, concluded the crawl was done, asked its question against an empty
        index and got an answer with no citation -- the same shape as waiting on a phrase
        from the expected answer and matching the question instead.

        Requiring the task to appear first makes the empty state unmatchable at the point
        where it would lie. After it has appeared, an emptied list genuinely does mean
        finished, so it stays a valid signal for the second wait.
        """
        start = time.monotonic()
        self.wait_for(rf"\b({kind})\b", timeout=appear)
        left = max(5.0, finish - (time.monotonic() - start))
        self.wait_for(rf"({kind})\s+(done|complete)|all caught up", timeout=left)
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
        # Escape out of the chat input is what leaves insert mode. It also dismisses
        # modals, where it does not, but insert() reconciles against the chip whenever
        # one is on screen, so guessing NORMAL here is self-correcting.
        self._mode = "NORMAL"

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

    def palette_to(self, screen: str, marker: str, timeout: float = 25.0) -> None:
        """Jump to a screen through the command palette rather than the nav ring.

        The ring walk reads the tab strip, and the strip does not always carry names: on a
        pod under engine load it rendered as bare separators, so a walk looking for "Chat"
        matched nothing and stepped into Settings instead. The palette matches on the
        command, not on a strip that may still be painting.
        """
        # Leave any focused field first. A walk that ends up in Settings lands focus in
        # an Input, and a focused Input swallows ctrl+p, so the palette never opens and
        # the fallback fails for a different reason than the walk did.
        self.esc(2)
        time.sleep(0.5)
        self.key("C-p", after=0.9)
        self.wait_for(r"Search for commands|Search for a command", timeout=15)
        self.type_text(screen, rate=0.045)
        time.sleep(0.7)
        self.key("Enter", after=1.0)
        self.wait_for(marker, timeout=timeout)

    def goto(self, screen: str, *, forward: bool = True, limit: int = 10,
             marker: str | None = None) -> None:
        """Walk the screen ring until `screen` is showing, reading the strip each step.

        Counting bracket presses is what the old tapes did and it is now wrong; the ring
        grew and the count no longer lands where the tape assumed.

        Falls back to the palette when the walk runs out of steps: the strip can render
        without names while the app is busy, and a reel should not die because a widget
        was mid-paint.
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
        try:
            self.palette_to(screen, probe)
            return
        except Timeout:
            pass
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
