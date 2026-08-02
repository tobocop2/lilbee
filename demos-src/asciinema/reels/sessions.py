#!/usr/bin/env python3
"""sessions: conversations persist, and you can go back to one.

Builds its own evidence. The drawer opens empty on a fresh root, the reel holds three
separate conversations on camera, and only then does it open the drawer to find all three
and resume the first. A pre-populated list would prove nothing about whether anything was
saved.

A new conversation is started from inside the drawer: ^o then ^n, which reports "Started a
new chat". Pressing ^n on the chat screen does nothing, which is why an earlier take put
every question into one conversation and showed a list of one.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "sessions"
COLS, ROWS = 128, 41
MUST_STRINGS = ("towing a trailer", "resume", "Sources:")
# What this reel must actually show, in order. Checked against the rendered screens,
# because a green scorecard once shipped a version of this reel whose drawer listed one
# conversation instead of three.
BEATS = (
    ("first answer cited", r"Sources:"),
    ("a second conversation started", r"Started a new chat"),
    ("three conversations listed", r"(?s)msgs.*msgs.*msgs"),
    ("the first one resumed", r"towing a trailer"),
)

TAIL_FORBID = ("Cancel stream",)
SPEED_WINDOWS = ("gen", "gen2", "gen3")

ROOT = pathlib.Path.home() / ".cache/lilbee-reel/sessions"
STAGE = pathlib.Path.home() / ".cache/lilbee-reel/lilbee"

# Three short questions the corpus can actually answer, so the list has three distinct
# titles rather than three variations of one.
QUESTIONS = [
    "what does the manual say about towing a trailer?",
    "what should I do if the engine overheats?",
    "how do I use the tire pressure monitoring system?",
]


def _new_chat(s: drive.Session) -> None:
    """Start a fresh conversation from the drawer, then close it."""
    s.key("C-o", after=1.2)
    s.wait_for(r"resume|No saved conversations", timeout=25)
    time.sleep(1.2)
    s.key("C-n", after=1.2)
    s.wait_for(r"Started a new chat", timeout=20)
    time.sleep(0.8)


def record(cast: pathlib.Path) -> dict:
    # A copy of the staged root, so the drawer genuinely starts empty while the corpus is
    # already indexed.
    shutil.rmtree(ROOT, ignore_errors=True)
    shutil.copytree(STAGE, ROOT)
    # Mistral rather than Qwen: a different family, and three generations in one reel
    # want the faster model.
    cfg = ROOT / "config.toml"
    lines = [ln for ln in cfg.read_text().splitlines() if not ln.startswith("chat_model")]
    cfg.write_text('chat_model = "bartowski/Mistral-7B-Instruct-v0.3-GGUF/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"\n' + "\n".join(lines) + "\n")
    for stale in (ROOT / "sessions", ROOT / "data/sessions"):
        shutil.rmtree(stale, ignore_errors=True)

    s = drive.Session("reel-sessions", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(ROOT)})
    try:
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=120)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia", timeout=20)
        time.sleep(0.6)
        s.mark("boot_end")

        for i, question in enumerate(QUESTIONS, start=1):
            if i > 1:
                _new_chat(s)
            s.ask(question, rate=0.05)
            mark = "gen" if i == 1 else f"gen{i}"
            s.mark(f"{mark}_start")
            timings[f"answer{i}"] = s.await_answer()
            s.mark(f"{mark}_end")
            time.sleep(1.6)
            s.esc()
            time.sleep(0.4)

        # The drawer now holds all three, newest first, so the first question asked is the
        # last entry. Resume it.
        s.key("C-o", after=1.2)
        s.wait_for(r"resume", timeout=25)
        time.sleep(2.6)
        s.key("Down", after=0.7)
        time.sleep(0.7)
        s.key("Down", after=0.7)
        time.sleep(1.4)
        s.key("Enter", after=1.4)
        # A resumed conversation opens scrolled to its answer, so the question that names
        # it is above the fold and capture-pane cannot see it. Wait for the citations,
        # then scroll back up to show which conversation this actually is.
        s.wait_for(r"Sources:", timeout=60)
        time.sleep(1.6)
        s.key(*(["k"] * 18), after=0.045)
        s.wait_for(r"towing a trailer", timeout=20)
        time.sleep(3.2)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sessions.cast")))
