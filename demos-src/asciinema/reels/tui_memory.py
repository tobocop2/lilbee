#!/usr/bin/env python3
"""tui-memory: lilbee remembering things you tell it, and using them.

Arrives with memory already on. An earlier cut turned it on inside the reel, which meant
opening on a rejected command and an error toast -- a demonstration of the feature
failing, whatever the following beats then showed. Where the switch lives belongs in the
settings reel, which walks the Memory pane anyway.

Three beats. `/remember` saves a fact and a preference; `/memories` opens the browser
showing both; and a question is answered using what was remembered rather than what was
retrieved from a document, which is the distinction the feature exists to draw.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-memory"
COLS, ROWS = 128, 41
MUST_STRINGS = ("remember", "1989 Caprice", "300,000 miles")
BEATS = (
    ("fact remembered", r"1989 Caprice"),
    ("preference remembered", r"short paragraphs"),
    # The browser is a table; "Memories" is nowhere on it. Its columns are.
    ("memory browser lists both", r"Shared"),
    ("asked in a fresh conversation", r"Slash commands"),
    ("answered from memory", r"Caprice"),
)

# The reel must never show memory being refused. That is what the previous cut opened on.
FORBID_STRINGS = ("Memory is off",)

TAIL_FORBID = ("Cancel stream",)
SPEED_WINDOWS = ("gen",)
# The answer, once it has arrived, is never compressed: it is the thing being proved.
PROTECT_WINDOWS = ("answer",)

ROOT = pathlib.Path.home() / ".cache/lilbee-reel/memory"
STAGE = pathlib.Path.home() / ".cache/lilbee-reel/lilbee"

FACT = "my car is a 1989 Caprice with 300,000 miles on it"
PREF = "pref: answer me in short paragraphs, no bullet lists"
QUESTION = "what car do I drive, and how should you answer me?"


def _enable_memory(cfg: pathlib.Path) -> None:
    """Set memory_enabled at top level, above the first table header.

    Appending is wrong: the staged config ends with a [linked_roots] table, so an
    appended key lands inside that table and silently does nothing.
    """
    lines = [ln for ln in cfg.read_text().splitlines()
             if not ln.strip().startswith("memory_enabled")]
    at = next((i for i, ln in enumerate(lines) if ln.strip().startswith("[")), len(lines))
    lines.insert(at, "memory_enabled = true")
    cfg.write_text("\n".join(lines) + "\n")


def record(cast: pathlib.Path) -> dict:
    shutil.rmtree(ROOT, ignore_errors=True)
    shutil.copytree(STAGE, ROOT)
    # Gemma, so the set does not read as one model recorded many times.
    cfg = ROOT / "config.toml"
    lines = [ln for ln in cfg.read_text().splitlines() if not ln.startswith("chat_model")]
    cfg.write_text('chat_model = "ggml-org/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0.gguf"\n' + "\n".join(lines) + "\n")
    for stale in (ROOT / "memories", ROOT / "data/memories"):
        shutil.rmtree(stale, ignore_errors=True)
    _enable_memory(cfg)

    s = drive.Session("reel-memory", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(ROOT)})
    try:
        timings["boot"] = s.await_chat(timeout=120)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia|Slash commands", timeout=20)
        time.sleep(0.6)
        s.mark("boot_end")

        # 1. Save a fact and a preference.
        for text in (FACT, PREF):
            s.insert()
            s.type_text(f"/remember {text}", rate=0.045)
            time.sleep(0.8)
            s.key("Enter", after=1.2)
            time.sleep(1.6)

        # 2. The browser, holding long enough to read both entries.
        s.insert()
        s.type_text("/memories", rate=0.05)
        time.sleep(0.6)
        s.key("Enter", after=1.2)
        s.wait_for(r"Caprice", timeout=30)
        time.sleep(3.4)
        s.key(*(["j"] * 6), after=0.06)
        time.sleep(1.6)
        s.esc()
        time.sleep(0.8)

        # 3. The proof. Asking in the same conversation proves nothing: the facts are a
        # few lines up the transcript, and any model would read them back. So the reel
        # opens a NEW chat first, where the history is empty, and asks there. An answer
        # that still knows the car can only have come from memory.
        s.goto("Chat", forward=False, limit=8, marker=r"Slash commands")
        time.sleep(0.6)
        s.key("C-o", after=1.0)          # sessions drawer
        time.sleep(1.0)
        s.key("C-n", after=1.2)          # new conversation, empty history
        time.sleep(1.6)
        s.esc()
        time.sleep(0.8)
        s.ask(QUESTION, rate=0.045)
        s.mark("gen_start")
        timings["answer"] = s.await_answer()
        s.mark("gen_end")
        # The answer is the payoff, so it is held at real speed and long enough to read
        # rather than hurried past by the timelapse.
        s.mark("answer_start")
        time.sleep(7.0)
        s.mark("answer_end")

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-memory.cast")))
