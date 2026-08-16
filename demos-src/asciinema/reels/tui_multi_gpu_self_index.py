#!/usr/bin/env python3
"""tui-multi-gpu-self-index: lilbee indexing its own source across two cards.

The corpus is a slice of lilbee's own source -- the retrieval and core packages -- which
makes the reel self-referential on purpose: the thing being indexed is the thing doing the
indexing, so nobody has to take a staged corpus on trust. A small slice, for two reasons. The full package took over forty minutes to embed, which
is not a reel. And with a 70B resident across both cards the ingest twice took the whole
recording down about nineteen minutes in -- tmux gone, both GPUs back to idle, cast
truncated -- so the corpus is now sized to finish in a couple of minutes, well inside
that window. The underlying crash is filed separately; this reel routes around it rather
than waiting on it. The placement drawer stays open through the ingest, so both cards are visible
doing work rather than one card doing work and another sitting idle in a table.

Runs Llama 3.3 70B at Q4_K_M -- about 40 GB of weights, which no single consumer card
holds. Two RTX 5090s do, and the placement drawer shows the model genuinely split across
them rather than one card working while the other idles. That split is the subject; an 8B
that fits on either card alone would not have one.

Recorded on a pod. On one card this reel has no subject.
"""
from __future__ import annotations

import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-multi-gpu-self-index"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Placement", "Background Tasks", "Sources:")
BEATS = (
    ("first card listed", r"CUDA0"),
    ("second card listed", r"CUDA1"),
    ("the model split across them", r"Llama-3\.3-70B"),
    ("ingest finished", r"add\s+(done|complete)|all caught up"),
    ("cited answer", r"Sources:"),
)

TAIL_FORBID = ("Cancel stream",)
SPEED_WINDOWS = ("ingest", "gen")

ROOT = "/workspace/reelroot"
CHAT_MODEL = "bartowski/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
CORPUS = "/workspace/corpus-tiny"
QUESTION = "how does lilbee decide which GPU a model runs on?"


def record(cast: pathlib.Path) -> dict:
    s = drive.Session("reel-selfindex", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": ROOT})
    try:
        # A fresh data root opens on the first-run wizard, which the chat marker never
        # matches. Dismiss it if it is there before waiting for chat.
        try:
            s.wait_for(r"Welcome to lilbee", timeout=25)
            s.esc()
            time.sleep(1.0)
        except drive.Timeout:
            pass
        timings["boot"] = s.await_chat(timeout=300)
        time.sleep(1.5)
        s.repaint()
        s.wait_for(r"personal encyclopedia|Slash commands", timeout=30)
        time.sleep(0.6)
        s.mark("boot_end")

        # 1. Both cards, before any work.
        s.key("C-g", after=1.5)
        s.wait_for(r"Placement", timeout=40)
        time.sleep(2.4)

        # 2. Point it at its own source.
        s.insert()
        s.type_text(f"/add {CORPUS}", rate=0.05)
        time.sleep(1.0)
        s.key("Enter", after=1.2)
        time.sleep(2.5)

        # 3. The Task Center, with the placement drawer still up beside it.
        s.esc()
        time.sleep(0.4)
        s.key("t", after=0.9)
        s.wait_for(r"Background Tasks", timeout=40)
        s.mark("ingest_start")
        # Fail fast on a failed add instead of waiting out the timeout. lilbee marks the
        # whole add "failed" when any single file yields no text, so two empty __init__.py
        # files were enough to leave this reel waiting forty minutes for a "done" that was
        # never coming.
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            screen = s.screen()
            if re.search(r"add\s+(done|complete)|all caught up", screen):
                break
            if re.search(r"add\s+failed", screen):
                raise SystemExit(f"ingest failed on camera:\n{screen}")
            time.sleep(1.0)
        timings["ingest"] = 0.0
        s.mark("ingest_end")
        time.sleep(2.4)

        # 4. Ask the source about itself.
        s.goto("Chat", forward=False, limit=8, marker=r"Slash commands")
        time.sleep(0.8)
        s.ask(QUESTION, rate=0.05)
        s.mark("gen_start")
        timings["answer"] = s.await_answer()
        s.mark("gen_end")
        time.sleep(3.5)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                              else "/tmp/tui-multi-gpu-self-index.cast")))
