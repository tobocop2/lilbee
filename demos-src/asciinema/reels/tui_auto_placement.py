#!/usr/bin/env python3
"""tui-auto-placement: a model too big for one card, spread across three without asking.

The companion to tui-manual-placement, and deliberately the shorter of the two: nothing
is configured here. lilbee sizes each role with gguf-parser, keeps headroom on every
card, tensor-splits the chat model across the fewest GPUs that fit, and places the
embedder alongside it. The reel opens the placement drawer only to show what was decided,
then asks a question a mechanic would ask and lets the answer arrive while all three
rows carry weight.

Three cards is a measured requirement, not a preference. Llama 3.3 70B at Q4_K_M is
39.6 GiB; on two 4090s that leaves so little room for KV that the served context collapses
to 512 tokens -- under lilbee's own retrieval prompt -- so the model loads, splits, fills
41 GB and answers nothing. gguf-parser puts it at roughly 22.3 GiB per card on two and
15 GiB on three at an 8k context.

Prerequisite: the manual must already be indexed into ROOT. This reel does not ingest;
the self-index reel covers that, and ingesting with a 70B resident is its own problem.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-auto-placement"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Placement", "auto", "70B", "cv-manual", "Sources:")
BEATS = (
    ("first card listed", r"CUDA0"),
    ("second card listed", r"CUDA1"),
    ("third card listed", r"CUDA2"),
    ("the model split across them", r"Llama-3\.3-70B"),
    ("asked a multi-part question", r"overheats in stop-and-go"),
    ("answered with a citation", r"Sources:"),
)

TAIL_FORBID = ("Cancel stream",)
# Generation is never compressed. How fast the answer actually arrives is part of what
# these reels are showing -- a 70B split across consumer cards answering in real time is
# the claim -- and a timelapse over the stream makes the model look faster than it is.
# Only the model load compresses; that is minutes of a progress bar and says nothing.
SPEED_WINDOWS = ("load",)
# "gen" covers the stream itself, "answer" the readable hold after it finishes. Protected
# spans are exempt from the auto wait/slow detection too, so token streaming cannot be
# swept up as a slow section.
PROTECT_WINDOWS = ("gen", "answer")

ROOT = "/workspace/reelroot"
QUESTION = ("a customer says the engine overheats in stop-and-go traffic and the "
            "temperature gauge climbs into the red. what does the manual say to do "
            "right then, what can cause it, and what should I check before putting "
            "the car back on the road?")


def record(cast: pathlib.Path) -> dict:
    s = drive.Session("reel-autoplace", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": ROOT})
    try:
        timings["boot"] = s.await_chat(timeout=300)
        time.sleep(1.5)
        s.repaint()
        s.wait_for(drive.Session.CHAT_MARKER, timeout=30)
        time.sleep(0.6)
        s.mark("boot_end")

        # 1. Wait for the engine to be resident before opening the drawer: placement
        # reports "probing GPUs" until then, and a drawer full of placeholders is not
        # what this reel is about. 39.6 GiB off disk across three cards takes minutes.
        s.mark("load_start")
        try:
            s.wait_for(r"warming up|starting engine", absent=True, timeout=900)
        except drive.Timeout:
            pass
        time.sleep(3.0)
        s.mark("load_end")

        # 2. What lilbee decided, unprompted. Held long enough to read all three rows.
        s.key("C-g", after=1.5)
        s.wait_for(r"CUDA0", timeout=120)
        time.sleep(3.0)
        s.wait_for(r"CUDA2", timeout=60)
        time.sleep(5.0)

        # 3. The payoff: a mechanic's question, answered while every card carries weight.
        s.ask(QUESTION, rate=0.045)
        s.mark("gen_start")
        timings["answer"] = s.await_answer()
        s.mark("gen_end")
        s.mark("answer_start")
        time.sleep(8.0)
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
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                              else "/tmp/tui-auto-placement.cast")))
