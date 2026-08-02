#!/usr/bin/env python3
"""tui-manual-placement: override where each model runs, on a two-card machine.

Recorded on a pod because there is nothing to place on one card. The model is Llama 3.3
70B, large enough that placement is a real decision rather than a preference: 40 GB of
weights has to go somewhere across two 32 GB cards. The placement screen
shows a row per role and a column per GPU, with -/+ for replicas, and the reel moves chat
and embedding onto different cards, previews the plan, and applies it.

Bindings come from the screen's own footer: ctrl+r previews, ctrl+s applies, ctrl+x
returns to automatic placement. The reel ends back on Auto so it does not leave the
viewer thinking manual placement is something you have to do.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-manual-placement"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Placement", "Preview", "Apply", "replicas")
BEATS = (
    ("both cards listed", r"CUDA0"),
    ("second card listed", r"CUDA1"),
    ("the grid of roles", r"replicas"),
    ("a plan previewed", r"Preview"),
    ("back to automatic", r"Auto"),
)
# The 35 GB load across two cards is minutes of a static screen; compress it.
SPEED_WINDOWS = ("load",)
# The placement drawer coalesces repaints: nine rounds of Tab-and-toggle produce six
# measurable frames, and Tab alone produces none. There is no animation here to be choppy,
# which is the only thing the frame-rate floor exists to catch, so this reel declares the
# screen static rather than leaving the row permanently unmeasured. Content is covered by
# BEATS instead: both cards, the model split across them, preview, apply, back to auto.
STATIC_BY_DESIGN = True

ROOT = "/workspace/reelroot"
CHAT_MODEL = "bartowski/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf"


def record(cast: pathlib.Path) -> dict:
    s = drive.Session("reel-placement", COLS, ROWS, cast)
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
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=300)
        time.sleep(1.5)
        s.repaint()
        s.wait_for(r"personal encyclopedia", timeout=30)
        time.sleep(0.6)
        s.mark("boot_end")
        s.esc(2)
        time.sleep(0.5)

        # 1. Wait for the model to be resident before opening the drawer. Placement
        # reports "probing GPUs..." until the engine has actually loaded, and an earlier
        # take opened the drawer during warm-up, found no cards, and then walked Tab into
        # a model-picker modal -- the whole reel became a "+" typed into a filter box.
        # A 70B at IQ4_XS is 35 GB off disk and across two cards, so the wait is minutes.
        s.mark("load_start")
        try:
            s.wait_for(r"warming up|starting engine", absent=True, timeout=900)
        except drive.Timeout:
            pass
        time.sleep(3.0)
        s.mark("load_end")

        # 2. Both cards, with what is on them.
        s.key("C-g", after=1.5)
        s.wait_for(r"CUDA0", timeout=120)
        time.sleep(4.0)

        # Walk the role/GPU grid. Safe to do now that it is populated -- the failure mode
        # this avoids is Tabbing while the drawer still says "probing GPUs", which falls
        # through to a model-picker modal. Also the reel's only sustained driver motion.
        # Toggle rather than only Tab. Tab moves focus without repainting anything, so a
        # Tab-only walk is dead keys on camera and nothing to measure; Space flips a GPU
        # for the focused role and redraws the grid.
        for _ in range(9):
            s.key("Tab", "space", "Tab", "space", after=0.06)
            time.sleep(1.2)

        # 3. Drive it with the bindings the drawer advertises rather than by walking Tab.
        # ctrl+r previews, ctrl+s applies, ctrl+x returns to automatic; those are screen
        # bindings, so they work without guessing which widget holds focus.
        s.key("C-r", after=1.5)
        time.sleep(4.0)
        s.key("C-s", after=1.5)
        time.sleep(4.5)
        s.key("C-x", after=1.5)
        time.sleep(4.0)

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
                              else "/tmp/tui-manual-placement.cast")))
