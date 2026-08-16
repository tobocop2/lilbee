#!/usr/bin/env python3
"""tui-manual-placement: override where each model runs, on a three-card machine.

Recorded on a pod because there is nothing to place on one card. The model is Llama 3.3
70B at Q4_K_M -- 39.6 GiB of weights that no single consumer card holds.

Three cards, not two, and that is a measured decision rather than a preference. On two
4090s the weights leave so little room for KV that the served context collapses to 512
tokens, below lilbee's own retrieval prompt, so the model loads, splits, occupies 41 GB
and cannot answer anything (filed separately). gguf-parser puts the same model at about
22.3 GiB per card on two and about 15 GiB per card on three at an 8k context. Three cards
is where this demo becomes honest.

The placement screen shows a row per role and a column per GPU, with -/+ for replicas,
and the reel moves chat and embedding onto the cards it chooses, previews the plan, and
applies it.

Bindings come from the screen's own footer: ctrl+r previews, ctrl+s applies, ctrl+x
returns to automatic placement.

Ends by asking the split model a question with the drawer still open. Without that the
reel was a config screen someone moved around in and then left -- reported as not
understandable at all, and fairly: splitting a model across two cards only means anything
if something then runs on it. The answer arriving while both CUDA rows carry weight is
the point of the whole screen. Auto is restored after, so the reel does not leave anyone
thinking manual placement is something you have to do.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-manual-placement"
COLS, ROWS = 128, 41
MUST_STRINGS = ("Placement", "Preview", "Apply", "replicas", "70B", "cv-manual")
BEATS = (
    ("first card listed", r"CUDA0"),
    ("second card listed", r"CUDA1"),
    ("third card listed", r"CUDA2"),
    ("the grid of roles", r"replicas"),
    # The reel pins chat to one card, which cannot hold it, and lilbee answers with the
    # arithmetic rather than failing at load time. That refusal is the demonstration: it
    # is the difference between a placement editor and a text box that accepts anything.
    # An earlier cut asserted the header flipped to "manual", which never happens
    # precisely because the plan was rejected -- the beat was wrong, not the take.
    ("an impossible layout refused with the shortfall", r"needs [\d.]+ GiB but device"),
    ("a plan previewed", r"Preview"),
    ("asked the split model a multi-step question", r"overheats in stop-and-go"),
    ("it answered while split", r"Sources:"),
)

TAIL_FORBID = ("Cancel stream",)
# The 35 GB load across two cards is minutes of a static screen; compress it.
# Generation is never compressed. How fast the answer actually arrives is part of what
# these reels are showing -- a 70B split across consumer cards answering in real time is
# the claim -- and a timelapse over the stream makes the model look faster than it is.
# Only the model load compresses; that is minutes of a progress bar and says nothing.
SPEED_WINDOWS = ("load",)
# "gen" covers the stream itself, "answer" the readable hold after it finishes. Protected
# spans are exempt from the auto wait/slow detection too, so token streaming cannot be
# swept up as a slow section.
PROTECT_WINDOWS = ("gen", "answer")
# Stop on the readable hold rather than running on through the return to
# automatic: that tail is a static answer with nothing happening in it.
END_MARK = "answer_end"
# The placement drawer coalesces repaints: nine rounds of Tab-and-toggle produce six
# measurable frames, and Tab alone produces none. There is no animation here to be choppy,
# which is the only thing the frame-rate floor exists to catch, so this reel declares the
# screen static rather than leaving the row permanently unmeasured. Content is covered by
# BEATS instead: both cards, the model split across them, preview, apply, an answer
# generated while split, and back to auto.
STATIC_BY_DESIGN = True

# PREREQUISITE: the Crown Vic manual must already be indexed into this root before the
# reel runs. This reel deliberately does not ingest -- the sibling self-index reel does,
# and ingesting with a 70B resident is what killed two takes there -- so the question at
# the end has nothing to retrieve unless the pod setup indexed the manual first.
ROOT = "/workspace/reelroot"
CHAT_MODEL = "bartowski/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
# A question sized to the model, which is the point of putting a 70B on two cards. It
# cannot be answered by retrieving one chunk: it needs the maintenance schedule and the
# severe-service conditions read together, then a judgement about which applies. A 4B
# reliably answers half of it. Deliberately prose rather than a specification table --
# sideways pages still extract as scrambled text, so a table question would fail for a
# reason that has nothing to do with the model (bb-depek).
QUESTION = ("a customer says the engine overheats in stop-and-go traffic and the "
            "temperature gauge climbs into the red. what does the manual say to do "
            "right then, what can cause it, and what should I check before putting "
            "the car back on the road?")


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
        timings["boot"] = s.await_chat(timeout=300)
        time.sleep(1.5)
        s.repaint()
        s.wait_for(r"personal encyclopedia|Slash commands", timeout=30)
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

        # Make ONE deliberate change, from a known-good starting point.
        #
        # An earlier cut toggled the grid blindly -- nine rounds of Tab/space -- which is
        # motion on camera and a coin flip in effect: it can land on a plan that does not
        # fit, and applying that leaves the cards empty. The recording then waits fifteen
        # minutes for an answer no model is loaded to give, which is exactly how the last
        # take was lost. Placement is a demonstration of control, so the reel has to look
        # like someone who knows what they want.
        #
        # ctrl+x restores automatic placement first, so whatever the grid inherited is
        # replaced by a layout known to fit. From there a single Tab/space moves one role,
        # which is legible on camera in a way that eighteen toggles never were.
        s.key("C-x", after=1.5)
        time.sleep(3.0)
        s.key("Tab", after=0.5)
        time.sleep(1.0)
        s.key("space", after=0.5)
        time.sleep(2.0)
        s.key("Tab", after=0.5)
        time.sleep(1.0)
        s.key("space", after=0.5)
        time.sleep(2.5)

        # 3. Drive it with the bindings the drawer advertises rather than by walking Tab.
        # ctrl+r previews, ctrl+s applies; those are screen bindings, so they work without
        # guessing which widget holds focus.
        s.key("C-r", after=1.5)
        time.sleep(4.0)
        s.key("C-s", after=1.5)
        time.sleep(5.0)
        # The apply reloads the engine. Wait for it to be resident again before asking:
        # a question typed into a reloading engine is the same lost take by another route.
        try:
            s.wait_for(r"warming up|starting engine", absent=True, timeout=900)
        except drive.Timeout:
            pass
        time.sleep(3.0)

        # 4. The payoff: ask it something with the drawer still open, so the answer
        # streams next to the two rows carrying the weights.
        s.ask(QUESTION, rate=0.045)
        s.mark("gen_start")
        timings["answer"] = s.await_answer()
        s.mark("gen_end")
        # Held at real speed: a 70B reasoning across two sections is what the two cards
        # are for, and hurrying past the answer wastes the demonstration.
        s.mark("answer_start")
        time.sleep(8.0)
        s.mark("answer_end")

        # 5. Back to automatic, after the demonstration rather than before it.
        s.key("C-x", after=1.5)
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
                              else "/tmp/tui-manual-placement.cast")))
