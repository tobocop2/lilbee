#!/usr/bin/env python3
"""wiki-lazy: lilbee writes a wiki page on demand, the moment you open it.

The browse tree is built from LLM-free NER extraction, so every entity appears
instantly, ungenerated ones as dim "not written" stubs. Opening a stub asks
first (it spends GPU time), then generates that one page live through citation
verification into a fully cited, cross-linked article. This reel drives that
loop end to end: Chat -> Wiki -> open a stub -> confirm -> watch it write ->
read the cited page. Recorded on a GPU pod; the generation window is marked so
post-trim compresses it 6x instead of shipping a progress bar.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "wiki-lazy"
COLS, ROWS = 128, 44
MUST_STRINGS = ("not written", "Write this page?", "gas giant")
FORBID_STRINGS = ("Traceback", "Could not write")
# The live 70B generation (model load + write) is the payoff, but in real time it
# is long. Bracketed by gen_start/gen_end and compressed hard so it reads as a
# moment, not a wait. Tuned at render time (--no-record), so the single take is
# never re-shot to adjust it.
SPEED_WINDOWS = ("gen",)
SPEED_FACTOR = 16

# The command + data root differ between the pod (where record() runs against a
# real GPU) and any local dry render. Both come from the environment so the
# module itself stays host-agnostic.
LILBEE = os.environ.get("REEL_LILBEE_CMD", "lilbee")
DATA = os.environ.get("REEL_LILBEE_DATA", str(pathlib.Path.home() / "lilbee-wiki-reel" / "data"))

ENV = {
    "LILBEE_DATA": DATA,
    "LILBEE_WIKI": "true",
}
# On the pod the chat/embed models are served from explicit gguf paths and the
# context target is widened so one entity's cross-source chunks fit. These are
# only present in the environment when recording against the GPU.
for _k in ("LILBEE_CHAT_MODEL", "LILBEE_EMBEDDING_MODEL", "LILBEE_CHAT_N_CTX_TARGET", "VIRTUAL_ENV"):
    if os.environ.get(_k):
        ENV[_k] = os.environ[_k]

# Down-arrow counts into the page tree, filled in after inspecting the live tree
# on the pod (the stub sits under the "Not written yet" group; after generation
# the same title moves under Entities, so the second hop differs).
STUB_HOPS = int(os.environ.get("REEL_STUB_HOPS", "0"))
PAGE_HOPS = int(os.environ.get("REEL_PAGE_HOPS", "0"))


def _open_generated_page(s) -> None:
    """Reopen Jupiter after generation, when it is a real page not a stub.

    Sequence and hop counts are set from probe3's captured post-generation tree.
    """
    # placeholder; overwritten once probe3 confirms the post-gen structure
    s.key("enter", after=1.0)


def record(cast: pathlib.Path) -> dict:
    s = drive.Session("reel-wikilazy", COLS, ROWS, cast)
    t: dict[str, float] = {}
    s.start(LILBEE, env=ENV)
    try:
        s.wait_for(r"personal encyclopedia|Welcome to lilbee", timeout=150)
        time.sleep(1.3)
        if "Welcome to lilbee" in s.screen():
            s.esc(2)
            time.sleep(1.6)
        else:
            s.esc(2)
            time.sleep(0.7)
        s.mark("boot_end")

        # One hop straight into the Wiki view. The reel is about the wiki, so no
        # time is spent on the other screens.
        for _ in range(5):
            s.key("]", after=0.8)
        s.wait_for(r"Not written yet|Filter pages", timeout=90)
        time.sleep(1.8)

        # Search for the page. Typing the query is the only driver-paced motion,
        # so the burst rate doubles as the motion_fps window.
        s.mark("browse_start")
        s.key("/", after=0.6)
        time.sleep(0.4)
        s.key(*"jupiter", after=0.11)
        s.wait_for(r"Not written yet", timeout=15)
        time.sleep(1.4)
        s.mark("browse_end")

        # Tab moves focus into the filtered tree without clearing the query. The
        # stub group is collapsed, so expand it, land on the stub, and open it.
        s.key("Tab", after=0.6)
        s.key("g", after=0.45)
        s.key("j", after=0.45)
        s.key("j", after=0.45)
        s.key("l", after=0.6)
        s.key("j", after=0.5)
        s.key("enter", after=1.0)
        s.wait_for(r"Write this page\?", timeout=25)
        time.sleep(2.6)

        # Confirm. The Task Center spins on "Write Jupiter" while the 70B writes
        # the page from its cross-source chunks; this window is compressed in post.
        s.mark("gen_start")
        s.key("y", after=0.7)
        s.wait_for(r"Write Jupiter|Writing", timeout=20)
        t["gen"] = s.wait_for(r"Wrote Jupiter", timeout=400)
        time.sleep(1.2)
        s.mark("gen_end")

        # POST-GEN reopen sequence: filled in from probe3 (Jupiter is now a real
        # page, so the tree structure and hops differ from the stub open).
        _open_generated_page(s)
        s.wait_for(r"gas giant|fifth planet", timeout=25)
        time.sleep(4.0)
        s.mark("payload_end")
        time.sleep(0.8)
    finally:
        s.kill()
    t["marks"] = dict(s.marks)
    t["motion_spans"] = list(s.motion_spans)
    return t
