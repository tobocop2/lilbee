#!/usr/bin/env python3
"""tui-unsupported: lilbee tells you a model will not run before you download it.

The retired tape searched "deepseek-v4", a name that never existed, so the rows it showed
were unsupported only because their metadata was unreadable. This one uses MiniMax M3: a
real, current model whose GGUFs are on the Hub and whose architecture the bundled
llama.cpp build genuinely does not carry. The refusal is therefore about a model someone
would actually try to pull.

Beats: the pills on the cards, the same rows in list view, then the confirm dialog that
names the architecture and offers to pull anyway. The reel declines.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-unsupported"
COLS, ROWS = 128, 41
MUST_STRINGS = ("unsupported", "Architecture not supported", "minimax-m3")
BEATS = (
    ("hub results", r"unsupported"),
    ("refusal names the architecture", r"minimax-m3"),
    ("declined", r"Yes \(y\)|No \(n\)"),
)

# If the arch ever becomes supported the dialog stops appearing, and a reel about a
# refusal that silently became a download is worse than a failed take.
FORBID_STRINGS = ("Queued download: MiniMax",)

STAGE = pathlib.Path.home() / ".cache/lilbee-reel/lilbee"


def record(cast: pathlib.Path) -> dict:
    if not (STAGE / "data").exists():
        raise SystemExit("staged root missing; build it first")

    s = drive.Session("reel-unsupported", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(STAGE)})
    try:
        timings["boot"] = s.await_chat(timeout=90)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia|Slash commands", timeout=20)
        time.sleep(0.5)
        s.mark("boot_end")
        s.esc(2)
        time.sleep(0.5)

        s.key("]", after=0.5)
        s.wait_for(r"Discover", timeout=40)
        time.sleep(1.2)
        s.key("2", after=0.6)
        time.sleep(0.8)

        # 1. Ask the Hub for a model this build cannot run.
        s.key("/", after=0.6)
        s.type_text("minimax m3", rate=0.05)
        time.sleep(0.6)
        s.key("Enter", after=0.8)
        timings["hub_search"] = s.wait_for(r"unsupported", timeout=60)
        time.sleep(2.6)

        # 2. Scroll the results. The verdict is on every row, not just the first three,
        # and scrolling pulls further Hub pages in so the point is made on more than one
        # repo. It is also the only motion in the reel: without it the frame rate has too
        # few animating frames to measure and the gate refuses to score it.
        s.key("Tab", after=0.9)
        time.sleep(0.6)
        s.key(*(["j"] * 12), after=0.045)
        time.sleep(1.8)

        # 3. List view: the same verdict as inline tags rather than stacked pills.
        s.key("v", after=0.8)
        time.sleep(1.8)
        s.key(*(["j"] * 10), after=0.045)
        time.sleep(1.6)
        s.key(*(["k"] * 10), after=0.045)
        time.sleep(0.8)
        s.key("v", after=0.8)
        time.sleep(1.4)

        # 4. Pull it anyway and let lilbee say why not. This is the payload.
        s.key("Enter", after=1.0)
        s.wait_for(r"Architecture not supported", timeout=25)
        time.sleep(3.4)
        s.key("n", after=1.0)
        s.wait_for(r"Architecture not supported", absent=True, timeout=15)
        time.sleep(1.6)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-unsupported.cast")))
