#!/usr/bin/env python3
"""tui-catalog: the model catalog, ending in a real download.

Six inner tabs (Discover, Chat, Embed, Vision, Rerank, Library), a grid/list toggle, a
search that falls through to the HF Hub when the local set has no match, and a pull that
finishes on camera.

Interaction notes worth the takes they cost:
  * Search is a focused Input. Enter inside it re-runs the search rather than acting on a
    card; Tab is what moves focus into the grid, and Enter there starts the pull.
  * Entering the Catalog costs about five seconds of frozen UI while it fetches the Hub
    index (filed against the app). Wait on content rather than sleeping through it.
  * SmolLM2 135M is the pull target because it is 0.1 GB and the installed set holds only
    the 360M, so the card genuinely reads as not-installed and the download is real.
"""
from __future__ import annotations

import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-catalog"
COLS, ROWS = 128, 41
MUST_STRINGS = ("SmolLM2 135M", "Search HuggingFace", "Installed")
# Both densities and a download that visibly progressed, since most of an earlier take
# was a static progress bar and every property row was green.
BEATS = (
    ("list view", r"Grid\s*.\s*List"),
    ("hub search", r"Search HuggingFace"),
    ("download running", r"\d+/\d+ MB|\d+\.\d+%"),
    ("download finished", r"download\s+(done|complete)"),
)

# The pull is a progress bar for most of its length; compress it like a generation.
SPEED_WINDOWS = ("pull",)

STAGE = pathlib.Path.home() / ".cache/lilbee-reel/lilbee"

TABS = [("2", "Chat"), ("3", "Embed"), ("4", "Vision"), ("5", "Rerank"), ("6", "Library")]

# A failed pull still leaves a task row, and the summary line always contains the word
# "done" ("0 running · 0 queued · 0 done"), so a loose match reports success on a reel
# whose download died. This once passed a take where the disk was full.
FORBID_STRINGS = ("download   failed", "I/O error downloading")


def _await_download(s: drive.Session, timeout: float = 420.0) -> float:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        screen = s.screen()
        if re.search(r"download\s+(done|complete|ready|installed)", screen, re.I):
            return time.monotonic() - start
        if re.search(r"download\s+failed|error downloading", screen, re.I):
            raise SystemExit(f"pull failed on camera:\n{screen}")
        time.sleep(0.5)
    raise drive.Timeout(f"download never finished\n{s.screen()}")


def _reset_pull_target() -> None:
    """Remove the model this reel pulls, so the pull is real every time.

    The precondition is part of the take. After one recording the model is installed, and
    the next take shows an instant no-op that still passes every property gate -- the reel
    looks fine and demonstrates nothing. Resetting by hand worked until it was forgotten,
    twice, so the reel does it itself.
    """
    import shutil

    models = pathlib.Path.home() / "Library/Application Support/lilbee/models"
    repo = models / "models--bartowski--SmolLM2-135M-Instruct-GGUF"
    for snap in repo.glob("snapshots/*/*Q4_K_M*"):
        snap.unlink(missing_ok=True)
    for blob in repo.glob("blobs/*"):
        if blob.is_file() and blob.stat().st_size > 90_000_000:
            blob.unlink()
    shutil.rmtree(models / "manifests/bartowski--SmolLM2-135M-Instruct-GGUF",
                  ignore_errors=True)


def record(cast: pathlib.Path) -> dict:
    if not (STAGE / "data").exists():
        raise SystemExit("staged root missing; build it first")
    _reset_pull_target()

    s = drive.Session("reel-catalog", COLS, ROWS, cast)
    timings: dict[str, float] = {}
    t0 = time.monotonic()
    s.start("lilbee", env={"LILBEE_DATA": str(STAGE)})
    try:
        timings["boot"] = s.wait_for(r"personal encyclopedia", timeout=90)
        time.sleep(1.2)
        s.repaint()
        s.wait_for(r"personal encyclopedia", timeout=20)
        time.sleep(0.5)
        s.mark("boot_end")
        s.esc(2)
        time.sleep(0.5)

        s.key("]", after=0.5)
        timings["catalog_open"] = s.wait_for(r"Discover", timeout=40)
        time.sleep(1.6)

        # 1. Every task the catalog knows how to fill, one tab at a time.
        for key, label in TABS:
            s.key(key, after=0.5)
            s.wait_for(label, timeout=30)
            time.sleep(1.5)

        # 2. Grid and list are the same rows at two densities. List first, since the
        # Library tab is where the local set is small enough for both to fit.
        # Both densities get the same treatment: the same dwell and the same scroll, so
        # the reel does not read as a grid demo with a list view mentioned in passing.
        s.key("v", after=0.6)
        time.sleep(2.0)
        s.key(*(["j"] * 24), after=0.045)
        time.sleep(1.6)
        s.key(*(["k"] * 24), after=0.045)
        time.sleep(1.2)
        s.key("v", after=0.6)
        time.sleep(2.0)
        s.key(*(["j"] * 24), after=0.045)
        time.sleep(1.6)

        # 3. Search on the Chat tab: the local set has SmolLM2 360M, the Hub has the rest.
        s.key("2", after=0.6)
        time.sleep(0.8)
        s.key("/", after=0.6)
        # Narrow enough that no installed model matches. Searching plain "smollm2" puts
        # the already-installed 360M in an Installed section above the Hub results, Tab
        # focuses that card, and Enter "pulls" a model that is already there: the task
        # completes instantly, nothing downloads, and the reel looks fine.
        s.type_text("smollm2-135m", rate=0.05)
        time.sleep(0.6)
        s.key("Enter", after=0.8)
        timings["hub_search"] = s.wait_for(r"SmolLM2 135M", timeout=60)
        time.sleep(2.0)

        # 4. Tab moves focus out of the search input and into the grid; Enter pulls. The
        # focused card advertises the action, so wait for that rather than assume focus.
        s.key("Tab", after=1.0)
        s.wait_for(r"Enter to install", timeout=15)
        time.sleep(0.8)
        s.key("Enter", after=1.0)
        s.wait_for(r"SmolLM2 135M\s+\d", timeout=30)
        time.sleep(1.5)

        # 5. The Task Center owns the download; watch it finish there rather than in a
        # one-line status.
        s.key("t", after=0.8)
        s.wait_for(r"Background Tasks", timeout=25)
        s.mark("pull_start")
        timings["download"] = _await_download(s)
        s.mark("pull_end")
        time.sleep(1.8)

        # 6. Back to the card, which now reads installed. That is the payoff.
        # `q` is back-to-chat, not back-to-previous, so walk the ring instead.
        s.goto("Catalog", forward=False, limit=6, marker=r"Discover")
        time.sleep(0.6)
        s.key("/", after=0.5)
        s.key("Enter", after=0.8)
        s.wait_for(r"SmolLM2 135M", timeout=60)
        time.sleep(2.4)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-catalog.cast")))
