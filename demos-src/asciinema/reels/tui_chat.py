#!/usr/bin/env python3
"""tui-chat: one question against an indexed manual, and the cited answer.

The narrowest reel in the set on purpose. No navigation, no setup, no Task Center: a
question goes in, an answer comes back with a page reference, and that is the whole
claim. Everything else in the set exists to show how a document got there; this one shows
what it is for.

The placement drawer is open so the card doing the work is visible next to the answer,
and the generation window is compressed with a label rather than played out, because the
part worth watching is the answer rather than the wait for it.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import drive  # noqa: E402

NAME = "tui-chat"
COLS, ROWS = 128, 41
MUST_STRINGS = ("towing a trailer", "Sources:", "cv-manual.pdf")
BEATS = (
    ("question asked", r"towing a trailer"),
    ("placement visible while answering", r"Placement"),
    ("cited answer", r"Sources:"),
)

TAIL_FORBID = ("Cancel stream",)

STAGE = pathlib.Path.home() / ".cache/lilbee-reel/lilbee"
ROOT = pathlib.Path.home() / ".cache/lilbee-reel/chat"
# Mistral rather than Qwen, so the set shows more than one family answering questions.
CHAT_MODEL = "bartowski/Mistral-7B-Instruct-v0.3-GGUF/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
QUESTION = "what does the manual say about towing a trailer?"


def record(cast: pathlib.Path) -> dict:
    if not (STAGE / "data").exists():
        raise SystemExit("staged root missing; build it first")
    shutil.rmtree(ROOT, ignore_errors=True)
    shutil.copytree(STAGE, ROOT)
    cfg = ROOT / "config.toml"
    lines = [ln for ln in cfg.read_text().splitlines() if not ln.startswith("chat_model")]
    cfg.write_text(f'chat_model = "{CHAT_MODEL}"\n' + "\n".join(lines) + "\n")

    s = drive.Session("reel-chat", COLS, ROWS, cast)
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

        s.key("C-g", after=1.0)
        s.wait_for(r"Placement", timeout=25)
        time.sleep(1.4)
        s.ask(QUESTION, rate=0.05)
        s.mark("gen_start")
        timings["answer"] = s.await_answer()
        s.mark("gen_end")
        time.sleep(2.0)

        # Read back through the answer. Also the reel's only driver-paced motion.
        s.esc()
        time.sleep(0.4)
        s.key(*(["k"] * 12), after=0.045)
        time.sleep(0.8)
        s.key(*(["j"] * 12), after=0.045)
        time.sleep(2.2)

        timings["total"] = time.monotonic() - t0
        s.mark("payload_end")
        time.sleep(1.0)
    finally:
        s.kill()
    timings["marks"] = dict(s.marks)
    timings["motion_spans"] = list(s.motion_spans)
    return timings


if __name__ == "__main__":
    print(record(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tui-chat.cast")))
