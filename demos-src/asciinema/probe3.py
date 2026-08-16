#!/usr/bin/env python3
"""Verify content-pane scrolling and the GPU drawer (C-g) during generation."""
from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, "/root/reelkit")
import drive  # noqa: E402

ENV = {
    "LILBEE_DATA": "/root/reel-data", "LILBEE_WIKI": "true",
    "LILBEE_CHAT_MODEL": os.environ["LILBEE_CHAT_MODEL"],
    "LILBEE_EMBEDDING_MODEL": os.environ["LILBEE_EMBEDDING_MODEL"],
    "LILBEE_CHAT_N_CTX_TARGET": "16384", "VIRTUAL_ENV": "/root/lilbee/.venv",
}


def dump(s, label):
    print(f"\n===== {label} =====")
    print(s.screen())


s = drive.Session("probe3", 128, 44, pathlib.Path("/root/out/probe3.cast"))
s.start("/root/lilbee/.venv/bin/lilbee", env=ENV)
try:
    s.wait_for(r"personal encyclopedia|Welcome to lilbee", 150); time.sleep(1.2)
    if "Welcome to lilbee" in s.screen(): s.esc(2); time.sleep(1.6)
    else: s.esc(2); time.sleep(0.6)
    for _ in range(5): s.key("]", after=0.6)
    s.wait_for(r"Not written yet|Filter pages", 90); time.sleep(1.2)

    # open Earth (row 15) then try to scroll its content
    s.key("g", after=0.4); s.key(*(["j"] * 15), after=0.04); s.key("enter", after=0.9)
    time.sleep(1.5); dump(s, "EARTH top")
    s.key("Tab", after=0.5)
    dump(s, "after Tab (focus?)")
    s.key(*(["Down"] * 12), after=0.06)
    time.sleep(0.6); dump(s, "after Tab+Down12 (scrolled?)")
    # try alternative: pagedown without tab
    s.key("pagedown", after=0.4)
    time.sleep(0.4); dump(s, "after pagedown")

    # create jupiter -> task center -> C-g GPU panel
    s.key("/", after=0.4); s.key("C-u", after=0.2); s.key(*"jupiter", after=0.11)
    time.sleep(1.0)
    s.key("Tab", after=0.4); s.key("g", after=0.35); s.key("j", after=0.35)
    s.key("j", after=0.35); s.key("l", after=0.5); s.key("j", after=0.4); s.key("enter", after=0.9)
    s.wait_for(r"Write this page\?", 25); time.sleep(1.2)
    s.key("y", after=0.7)
    s.wait_for(r"Write Jupiter|Writing", 20); time.sleep(0.6)
    s.key("t", after=0.8)
    s.wait_for(r"Background Tasks", 15); time.sleep(1.2)
    s.key("C-g", after=0.8)
    time.sleep(2.0); dump(s, "TASK CENTER + C-g (GPU panel? util bars?)")
    w = s.wait_for(r"1 done", 400); print(f"\n[gen wall ~ {w:.0f}s]")
    time.sleep(1.0); dump(s, "gen done (GPU panel still there?)")
finally:
    s.kill()
