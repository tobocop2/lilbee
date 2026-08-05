#!/usr/bin/env python3
"""Verify the warm-wait + gradual scroll-through actually reads the full article."""
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


def body(s):
    return [l.split("│", 1)[1].strip() for l in s.screen().splitlines()
            if "│" in l and l.split("│", 1)[1].strip()][2:6]


s = drive.Session("p5win", 128, 44, pathlib.Path("/root/out/probe5.cast"))
s.start("/root/lilbee/.venv/bin/lilbee", env=ENV)
try:
    s.wait_for(r"personal encyclopedia|Welcome to lilbee", 150); time.sleep(1.2)
    if "Welcome to lilbee" in s.screen(): s.esc(2); time.sleep(1.6)
    else: s.esc(2); time.sleep(0.6)
    for _ in range(5): s.key("]", after=0.6)
    s.wait_for(r"Not written yet|Filter pages", 90); time.sleep(1.2)
    s.wait_for(r"warming up", absent=True, timeout=200); time.sleep(1.5)
    print("WARM done")
    # open Earth via filter
    s.key("/", after=0.4); s.key("C-u", after=0.2); s.key(*"earth", after=0.11); time.sleep(1.0)
    s.key("Tab", after=0.4); s.key("G", after=0.5); s.key("enter", after=0.9); time.sleep(2.0)
    print("opened:", body(s))
    # gradual scroll-through, dump each step
    s.key("Tab", after=1.3)
    for i in range(6):
        s.key(*(["Down"] * 7), after=0.06); time.sleep(1.7)
        print(f"step {i+1}:", body(s))
finally:
    s.kill()
