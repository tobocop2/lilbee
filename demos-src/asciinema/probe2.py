#!/usr/bin/env python3
"""Verify the unfiltered-browse + task-center + return choreography on the pod."""
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


def open_row(s, row):
    s.key("g", after=0.4)
    s.key(*(["j"] * row), after=0.05)
    s.key("enter", after=0.9)


s = drive.Session("probe2", 128, 44, pathlib.Path("/root/out/probe2.cast"))
s.start("/root/lilbee/.venv/bin/lilbee", env=ENV)
try:
    s.wait_for(r"personal encyclopedia|Welcome to lilbee", 150); time.sleep(1.2)
    if "Welcome to lilbee" in s.screen(): s.esc(2); time.sleep(1.6)
    else: s.esc(2); time.sleep(0.6)
    for _ in range(5): s.key("]", after=0.6)
    s.wait_for(r"Not written yet|Filter pages", 90); time.sleep(1.2)

    for row, who in ((15, "Earth"), (29, "Mars"), (34, "Neptune")):
        open_row(s, row)
        time.sleep(1.6)
        dump(s, f"OPEN row {row} (expect {who})")

    # create Jupiter (stub filter path)
    s.key("/", after=0.4); s.key("C-u", after=0.2); s.key(*"jupiter", after=0.11)
    time.sleep(1.0)
    s.key("Tab", after=0.4); s.key("g", after=0.35); s.key("j", after=0.35)
    s.key("j", after=0.35); s.key("l", after=0.5); s.key("j", after=0.4); s.key("enter", after=0.9)
    s.wait_for(r"Write this page\?", 25); time.sleep(1.5); dump(s, "CONFIRM")
    s.key("y", after=0.7)
    s.wait_for(r"Write Jupiter|Writing", 20); time.sleep(0.6)
    s.key("t", after=0.8)
    s.wait_for(r"Background Tasks", 15); time.sleep(1.5); dump(s, "TASK CENTER running")
    w = s.wait_for(r"1 done", 400); print(f"\n[gen wall ~ {w:.0f}s]")
    time.sleep(1.4); dump(s, "TASK CENTER done")
    s.key("]", after=0.9)
    s.wait_for(r"Filter pages|Entities", 20); time.sleep(0.6); dump(s, "BACK to wiki via ]")
    s.key("/", after=0.4); s.key("C-u", after=0.2); s.key(*"jupiter", after=0.11)
    time.sleep(1.0); s.key("Tab", after=0.4); s.key("G", after=0.6); s.key("enter", after=1.0)
    time.sleep(1.5); dump(s, "FINAL page (gas giant?)")
finally:
    s.kill()
