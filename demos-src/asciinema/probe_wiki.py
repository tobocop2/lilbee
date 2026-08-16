#!/usr/bin/env python3
"""Dump the wiki screen at each choreography step so the reel keystrokes are exact.

Runs on the pod against the real TUI. Does NOT generate (stops at the confirm
dialog) so it can be run repeatedly while Jupiter is a stub.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import drive  # noqa: E402

ENV = {
    "LILBEE_DATA": "/root/reel-data",
    "LILBEE_WIKI": "true",
    "LILBEE_CHAT_MODEL": os.environ["LILBEE_CHAT_MODEL"],
    "LILBEE_EMBEDDING_MODEL": os.environ["LILBEE_EMBEDDING_MODEL"],
    "LILBEE_CHAT_N_CTX_TARGET": "16384",
    "VIRTUAL_ENV": "/root/lilbee/.venv",
}


def dump(s, label):
    print(f"\n===== {label} =====")
    print(s.screen())


s = drive.Session("probe-wiki", 128, 44, pathlib.Path("/root/out/probe.cast"))
s.start("/root/lilbee/.venv/bin/lilbee", env=ENV)
try:
    s.wait_for(r"personal encyclopedia|Welcome to lilbee", timeout=150)
    time.sleep(1.3)
    if "Welcome to lilbee" in s.screen():
        s.esc(2)
        time.sleep(1.6)
    else:
        s.esc(2)
        time.sleep(0.7)
    dump(s, "HOME")

    for _ in range(5):
        s.key("]", after=0.8)
    s.wait_for(r"Not written yet|Filter pages", timeout=90)
    time.sleep(1.6)
    dump(s, "WIKI (tree, unfiltered)")

    # focus search, type jupiter
    s.key("/", after=0.6)
    time.sleep(0.4)
    for ch in "jupiter":
        s.key(ch, after=0.12)
    time.sleep(1.4)
    dump(s, "WIKI (filtered jupiter)")

    # Tab moves focus from the search input to the tree, preserving the filter.
    s.key("Tab", after=0.6)
    time.sleep(0.5)
    dump(s, "AFTER TAB (tree focused?)")
    s.key("g", after=0.5)
    s.key("j", after=0.5)
    dump(s, "AFTER g,j (Log?)")
    s.key("j", after=0.5)
    dump(s, "AFTER j (Not written yet?)")
    s.key("l", after=0.6)
    time.sleep(0.5)
    dump(s, "AFTER l (expanded, Jupiter child?)")
    s.key("j", after=0.5)
    dump(s, "AFTER j (Jupiter selected?)")
    s.key("enter", after=1.2)
    time.sleep(1.2)
    dump(s, "AFTER ENTER (confirm dialog?)")

    # Confirm and generate live, then discover how to reopen the finished page.
    s.key("y", after=0.8)
    s.wait_for(r"Write Jupiter|Writing", timeout=30)
    dump(s, "GEN STARTED (task bar)")
    waited = s.wait_for(r"Wrote Jupiter", timeout=400)
    print(f"\n[gen wall ~ {waited:.0f}s]")
    time.sleep(1.5)
    dump(s, "AFTER Wrote (tree reloaded, filter state?)")
    # Try: cursor may still be on Jupiter -> Enter reopens as a page.
    s.key("enter", after=1.2)
    time.sleep(1.0)
    dump(s, "AFTER ENTER on current node")
    # Fallback: re-filter and navigate under Entities.
    s.key("/", after=0.5)
    for _ in range(12):
        s.key("backspace", after=0.05)
    for ch in "jupiter":
        s.key(ch, after=0.1)
    time.sleep(1.2)
    s.key("Tab", after=0.5)
    s.key("g", after=0.4)
    dump(s, "REFILTER post-gen (structure?)")
    for lbl, seq in (("j1", ["j"]), ("j2", ["j"]), ("l", ["l"]), ("j3", ["j"]), ("j4", ["j"])):
        s.key(*seq, after=0.4)
        dump(s, f"post-gen nav {lbl}")
    s.key("enter", after=1.2)
    time.sleep(1.2)
    dump(s, "post-gen AFTER ENTER (page rendered?)")
finally:
    s.kill()
