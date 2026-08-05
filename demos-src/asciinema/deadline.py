#!/usr/bin/env python3
"""Compute the idle-watchdog hard cap (seconds) for a pod from the sum of its
reels' take budgets — heavy multi-reel groups (B/C carry 3x 235B) run hours,
so a flat 4h cap would kill them mid-work. Capped at 7h absolute.

Usage: deadline.py <reel-name>...
"""
import sys

import yaml

m = yaml.safe_load(open("/root/kit/reels.yaml"))
tot = 1800  # materialize + boot slack
for reel in sys.argv[1:]:
    r = m["reels"].get(reel, {})
    w = r.get("windows") or {}
    d = r.get("duration_s") or {}
    tot += (int(w.get("boot", 120)) + 3 * int(d.get("max", 300)) + 600) * 3
print(min(tot, 25200))
