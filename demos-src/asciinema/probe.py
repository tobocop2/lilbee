#!/usr/bin/env python3
"""Throwaway live prober: launch lilbee on the staged root, run a key script, dump screens.

Exists because the retired tapes' key sequences no longer land where they assumed, and
guessing costs a full take. Usage: probe.py '<key>,<key>,...' where a step is either a
tmux key name or 'wait:<seconds>' / 'type:<text>'.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import drive  # noqa: E402

STAGE = pathlib.Path.home() / ".cache/lilbee-reel/lilbee"


def main() -> None:
    steps = sys.argv[1].split(",") if len(sys.argv) > 1 else []
    s = drive.Session("reel-probe", 128, 41, pathlib.Path("/tmp/probe.cast"))
    s.start("lilbee", env={"LILBEE_DATA": str(STAGE)})
    try:
        s.wait_for(r"personal encyclopedia", timeout=90)
        time.sleep(1.0)
        s.esc(2)
        time.sleep(0.6)
        print("=== after boot ===")
        print(s.screen())
        for step in steps:
            step = step.strip()
            if step.startswith("wait:"):
                time.sleep(float(step[5:]))
            elif step.startswith("type:"):
                s.type_text(step[5:], rate=0.02)
                time.sleep(0.6)
            elif step == "esc":
                s.esc()
                time.sleep(0.4)
            else:
                s.key(step, after=0.9)
            print(f"=== after {step!r} ===")
            print(s.screen())
    finally:
        s.kill()


if __name__ == "__main__":
    main()
