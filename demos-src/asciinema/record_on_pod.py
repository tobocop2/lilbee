#!/usr/bin/env python3
"""Record a reel's cast on the pod (where the GPU is), for local rendering.

The reel module's record() drives the real lilbee TUI through drive.Session
(tmux + asciinema) here on the pod, so the frame timestamps are pod-local with
no network smear. The resulting .cast and .marks.json are copied back and
rendered/gated locally with `make_reel.py <name> --no-record`.
"""
from __future__ import annotations

import importlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

name = sys.argv[1]
mod = importlib.import_module(f"reels.{name.replace('-', '_')}")
out = pathlib.Path("/root/out")
out.mkdir(parents=True, exist_ok=True)
cast = out / f"{name}.cast"

timings = mod.record(cast)

marks_path = out / f"{name}.marks.json"
# drive.Session stores marks on the session; record() closed it, so re-read from
# the module if it exposed them. Fall back to timings.
payload = {"timings": timings}
print("RECORD_DONE", json.dumps(payload))
print("cast:", cast, cast.stat().st_size if cast.exists() else "MISSING")
