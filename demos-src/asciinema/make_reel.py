#!/usr/bin/env python3
"""Record, render, trim and gate one reel, then write its scorecard.

One entry point so no reel can skip a step. The order is fixed: record the full cast,
render the whole thing, then trim in the frame domain (see frametrim). Marks recorded by
the driver are timestamps against the session clock, and agg preserves cast timing, so
frame offsets and mark offsets are the same clock.

Usage: make_reel.py <name> [--no-record]
``--no-record`` re-renders and re-gates the cast already in out/, which is what to use
when only the trim window or the gate thresholds changed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import shutil
import sys
import time

KIT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
OUT = KIT / "out"

import agg_finish  # noqa: E402
import frametrim  # noqa: E402
import gates  # noqa: E402


def build(name: str, *, record: bool = True) -> tuple[str, bool]:
    mod = importlib.import_module(f"reels.{name.replace('-', '_')}")
    cast = OUT / f"{name}.cast"
    marks_path = OUT / f"{name}.marks.json"
    OUT.mkdir(exist_ok=True)

    if record:
        t0 = time.monotonic()
        timings = mod.record(cast)
        timings["wall"] = time.monotonic() - t0
        marks_path.write_text(json.dumps(timings, indent=2))
        print(f"recorded {cast} in {timings['wall']:.0f}s")
    timings = json.loads(marks_path.read_text())
    marks = timings.get("marks", {})

    full = OUT / f"{name}-full"
    agg_finish.render(cast, full)
    gif = OUT / f"{name}.gif"
    # Head-trim to the mark, not to zero: the boot banner and shell prompt are not the
    # reel. A reel with no marks ships whole rather than guessing a window.
    # Compress the generation window if the reel marked one. On a laptop most of a chat
    # reel can be a progress bar, and the answer is the part worth watching.
    speedup = None
    if "gen_start" in marks and "gen_end" in marks:
        speedup = (marks["gen_start"], marks["gen_end"],
                   getattr(mod, "SPEED_FACTOR", 6))
    info = frametrim.trim_gif(full.with_suffix(".gif"), gif,
                              start=marks.get("boot_end", 0.0),
                              end=marks.get("payload_end"),
                              speedup=speedup)
    print("trimmed " + str({k: v for k, v in info.items() if k != "kept_starts"}))
    for stale in (full.with_suffix(".gif"), full.with_suffix(".mp4"), full.with_suffix(".png")):
        stale.unlink(missing_ok=True)
    # Keep what the renderer produced, before anything optimises it, so the shipped file
    # can be compared against it rather than trusted.
    reference = OUT / f"{name}.reference.gif"
    shutil.copy(gif, reference)
    _optimize(gif)
    _derive(gif)

    forbid = ("Traceback", "not ready yet", "Error 1213", "No space left on device",
              *getattr(mod, "FORBID_STRINGS", ()))
    rows = gates.cast_gate(cast, must=tuple(getattr(mod, "MUST_STRINGS", ())), forbid=forbid,
                           window=(marks.get("boot_end", 0.0), marks.get("payload_end")),
                           tail_forbid=getattr(mod, "TAIL_FORBID", ()))
    # Map driver-motion spans onto frame indices using each frame's ORIGINAL time. Hold
    # clamping shortens frames, so a position in the finished gif no longer says when that
    # frame happened; matching on output timing put every span in the wrong place and
    # scored zero frames.
    shift = marks.get("boot_end", 0.0)
    spans = [(lo - shift, hi - shift) for lo, hi in timings.get("motion_spans", [])]
    starts = [t - info["kept_starts"][0] for t in info["kept_starts"]]
    motion_idx = {i for i, t in enumerate(starts)
                  if any(lo <= t <= hi for lo, hi in spans)}
    rows += gates.render_gate(gif, motion_idx=motion_idx or None)
    rows.append(gates.artifact_gate(gif, reference))
    reference.unlink(missing_ok=True)
    text, ok = gates.scorecard(name, rows)
    (OUT / f"{name}.score.txt").write_text(text + "\n")
    print(text)
    return text, ok


def _optimize(gif: pathlib.Path) -> None:
    """Shrink the gif with gifsicle, losslessly.

    ``--lossy`` is not an option here and never was. It decides some changed pixels are
    close enough to skip, so those pixels keep whatever the previous frame left in them,
    and text that changes between frames renders as two overlapping copies of itself --
    the footer hint in every reel came out looking struck through. It shipped once because
    the gates measured stroke weight and text colour, which lossy quantisation barely
    moves, and nothing compared the shipped frames against the ones agg drew. That
    comparison is now its own gate row.

    A full-screen scroll changes nearly every pixel, so inter-frame deltas save little and
    a scrolling reel lands large. If that ever exceeds the size cap the answer is fewer
    beats, not a dirtier picture.
    """
    import shutil
    import subprocess

    if not shutil.which("gifsicle"):
        return
    tmp = gif.with_suffix(".opt.gif")
    subprocess.run(["gifsicle", "-O3", str(gif), "-o", str(tmp)],
                   check=True, capture_output=True)
    if tmp.stat().st_size < gif.stat().st_size:
        tmp.replace(gif)
    else:
        tmp.unlink(missing_ok=True)


def _derive(gif: pathlib.Path) -> None:
    """mp4 and poster come from the shipped gif so they cannot disagree with it."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return
    mp4 = gif.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-i", str(gif), "-movflags", "faststart",
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-sseof", "-0.3", "-i", str(mp4), "-vframes", "1",
                    str(gif.with_suffix(".png"))], check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    _, ok = build(a.name, record=not a.no_record)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
