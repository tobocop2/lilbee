#!/usr/bin/env python3
"""Trim a rendered gif to its payload window, in the frame domain.

Supersedes trimming the cast. Cutting events out of a cast meant the first surviving
event painted a diff against frames that no longer existed, so the reel opened blank; the
fix for that was reconstructing the screen with a terminal emulator and re-emitting it,
which then had to reproduce every SGR attribute faithfully. It did not -- the restored
cursor landed on the INSERT chip, and the hand-rolled attribute reconstruction garbled
styled cells.

Rendering the whole cast and dropping frames avoids all of it. agg has already resolved
every attribute correctly, so the first kept frame is a real frame with nothing to
reconstruct. Frame durations carry the timing, so the window is exact.
"""
from __future__ import annotations

import argparse
import pathlib

from PIL import Image, ImageSequence


def trim_gif(gif: pathlib.Path, out: pathlib.Path, *, start: float = 0.0,
             end: float | None = None, freeze: float = 2.5,
             max_hold: float = 2.5) -> dict:
    """Cut to [start, end], clamp interior holds, and freeze the last frame.

    ``max_hold`` shortens any single frame that stays on screen longer than it. Those are
    not pacing: agg emits a frame per content change, so a five-second frame is five
    seconds of the app not painting -- a screen taking that long to mount, or a network
    fetch on the UI thread. Clamping shows the pause without making the viewer sit
    through it. The underlying stalls are worth filing against the app, not hiding, so
    the clamp count is returned.
    """
    im = Image.open(gif)
    frames, durs = [], []
    for f in ImageSequence.Iterator(im):
        frames.append(f.convert("RGB"))
        durs.append(f.info.get("duration", 40))

    clock, kept, kept_durs, kept_starts = 0.0, [], [], []
    for f, d in zip(frames, durs):
        t = clock / 1000.0
        clock += d
        if t < start:
            continue
        if end is not None and t > end:
            break
        kept.append(f)
        kept_durs.append(d)
        # Original time, before clamping shortens holds. Anything that needs to line a
        # frame up with something recorded during the take has to use this: clamping
        # moves every later frame earlier, so positions in the output no longer
        # correspond to when they happened.
        kept_starts.append(t)

    if not kept:
        raise SystemExit(f"window [{start}, {end}] kept no frames of {len(frames)}")

    cap = int(max_hold * 1000)
    clamped = sum(1 for d in kept_durs if d > cap)
    kept_durs = [min(d, cap) for d in kept_durs]

    # One hold, at the end, from one place. The pipeline once added a tail in two spots
    # and the freezes stacked into five seconds of dead air.
    kept_durs[-1] = int(freeze * 1000)

    kept[0].save(out, save_all=True, append_images=kept[1:],
                 duration=kept_durs, loop=0, optimize=True, disposal=1)
    return {"frames_in": len(frames), "frames_out": len(kept), "holds_clamped": clamped,
            "kept_starts": kept_starts, "duration": sum(kept_durs) / 1000.0,
            "dropped_head": sum(1 for i, _ in enumerate(frames)
                                if sum(durs[:i]) / 1000.0 < start)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gif", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--freeze", type=float, default=2.5)
    a = ap.parse_args()
    print(trim_gif(a.gif, a.out, start=a.start, end=a.end, freeze=a.freeze))


if __name__ == "__main__":
    main()
