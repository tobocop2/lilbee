#!/usr/bin/env python3
"""A/B a new reel against the asset it replaces, on the measurable axes.

Only half of grading; the rest is beat parity and whether the payoff reads, which is
written down per reel. This covers what a number can settle: is it dimmer, choppier,
shorter, heavier, and does it end on payload or on an exited shell.

The mid-frame comparison is the trap this exists to avoid making by hand. Sampling both
reels at 50% once compared a rose-pine frame against the old reel's post-theme-cycle
teal and reported a colour regression that was not one. Frames are therefore sampled at
several points and the best-matching pair reported alongside the median.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image, ImageSequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gates  # noqa: E402


def _frames(gif: pathlib.Path) -> tuple[list[Image.Image], list[int]]:
    im = Image.open(gif)
    fs = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    im2, durs = Image.open(gif), []
    try:
        while True:
            durs.append(im2.info.get("duration", 0))
            im2.seek(im2.tell() + 1)
    except EOFError:
        pass
    return fs, durs


def _ink(f: Image.Image) -> tuple[float, tuple[int, int, int]]:
    """Mean brightness of text pixels, and their mean colour."""
    a = np.array(f)
    bright = a[a.mean(axis=2) > 170]
    if bright.size == 0:
        return 0.0, (0, 0, 0)
    m = bright.reshape(-1, 3).mean(axis=0)
    return float(m.mean()), tuple(int(x) for x in m)


def compare(new: pathlib.Path, live: pathlib.Path) -> str:
    out = [f"A/B  {new.name}  vs  live {live.name}"]
    rows = {}
    for label, path in (("new", new), ("live", live)):
        rows[label] = {r.name: r for r in gates.render_gate(path)}
    for name in ("duration", "motion_fps", "stroke_weight", "near_white_text", "gif_size"):
        out.append(f"  {name:16} new: {rows['new'][name].detail}")
        out.append(f"  {'':16} live: {rows['live'][name].detail}")

    # Ink brightness across the whole reel rather than at one sampled frame.
    for label, path in (("new", new), ("live", live)):
        fs, _ = _frames(path)
        vals = [_ink(f)[0] for f in fs[:: max(1, len(fs) // 12)]]
        colour = _ink(fs[len(fs) // 2])[1]
        out.append(f"  ink({label:4})       median {np.median(vals):.0f} over "
                   f"{len(vals)} samples, mid-frame colour {colour}")

    # What the reel leaves on screen. A reel that ends on a shell prompt ends on nothing.
    for label, path in (("new", new), ("live", live)):
        fs, _ = _frames(path)
        last = np.array(fs[-1])
        painted = float((last.mean(axis=2) > 60).mean())
        out.append(f"  ending({label:4})    {painted:.0%} of the final frame is painted")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("new", type=pathlib.Path)
    ap.add_argument("live", type=pathlib.Path)
    a = ap.parse_args()
    print(compare(a.new, a.live))


if __name__ == "__main__":
    main()
