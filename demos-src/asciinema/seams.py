#!/usr/bin/env python3
"""Measure cell seams: background hairlines inside a run of tiling glyphs.

A bar, a border or a sparkline is a run of glyphs in adjacent cells. If the cell the
renderer lays out is wider than the ink it puts in it, or if two glyph edges each cover
part of the same pixel and are composited separately, the run comes apart into segments.
It reads as tearing and it appeared in every reel.

Detection is per-row: find long runs of one saturated colour, then count pixels inside the
run that are markedly darker than the run's own colour. A clean bar has none. The measure
is deliberately about interior pixels only -- the ends of a bar are supposed to be edges.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from PIL import Image, ImageSequence

# Matched to what the artifact actually measures: the bar is (183, 91, 121) and its seam
# pixels are (162, 83, 111), about 15% down, and on an antialiased border the seam is a
# few percent UP instead. An earlier threshold of "30% darker" found none of it.
DIM_BAND = (0.62, 0.985)
HOT_BAND = (1.015, 1.60)
SAME_TOLERANCE = 14
MIN_BRIGHT = 45


def _row_seams(row: np.ndarray) -> tuple[int, int]:
    """Return (seam pixels, bright pixels) for one row of RGB values.

    A seam is a single dark column between two bright neighbours of the same colour.
    Written as that exact three-pixel signature rather than as a run-walk: a run-walk over
    a row of text counts the gaps between letters and reports 96% of everything as a seam,
    which is what the first version of this did.
    """
    left, centre, right = row[:-2].astype(int), row[1:-1].astype(int), row[2:].astype(int)
    lum_l, lum_c, lum_r = left.mean(axis=1), centre.mean(axis=1), right.mean(axis=1)
    base = np.minimum(lum_l, lum_r)
    same = np.abs(left - right).max(axis=1) <= SAME_TOLERANCE
    lit = base > MIN_BRIGHT
    ratio = np.divide(lum_c, base, out=np.ones_like(lum_c), where=base > 0)
    seam = ((ratio > DIM_BAND[0]) & (ratio < DIM_BAND[1])) | \
           ((ratio > HOT_BAND[0]) & (ratio < HOT_BAND[1]))
    return int((same & lit & seam).sum()), int((row.mean(axis=1) > MIN_BRIGHT).sum())


def periodicity(gif: pathlib.Path, cols: int = 128, samples: int = 8) -> dict:
    """How strongly seam pixels line up with the terminal's cell pitch.

    This is the measurement that distinguishes the artifact from the detector's own floor.
    agg's seams occur at cell boundaries, so their x positions share one phase modulo the
    cell width; ordinary antialiasing inside glyphs has no such phase. Absolute seam
    counts cannot tell those apart -- a settings screen full of bordered inputs floors at
    5% with nothing visibly wrong, and the VHS assets these reels replace measure 6-12% --
    and neither can the reduction ratio, because a reel that starts near the floor has
    little left to close.

    Returns the magnitude of the first circular moment of the phase distribution: near 1
    when every seam sits at the same point in the cell, near 0 when they are scattered.
    """
    frames = [np.asarray(f.convert("RGB"))
              for f in ImageSequence.Iterator(Image.open(gif))]
    step = max(1, len(frames) // samples)
    cell = frames[0].shape[1] / cols
    phases = []
    for f in frames[::step]:
        for r in range(0, f.shape[0], 3):
            row = f[r]
            left, centre, right = row[:-2].astype(int), row[1:-1].astype(int), row[2:].astype(int)
            lum_l, lum_c, lum_r = left.mean(axis=1), centre.mean(axis=1), right.mean(axis=1)
            base = np.minimum(lum_l, lum_r)
            same = np.abs(left - right).max(axis=1) <= SAME_TOLERANCE
            lit = base > MIN_BRIGHT
            ratio = np.divide(lum_c, base, out=np.ones_like(lum_c), where=base > 0)
            seam = ((ratio > DIM_BAND[0]) & (ratio < DIM_BAND[1])) | \
                   ((ratio > HOT_BAND[0]) & (ratio < HOT_BAND[1]))
            xs = np.where(same & lit & seam)[0] + 1
            if len(xs):
                phases.append((xs % cell) / cell)
    if not phases:
        return {"n": 0, "alignment": 0.0}
    p = np.concatenate(phases)
    r = float(np.abs(np.exp(2j * np.pi * p).mean()))
    return {"n": int(len(p)), "alignment": r}


def measure(gif: pathlib.Path, samples: int = 8) -> dict:
    frames = [np.asarray(f.convert("RGB"))
              for f in ImageSequence.Iterator(Image.open(gif))]
    step = max(1, len(frames) // samples)
    seams = runlen = 0
    for f in frames[::step]:
        # Both axes. Bars and borders seam between columns; a horizontal rule seams
        # between rows, and measuring only one axis hides half the artifact.
        for r in range(0, f.shape[0], 3):
            s, n = _row_seams(f[r])
            seams += s
            runlen += n
        t_ = np.swapaxes(f, 0, 1)
        for c in range(0, t_.shape[0], 3):
            s, n = _row_seams(t_[c])
            seams += s
            runlen += n
    return {"seam_px": seams, "run_px": runlen,
            "rate": (seams / runlen) if runlen else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gif", type=pathlib.Path, nargs="+")
    a = ap.parse_args()
    for g in a.gif:
        m = measure(g)
        print(f"{g.name:28} seams {m['seam_px']:6d} in {m['run_px']:7d} run px "
              f"({m['rate']:.4%})")


if __name__ == "__main__":
    main()
