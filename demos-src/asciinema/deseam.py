#!/usr/bin/env python3
"""Close the one-pixel seams agg leaves at cell boundaries.

What the artifact is: a run of U+2588 that should be a solid bar renders with a hairline
every ~11px, one per terminal cell. The bar colour is (183, 91, 121) and the seam pixels
are (162, 83, 111) and (155, 79, 107) -- about 15% darker, one pixel wide, at the cell
pitch. Panel borders and sparklines show the same thing. It reads as tearing.

Why it happens: agg lays cells out at a width of 39/64 of the font size -- 10.969px at
size 18 -- and that is independent of the font. Advances from 592 to 614 units all render
to the same 1404x907, so the font is not the lever; glyph overhang is not either, because
each glyph is clipped to its cell. Cell edges therefore land inside pixels, each side is
composited separately, and the shared pixel ends up partially covered by both. The only
font size where 39/64 lands on a whole pixel is 64, which would mean rendering every reel
at 4992px wide and downsampling.

What this does: a pixel is restored to its neighbours' colour when it sits between two
pixels of the same colour and is slightly -- not much -- darker than them. That is the
signature of partial coverage and nothing else: inside a glyph the two sides of a gap are
not the same colour, and a real one-pixel dark line drawn by the app would be far darker
than 15%. Both axes, since the same thing happens between rows.

This repairs the renderer, not the recording. The terminal really did draw a solid bar.
"""
from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image, ImageSequence

# A seam is dimmer than its neighbours, but only slightly: full coverage on one side is
# still most of the pixel. Anything below the floor is something the app actually drew.
MAX_RATIO = 0.985
MIN_RATIO = 0.62
# The mirror case. Where two glyph edges overlap inside one pixel their coverage adds up,
# so a seam on an antialiased line is slightly BRIGHTER than the line rather than darker:
# the drawer's border reads 62, 62, 65, 68, 62 across a cell boundary. Same repair, other
# direction, with a ceiling so a genuinely bright pixel between two dim ones survives.
MIN_RATIO_BRIGHT = 1.015
MAX_RATIO_BRIGHT = 1.60
SAME_TOLERANCE = 14
MIN_BRIGHT = 45


def _pass(a: np.ndarray, axis: int) -> int:
    """Repair seams along one axis. Returns how many pixels were changed."""
    a = np.swapaxes(a, 0, axis)
    left, centre, right = a[:-2], a[1:-1], a[2:]
    lum_l, lum_c, lum_r = (x.mean(axis=-1) for x in (left, centre, right))
    same = np.abs(left.astype(np.int16) - right.astype(np.int16)).max(axis=-1) <= SAME_TOLERANCE
    lit = np.minimum(lum_l, lum_r) > MIN_BRIGHT
    ratio = np.divide(lum_c, np.minimum(lum_l, lum_r),
                      out=np.ones_like(lum_c), where=np.minimum(lum_l, lum_r) > 0)
    dim = (ratio < MAX_RATIO) & (ratio > MIN_RATIO)
    hot = (ratio > MIN_RATIO_BRIGHT) & (ratio < MAX_RATIO_BRIGHT)
    mask = same & lit & (dim | hot)
    centre[mask] = left[mask]
    return int(mask.sum())


# A seam two pixels wide needs more than one three-pixel window to close: each pass
# erodes it from the outside in. Measured on a real frame, one horizontal pass takes the
# seam rate from 9.5% to 3.5% and three take it to 1.6%; a fourth changes nothing.
PASSES = 4


def repair(frame: np.ndarray) -> tuple[np.ndarray, int]:
    out = frame.copy()
    fixed = 0
    for axis in (1, 0):            # columns first, then rows
        for _ in range(PASSES):
            n = _pass(out, axis)
            fixed += n
            if not n:
                break
    return out, fixed


def repair_gif(path: pathlib.Path) -> dict:
    """Repair every frame of a gif in place, preserving frame durations.

    Reports the seam rate before and after. The rate is the check that matters, and it is
    only meaningful as a ratio: the detector counts any dimmer-or-brighter pixel between
    two matching neighbours, so a screen full of bordered input boxes has a much higher
    floor than a chat pane, and an absolute limit calibrated on one fails the other while
    both look correct.
    """
    import seams
    im = Image.open(path)
    frames, durs = [], []
    for f in ImageSequence.Iterator(im):
        frames.append(np.asarray(f.convert("RGB")))
        durs.append(f.info.get("duration", 40))
    fixed, total = [], 0
    for a in frames:
        out, n = repair(a)
        total += n
        fixed.append(Image.fromarray(out))
    before = seams.measure(path)["rate"]
    fixed[0].save(path, save_all=True, append_images=fixed[1:],
                  duration=durs, loop=0, optimize=True, disposal=1)
    after = seams.measure(path)["rate"]
    return {"frames": len(fixed), "pixels_repaired": total,
            "seam_rate_before": round(before, 5), "seam_rate_after": round(after, 5)}
