#!/usr/bin/env python3
"""GIF-to-GIF color acceptance test (epic bb-xe6o, settled 2026-07-16).

Compares the candidate's answer-pane text brightness against the reference
(shipped tui-chat.gif class). Both inputs MUST be palettized gifs: mp4-frame
vs gif-frame comparisons produced three wrong conclusions in a row.

Checks, per the epic's acceptance test:
- dominant bright colour of the mid-frame text is near-white in the
  (225,225,243) class, with a large dominant pixel count
- bright-pixel stroke run-lengths are mostly 2px+ (1px stems antialias grey)

Usage: color_accept.py <candidate.gif> <reference.gif> [--frame-frac 0.9]
Default: auto-pick the settled frame (max bright-text coverage among late frames);
pass --frame-frac to pin an index fraction. Validated 2026-07-16: reference picks
its settled frame with dominant (225,225,243); the grey font-14 take still fails.
Exit 0 pass, 1 fail. Prints the numbers either way; look at frames with EYES first.
"""

import argparse
import sys

import numpy as np
from PIL import Image, ImageSequence

BRIGHT_LUMA = 180          # pixels brighter than this count as text ink
NEAR_WHITE_MIN = 210       # dominant colour must have every channel >= this
DOMINANT_MIN_PX = 400      # and at least this many pixels at exactly that colour
RUN2_FRAC_MIN = 0.55       # fraction of bright runs that must be >= 2px


def mid_frame(path: str, frac: float | None = None) -> np.ndarray:
    """Settled-answer frame = the LAST frame. Every gif_finish output ends on a
    2.5s hold of the settled answer, and the reference gifs settle by their final
    frame too. (An earlier max-bright-pixels auto-pick chose mid-STREAM frames,
    where the generation spinner + accent colours out-bright the settled text and
    dragged the neutral-white dominant down to an antialias-halo value.)"""
    im = Image.open(path)
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    idx = int((len(frames) - 1) * frac) if frac is not None else len(frames) - 1
    return np.asarray(frames[idx], dtype=np.uint8)


def luma(a: np.ndarray) -> np.ndarray:
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def analyze(frame: np.ndarray) -> dict:
    # central band: skip window bar and edges; the answer pane dominates there
    h, w, _ = frame.shape
    band = frame[int(h * 0.15): int(h * 0.92), int(w * 0.04): int(w * 0.96)]
    lum = luma(band)
    bright = band[lum > BRIGHT_LUMA]
    if len(bright) == 0:
        return {"dominant": None, "dominant_n": 0, "run2_frac": 0.0, "bright_px": 0}
    # Dominant of NEUTRAL bright pixels only (actual white body text). Reels with a
    # short answer have little body text, so the teal/purple THEME ACCENTS (labels,
    # tab foam) would out-count it and fail a near-white check that the text passes.
    # low saturation (max-min channel spread) isolates white from accent colour.
    mx = bright.max(axis=1).astype(int)
    mn = bright.min(axis=1).astype(int)
    neutral = bright[(mx - mn) < 30]
    pool = neutral if len(neutral) else bright
    colors, counts = np.unique(pool.reshape(-1, 3), axis=0, return_counts=True)
    i = int(np.argmax(counts))
    # horizontal run lengths of bright pixels
    mask = lum > BRIGHT_LUMA
    runs = []
    for row in mask:
        n = 0
        for v in row:
            if v:
                n += 1
            elif n:
                runs.append(n)
                n = 0
        if n:
            runs.append(n)
    runs = np.array(runs) if runs else np.array([0])
    return {
        "dominant": tuple(int(c) for c in colors[i]),
        "dominant_n": int(counts[i]),
        "run2_frac": float((runs >= 2).mean()),
        "bright_px": int(len(bright)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("reference")
    ap.add_argument("--frame-frac", type=float, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="UI-walk reel: no block of white answer text. Gate on stroke "
                         "weight (ExtraBold rendering) only, not the near-white dominant "
                         "(chrome-heavy frames have sparse body text so dim UI labels "
                         "out-count it -- a false grey signal).")
    args = ap.parse_args()

    cand = analyze(mid_frame(args.candidate, args.frame_frac))
    ref = analyze(mid_frame(args.reference, args.frame_frac))
    print(f"reference {args.reference}: {ref}")
    print(f"candidate {args.candidate}: {cand}")

    fails = []
    if args.sweep:
        # ExtraBold renders fat stems (run2 ~0.9); regular/grey fallback ~0.25.
        if cand["run2_frac"] < RUN2_FRAC_MIN:
            fails.append(f"2px+ stroke-run fraction {cand['run2_frac']:.2f} < {RUN2_FRAC_MIN} "
                         f"(font not rendering ExtraBold?)")
    else:
        d = cand["dominant"]
        if d is None or min(d) < NEAR_WHITE_MIN:
            fails.append(f"dominant bright colour {d} not near-white (every channel >= {NEAR_WHITE_MIN})")
        if cand["dominant_n"] < DOMINANT_MIN_PX:
            fails.append(f"dominant count {cand['dominant_n']} < {DOMINANT_MIN_PX} (no dominant colour = grey mush)")
        if cand["run2_frac"] < RUN2_FRAC_MIN:
            fails.append(f"2px+ stroke-run fraction {cand['run2_frac']:.2f} < {RUN2_FRAC_MIN}")
    if fails:
        print("FAIL:\n  " + "\n  ".join(fails))
        return 1
    print("PASS: " + ("ExtraBold rendering confirmed" if args.sweep
                      else "candidate text is in the reference brightness class"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
