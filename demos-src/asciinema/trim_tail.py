#!/usr/bin/env python3
"""Trim a reel's dead tail down to a short freeze.

Tapes sleep long enough that a slow generation is never cut off, which leaves
the finished answer sitting motionless for however long the model happened to
beat that budget. Find where motion actually stops and keep a fixed freeze
after it, so pace comes out the same no matter how fast the model ran.

Usage: trim_tail.py IN.mp4 OUT.mp4 [--freeze 2.5] [--threshold 0.002]
"""

import argparse
import pathlib
import subprocess
import sys


def frame_signature(path: pathlib.Path, fps: int = 5, ignore_bottom: float = 0.2):
    """Decode to grayscale frames as numpy arrays: (timestamp, HxW uint8).

    Resolution 480x300 (not 160x100): at the smaller size the down-scaled answer
    text blurs so much that a newly-streamed token changes pixels by <30 and the
    dither-immune threshold skips it -- so tail-trim cut mid-answer. 480x300 keeps
    glyph edges crisp enough that streaming registers.

    The bottom `ignore_bottom` fraction is cropped BEFORE motion detection: that
    strip holds the chat input box whose cursor blinks forever, which otherwise
    reads as perpetual motion (the answer content always settles in the pane above
    it). The footer goes with it."""
    w, h = 480, 300
    keep = 1.0 - ignore_bottom
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"fps={fps},crop=iw:ih*{keep}:0:0,scale={w}:{h},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )
    import numpy as np
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(buf) // (w * h)
    frames = buf[:n * w * h].reshape(n, h, w)
    return [(i / fps, frames[i]) for i in range(n)]


def last_motion(frames, threshold: float) -> float:
    """Timestamp of the last frame that differs from its predecessor. A pixel
    counts as changed only past |Δ|>30 (dither-immune); the fraction of changed
    pixels must exceed `threshold` (streaming a token clears this at 480x300)."""
    import numpy as np
    last = 0.0
    for (_, prev), (t, cur) in zip(frames, frames[1:]):
        diff = (np.abs(cur.astype(np.int16) - prev.astype(np.int16)) > 30).mean()
        if diff > threshold:
            last = t
    return last


def duration(path: pathlib.Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("dst", type=pathlib.Path)
    ap.add_argument("--freeze", type=float, default=2.5)
    ap.add_argument("--threshold", type=float, default=0.0004)
    ap.add_argument("--ignore-bottom", type=float, default=0.2)
    args = ap.parse_args()

    frames = frame_signature(args.src, ignore_bottom=args.ignore_bottom)
    if not frames:
        print("no frames decoded", file=sys.stderr)
        return 1
    settle = last_motion(frames, args.threshold)
    total = duration(args.src)
    end = min(settle + args.freeze, total)
    print(f"{args.src.name}: {total:.2f}s, settles at {settle:.2f}s -> keeping {end:.2f}s")
    if end >= total - 0.1:
        print("nothing to trim")
        return 0
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(args.src), "-t", f"{end:.3f}",
         "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
         str(args.dst)],
        check=True,
    )
    print(f"wrote {args.dst} ({duration(args.dst):.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
