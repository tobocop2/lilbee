#!/usr/bin/env python3
"""Clamp mid-reel dead air: any static span longer than --max is cut to --max.

Same contract as trim_tail.py but for interior gaps: tape windows are measured
ceilings, so a fast generation leaves the settled frame parked until the next
beat fires. Motion spans are never touched -- every generation stays real-time
at 1x; only parked frames between beats are shortened, so pace stops depending
on how much a ceiling overshot.

Usage: clamp_gaps.py IN.mp4 OUT.mp4 [--max 1.2] [--threshold 0.002]
"""

import argparse
import pathlib
import subprocess
import sys


def frame_signature(path: pathlib.Path, fps: int = 5) -> list[tuple[float, bytes]]:
    """Decode to small grayscale frames: (timestamp, raw pixels)."""
    w, h = 160, 100
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    size = w * h
    return [(i / fps, out[i * size:(i + 1) * size]) for i in range(len(out) // size)]


def static_spans(frames, threshold: float, min_len: float):
    """(start, end) spans where consecutive frames differ by < threshold."""
    spans, start = [], None
    for (t0, a), (t1, b) in zip(frames, frames[1:]):
        diff = sum(x != y for x, y in zip(a, b)) / len(a)
        if diff < threshold:
            start = t0 if start is None else start
        else:
            if start is not None and t0 - start >= min_len:
                spans.append((start, t0))
            start = None
    if start is not None and frames[-1][0] - start >= min_len:
        spans.append((start, frames[-1][0]))
    return spans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--max", type=float, default=1.2, help="seconds a static span may keep")
    ap.add_argument("--threshold", type=float, default=0.002)
    args = ap.parse_args()
    src = pathlib.Path(args.src)
    frames = frame_signature(src)
    spans = static_spans(frames, args.threshold, min_len=args.max + 0.8)
    if not spans:
        print("no clampable gaps; copying through")
        subprocess.run(["cp", str(src), args.dst], check=True)
        return 0
    # Keep segments between the clamped gaps; each gap contributes its first
    # --max seconds (the settle is visible, the parking is not).
    keeps, cursor = [], 0.0
    for s, e in spans:
        keeps.append((cursor, s + args.max))
        cursor = e
    keeps.append((cursor, frames[-1][0] + 0.4))
    parts = "".join(
        f"[0:v]trim=start={a:.2f}:end={b:.2f},setpts=PTS-STARTPTS[v{i}];"
        for i, (a, b) in enumerate(keeps))
    concat = "".join(f"[v{i}]" for i in range(len(keeps))) + f"concat=n={len(keeps)}:v=1:a=0[out]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                    "-filter_complex", parts + concat, "-map", "[out]",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                    args.dst], check=True)
    cut = sum(e - s - args.max for s, e in spans)
    print(f"{src.name}: clamped {len(spans)} gap(s), removed {cut:.1f}s of parked frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
