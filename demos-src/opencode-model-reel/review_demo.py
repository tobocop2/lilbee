"""Timeline review for a recorded demo (gif/webm/mp4).

A demo is NOT done until reviewed across its whole timeline, not just the final
frame. This extracts frames at evenly spaced timestamps so a reviewer (human or a
vision model) can confirm the arc: the prompt is on screen, the answer renders, and
the answer quality is good. It also runs a cheap heuristic to flag the failure mode
that slipped through before: a recording dominated by a near-static "dead screen"
(cold-start blinking cursor / generation spinner) with content only in the last frame.

Usage:
    python review_demo.py <demo-file> [--frames N] [--out DIR]

Exit code 0 = no heuristic red flag; 1 = likely dead-screen-dominated (review the
extracted frames). The heuristic never replaces looking at the frames; it only stops
an obviously-broken demo from being called done without a look.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _duration_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        # GIFs sometimes report no stream duration; fall back to format duration.
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        return float(out) if out and out != "N/A" else 0.0


def _extract_frame(path: Path, t: float, dest: Path) -> int:
    """Write the frame at *t* seconds to *dest*; return its byte size (0 on failure)."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{t:.2f}",
         "-i", str(path), "-frames:v", "1", str(dest)],
        check=False,
    )
    return dest.stat().st_size if dest.exists() else 0


def _motion_ratio(path: Path) -> float:
    """Fraction of frames that are NOT near-duplicates of the previous one.

    Uses ffmpeg's ``mpdecimate`` (the same near-duplicate detector used to drop
    static frames). A streaming demo where the screen keeps changing keeps most of
    its frames; a cold-start/spinner demo that sits on one screen for seconds is
    mostly duplicates, so the ratio collapses. This is a real motion signal, unlike
    raw byte size which the static prompt panel keeps high.
    """
    total = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(path), "-vf", "mpdecimate", "-an", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    # The null muxer's final "frame=" line counts frames that survived mpdecimate.
    kept = 0
    for line in proc.stderr.splitlines():
        if "frame=" in line:
            try:
                kept = int(line.split("frame=")[1].split()[0])
            except (IndexError, ValueError):
                pass
    try:
        n = int(total)
    except ValueError:
        return 1.0
    return kept / n if n else 1.0


def review(path: Path, frames: int, out: Path) -> bool:
    """Extract *frames* evenly-spaced frames for visual review; flag dead screens.

    The frames are the point: a reviewer (human or vision model) must look at them
    and confirm the prompt is on screen, the answer renders, and the answer quality
    is good. The motion heuristic is only a tripwire for the cold-start/spinner
    shape that pays off solely in the final frame; it never replaces looking.
    """
    out.mkdir(parents=True, exist_ok=True)
    dur = _duration_seconds(path)
    if dur <= 0:
        print(f"  could not read duration for {path.name}", file=sys.stderr)
        return False
    stamps = [dur * (i + 0.5) / frames for i in range(frames)]
    print(f"{path.name}  ({dur:.1f}s, {frames} frames)")
    for i, t in enumerate(stamps):
        dest = out / f"{path.stem}__{i:02d}_{t:05.1f}s.png"
        size = _extract_frame(path, t, dest)
        print(f"  {t:5.1f}s  {size/1024:6.1f} KB  {dest}")
    motion = _motion_ratio(path)
    # Below ~35% unique frames means the screen sat static for most of the runtime
    # (cold-start dead screen / generation spinner) -- the broken shape.
    ok = motion >= 0.35
    verdict = "LIVE" if ok else "DEAD-SCREEN RISK"
    print(f"  -> {verdict}: motion={motion:.0%} unique frames. Inspect the frames "
          f"above to confirm prompt + answer + quality (heuristic is only a tripwire).")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", type=Path)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("/tmp/demo_review"))
    args = ap.parse_args()
    ok = review(args.demo, args.frames, args.out / args.demo.stem)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
