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

from PIL import Image, ImageDraw, ImageFont, ImageSequence

KIT = pathlib.Path(__file__).resolve().parent
SUBTITLE_FONT = KIT / "fonts-extrabold/LilbeeReelMono-ExtraBold.ttf"
# Below this a generation window is not worth compressing; the label would cost more
# attention than the seconds it saves.
MIN_SPEEDUP_SECONDS = 10.0


def _label(frame: Image.Image, text: str) -> Image.Image:
    """Burn a speed label into a frame.

    Compressed time has to say so on the frame itself. A reel that quietly drops four out
    of every five frames while a model generates is claiming the model is five times
    faster than it is, and nothing in the file would contradict that.
    """
    out = frame.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.truetype(str(SUBTITLE_FONT), 22)
    pad, w, h = 14, out.size[0], out.size[1]
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    x, y = w - tw - pad * 3, h - th - pad * 3
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad * 2], fill=(38, 35, 58))
    draw.text((x, y), text, font=font, fill=(246, 193, 119))
    return out


def trim_gif(gif: pathlib.Path, out: pathlib.Path, *, start: float = 0.0,
             end: float | None = None, freeze: float = 2.5,
             max_hold: float = 2.5,
             speedup: list[tuple[float, float, int]] | None = None) -> dict:
    """Cut to [start, end], clamp interior holds, and freeze the last frame.

    ``max_hold`` shortens any single frame that stays on screen longer than it. Those are
    not pacing: agg emits a frame per content change, so a five-second frame is five
    seconds of the app not painting -- a screen taking that long to mount, or a network
    fetch on the UI thread. Clamping shows the pause without making the viewer sit
    through it. The underlying stalls are worth filing against the app, not hiding, so
    the clamp count is returned.

    ``speedup`` is a list of ``(lo, hi, factor)`` in the same original-time seconds: inside that
    window one frame in ``factor`` is kept and the rest are dropped, so the span plays
    that much faster and the file gets that much smaller. It exists for generation, where
    most of a reel can be a progress bar on a laptop. Kept frames in the window carry a
    label saying so. The window is ignored if it is shorter than MIN_SPEEDUP_SECONDS,
    which means a fast answer produces no label rather than a pointless one.
    """
    im = Image.open(gif)
    frames, durs = [], []
    for f in ImageSequence.Iterator(im):
        frames.append(f.convert("RGB"))
        durs.append(f.info.get("duration", 40))

    windows = [w for w in (speedup or []) if w[1] - w[0] >= MIN_SPEEDUP_SECONDS]
    counters = {i: 0 for i in range(len(windows))}
    sped = 0

    clock, kept, kept_durs, kept_starts = 0.0, [], [], []
    for f, d in zip(frames, durs):
        t = clock / 1000.0
        clock += d
        if t < start:
            continue
        if end is not None and t > end:
            break
        hit = next((i for i, (lo, hi, _) in enumerate(windows) if lo <= t <= hi), None)
        if hit is not None:
            factor = windows[hit][2]
            counters[hit] += 1
            if (counters[hit] - 1) % factor:
                continue
            f = _label(f, f"{factor}x speed")
            sped += 1
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
            "sped_frames": sped, "speed_windows": len(windows),
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
