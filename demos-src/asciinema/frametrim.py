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
import math
import pathlib

import numpy as np
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


# A stretch where the screen only twitches -- spinner, progress bar, percentage -- is a
# wait however much it animates. Matches gates.WAIT_CHANGE / gates.MAX_WAIT_S; the gate
# measures what this failed to remove.
WAIT_CHANGE = 0.012
# Deliberately below gates.MAX_WAIT_S. The compressor sees the gif before deseaming and
# optimisation, both of which shift pixels enough to merge two runs the compressor scored
# separately -- a 6.3s wait reached the gate having been measured as under the limit here.
# Compressing from 4.5s leaves margin for that rather than racing the threshold.
WAIT_MIN_SECONDS = 4.5
# What a compressed wait should play for. The factor is derived per window so a 12s wait
# and a 40s one both come out about this long, rather than a fixed factor leaving the
# long ones still too slow.
WAIT_TARGET_SECONDS = 4.0
# Cadence compressed frames are re-timed to. Must clear the HIGHEST floor any row holds
# a frame to, not just the pacing one: a compressed window can overlap a driver-motion
# span -- typing through a section slow enough to be compressed -- and 80ms put five
# reels at exactly 12.5fps against motion_fps's floor of 15. 60ms is 16.7fps, over both.
WINDOW_CADENCE_MS = 60
# A section that keeps repainting but sits on each frame this long reads as choppy
# whether or not the slowness is honest. The remedy is the same as for a wait.
# Mirrors gates.PACED_CHANGE / MIN_PACED_FPS / MIN_PACED_SECTION_S / MIN_PACED_FRAMES.
# The compressor removes what the gate would fail, so the two move together.
PACED_CHANGE = 0.0005
SLOW_MIN_FPS = 10.0
SLOW_MIN_SECONDS = 1.2
SLOW_MIN_FRAMES = 12


def detect_slow(frames: list[Image.Image], durs: list[int]) -> list[tuple[float, float]]:
    """Find stretches that are repainting properly but too slowly to watch.

    Distinct from detect_waits, which looks for a screen barely changing at all. Here the
    content really is being drawn -- a crawl filling in, tokens arriving -- just at a rate
    that reads as choppy. Both get compressed; nothing else can fix a frame rate the
    application, not the pipeline, decided.

    Segmented exactly the way gates.pacing_gate segments, on how much of the screen
    moved rather than on frame duration. Grouping by duration instead let a single quick
    frame split a slow section in two, so a 7.2s stretch at 2fps went undetected here and
    was then failed by the gate -- the compressor and the check have to agree on what a
    section is or they argue forever.
    """
    arrs = [np.asarray(f, dtype=np.int16) for f in frames]
    if len(arrs) < 3:
        return []
    px = arrs[0].shape[0] * arrs[0].shape[1]
    times, clock = [], 0.0
    for d in durs:
        times.append(clock / 1000.0)
        clock += d

    runs: list[tuple[float, float]] = []
    seg: list[tuple[int, int]] = []          # (frame index, duration)

    def flush(end_i: int) -> None:
        if len(seg) < SLOW_MIN_FRAMES:
            return
        span = times[end_i] - times[seg[0][0]]
        if span < SLOW_MIN_SECONDS:
            return
        med = float(np.median([d for _, d in seg]))
        if med and 1000.0 / med < SLOW_MIN_FPS:
            runs.append((times[seg[0][0]], times[end_i]))

    for i in range(1, len(arrs)):
        share = float((np.abs(arrs[i - 1] - arrs[i]).max(axis=2) > 24).sum()) / px
        if share >= PACED_CHANGE:
            seg.append((i, durs[i]))
            continue
        flush(i)
        seg = []
    flush(len(arrs) - 1)
    return runs


def detect_waits(frames: list[Image.Image], durs: list[int]) -> list[tuple[float, float]]:
    """Find stretches the viewer would sit through, in original-time seconds.

    Declared speed windows are opt-in, and a reel that forgot one shipped 53.6s of
    spinner inside 88.6s with every row green. Nothing checked that a reel with a long
    wait had declared a window, so the fix is to stop requiring the declaration: a wait
    is visible in the frames themselves and does not need announcing.
    """
    arrs = [np.asarray(f, dtype=np.int16) for f in frames]
    if len(arrs) < 3:
        return []
    px = arrs[0].shape[0] * arrs[0].shape[1]
    times, clock = [], 0.0
    for d in durs:
        times.append(clock / 1000.0)
        clock += d

    runs: list[tuple[float, float]] = []
    start_t, run_s = None, 0.0
    for i in range(1, len(arrs)):
        share = float((np.abs(arrs[i - 1] - arrs[i]).max(axis=2) > 24).sum()) / px
        if share < WAIT_CHANGE:
            if start_t is None:
                start_t = times[i - 1]
            run_s += durs[i] / 1000.0
            continue
        if start_t is not None and run_s >= WAIT_MIN_SECONDS:
            runs.append((start_t, times[i]))
        start_t, run_s = None, 0.0
    if start_t is not None and run_s >= WAIT_MIN_SECONDS:
        runs.append((start_t, times[-1]))
    return runs


def trim_gif(gif: pathlib.Path, out: pathlib.Path, *, start: float = 0.0,
             end: float | None = None, freeze: float = 2.5,
             max_hold: float = 2.5,
             auto_wait: bool = True,
             protect: list[tuple[float, float]] | None = None,
             motion: list[tuple[float, float]] | None = None,
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

    # Spans a reel declares as the demonstration itself. Nothing touches these: not the
    # timelapse, not the hold clamp. A launch reel exists so a viewer can see how long the
    # launch takes, and compressing it -- or shortening its pauses -- answers the question
    # the reel was recorded to ask. Two launch reels sit side by side in the README for
    # exactly that comparison.
    keep = list(protect or [])
    # Driver motion gets a weaker guarantee than a declared span, and deliberately so.
    # It is shielded from slow-section compression, because the driver's cadence is the
    # thing motion_fps measures and thinning it destroys the measurement. It is NOT
    # shielded from wait detection: a burst of keypresses against a log with nothing left
    # to scroll produces one-second frames that barely change, and 10.9s of that shipped
    # while being counted as protected motion. If the screen is not moving, it does not
    # matter that a key is being pressed -- the viewer sees a still either way.
    slow_only = list(motion or [])

    def guarded(lo: float, hi: float) -> bool:
        return any(not (hi <= klo or lo >= khi) for klo, khi in keep)

    spans = [(lo, hi) for lo, hi, _ in (speedup or [])
             if hi - lo >= MIN_SPEEDUP_SECONDS and not guarded(lo, hi)]
    auto = 0
    if auto_wait:
        # Detected stretches the reel did not declare. A declared window wins, but only
        # over the part it actually covers: clipping rather than discarding. Dropping a
        # whole span on any overlap threw away a genuine 2.6s slow stretch because it
        # began 0.1s before an adjacent wait window ended, and the gate then failed the
        # reel for exactly the section the compressor had decided to skip.
        detected = ([(lo, hi, False) for lo, hi in detect_waits(frames, durs)]
                    + [(lo, hi, True) for lo, hi in detect_slow(frames, durs)])
        for lo, hi, is_slow in detected:
            # Clip against protected spans, do not discard on overlap. Discarding here
            # was the third instance of this same mistake in this file: a wait that
            # merely touched a protected launch was dropped whole, compression fell to
            # zero, and a 9.7s wait shipped. Overlap means "trim the covered part", never
            # "abandon the whole stretch".
            shields = sorted(keep + (slow_only if is_slow else []))
            for klo, khi in shields:
                if hi <= klo or lo >= khi:
                    continue
                if lo < klo and hi > khi:
                    hi = klo
                elif lo >= klo:
                    lo = khi
                else:
                    hi = klo
            if hi - lo < SLOW_MIN_SECONDS:
                continue
            for wlo, whi in sorted(spans):
                if hi <= wlo or lo >= whi:
                    continue
                if lo < wlo and hi > whi:      # covered span sits inside; keep the head
                    hi = wlo
                elif lo >= wlo:
                    lo = whi
                else:
                    hi = wlo
            if hi - lo < SLOW_MIN_SECONDS:
                continue
            spans.append((lo, hi))
            auto += 1

    # Frame times, so a window can be resolved to the frames inside it before anything is
    # dropped. Choosing a stride needs to know how many frames there are.
    times, clock = [], 0.0
    for d in durs:
        times.append(clock / 1000.0)
        clock += d

    # Per window: a stride, and the honest speed factor to burn on the frames it keeps.
    # Kept frames are re-timed to a fixed cadence rather than carrying their original
    # durations. Dropping frames alone shortens a stretch without making it any smoother
    # -- a 15.7s section of 1.1s frames stays a slideshow at 0.9fps however many frames
    # come out of it -- and the reels this exists for are slow precisely because each
    # frame sat there, not because there were too many of them.
    plans: list[tuple[float, float, int, str]] = []
    for lo, hi in spans:
        idx = [i for i, t in enumerate(times) if lo <= t <= hi]
        if len(idx) < 2:
            continue
        want = max(2, int(WAIT_TARGET_SECONDS / (WINDOW_CADENCE_MS / 1000.0)))
        stride = max(1, math.ceil(len(idx) / want))
        out_s = math.ceil(len(idx) / stride) * WINDOW_CADENCE_MS / 1000.0
        factor = max(2, round((hi - lo) / out_s)) if out_s else 2
        plans.append((lo, hi, stride, f"{factor}x speed"))

    counters = {i: 0 for i in range(len(plans))}
    sped = 0

    clock, kept, kept_durs, kept_starts = 0.0, [], [], []
    for f, d in zip(frames, durs):
        t = clock / 1000.0
        clock += d
        if t < start:
            continue
        if end is not None and t > end:
            break
        hit = next((i for i, (lo, hi, _, _) in enumerate(plans) if lo <= t <= hi), None)
        if hit is not None:
            _, _, stride, label = plans[hit]
            counters[hit] += 1
            if (counters[hit] - 1) % stride:
                continue
            f = _label(f, label)
            d = WINDOW_CADENCE_MS
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
    in_keep = [any(klo <= t <= khi for klo, khi in keep) for t in kept_starts]
    clamped = sum(1 for d, g in zip(kept_durs, in_keep) if d > cap and not g)
    # A five-second pause inside a protected span is how long the app actually took.
    kept_durs = [d if g else min(d, cap) for d, g in zip(kept_durs, in_keep)]

    # One hold, at the end, from one place. The pipeline once added a tail in two spots
    # and the freezes stacked into five seconds of dead air.
    kept_durs[-1] = int(freeze * 1000)

    kept[0].save(out, save_all=True, append_images=kept[1:],
                 duration=kept_durs, loop=0, optimize=True, disposal=1)
    # Where the protected spans ended up in the finished file, so the pacing gate can be
    # told which stretches it must not fail the reel for.
    protected_out, clock_out = [], 0.0
    run_from = None
    for g, d in zip(in_keep, kept_durs):
        if g and run_from is None:
            run_from = clock_out / 1000.0
        elif not g and run_from is not None:
            protected_out.append((run_from, clock_out / 1000.0))
            run_from = None
        clock_out += d
    if run_from is not None:
        protected_out.append((run_from, clock_out / 1000.0))

    return {"frames_in": len(frames), "frames_out": len(kept), "holds_clamped": clamped,
            "protected": protected_out,
            "sped_frames": sped, "speed_windows": len(plans), "auto_waits": auto,
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


def compress_waits(gif: pathlib.Path, limit: float = 6.0,
                   protect: list[tuple[float, float]] | None = None) -> dict:
    """Second pass over a written gif: compress any wait that survived the first.

    The first pass measures the frames in memory; what ships has been quantised to a
    256-colour palette on save, and quantisation erases small differences the compressor
    counted as real content. So a stretch it scored as two short waits can reach the gate
    as one long one -- the same disagreement as the deseam ordering, one stage further
    down, and not fixable by another threshold guess because the pixels genuinely differ.

    Measuring what was actually written closes it. Only stretches still over the limit are
    touched, and those frames were never compressed on the first pass, so no frame ends up
    with two speed labels. Returns what it did; an empty windows list means the first pass
    was sufficient.
    """
    im = Image.open(gif)
    frames, durs = [], []
    for f in ImageSequence.Iterator(im):
        frames.append(f.convert("RGB"))
        durs.append(f.info.get("duration", 40))
    if len(frames) < 4:
        return {"windows": 0}

    # The deliberate hold on the last frame is a directorial beat, not a wait; excluded so
    # a reel is never compressed for ending on something readable.
    # Clip against protected spans rather than discarding on any overlap -- the same
    # mistake the first pass made and had fixed. A wait that begins inside a protected
    # launch and runs past its end was dropped whole here, so 6.3s of it survived to the
    # gate with the protected part correctly excused and the rest not compressed at all.
    guard = sorted(protect or [])
    spans = []
    for lo, hi in detect_waits(frames, durs[:-1] + [40]):
        for klo, khi in guard:
            if hi <= klo or lo >= khi:
                continue
            if lo < klo and hi > khi:
                hi = klo
            elif lo >= klo:
                lo = khi
            else:
                hi = klo
        if hi - lo >= limit - 1.5:
            spans.append((lo, hi))
    if not spans:
        return {"windows": 0}

    times, clock = [], 0.0
    for d in durs:
        times.append(clock / 1000.0)
        clock += d

    out_frames, out_durs = [], []
    for i, (f, d) in enumerate(zip(frames, durs)):
        hit = next((s for s in spans if s[0] <= times[i] <= s[1]), None)
        if hit is None:
            out_frames.append(f)
            out_durs.append(d)
            continue
        idx = [j for j, tt in enumerate(times) if hit[0] <= tt <= hit[1]]
        want = max(2, int(WAIT_TARGET_SECONDS / (WINDOW_CADENCE_MS / 1000.0)))
        stride = max(1, math.ceil(len(idx) / want))
        if idx.index(i) % stride:
            continue
        out_s = math.ceil(len(idx) / stride) * WINDOW_CADENCE_MS / 1000.0
        factor = max(2, round((hit[1] - hit[0]) / out_s)) if out_s else 2
        out_frames.append(_label(f, f"{factor}x speed"))
        out_durs.append(WINDOW_CADENCE_MS)

    out_durs[-1] = durs[-1]
    out_frames[0].save(gif, save_all=True, append_images=out_frames[1:],
                       duration=out_durs, loop=0, optimize=True, disposal=1)
    return {"windows": len(spans), "frames_out": len(out_frames),
            "duration": sum(out_durs) / 1000.0}
