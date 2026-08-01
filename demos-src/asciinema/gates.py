#!/usr/bin/env python3
"""Reel gates: run the checks, emit a scorecard, refuse to pass a row that cannot fail.

Two of the gates in the first draft of this program were structurally incapable of
failing, which is worse than having no gate because it launders a bad take as verified.
Both are fixed here and both are covered by ``selftest``:

  * "longest idle gap under threshold" can never go red on a Textual cast. The thinking
    header ticks at 0.1s and a focused input blinks at 0.5s, so the maximum gap is 0.1s
    by construction. Replaced with a gap measured between *content-bearing* changes,
    ignoring cursor-only and repaint-only events.

  * "the test fails on the stub" is satisfied by any check whatsoever, because the stub
    returns a constant. That one belongs to the Godot artifact runner; the equivalent
    here is that every threshold is asserted against a deliberately broken input before
    it is trusted.

Rows report PASS, FAIL, or UNTESTED. UNTESTED is not a pass: a scorecard with an
untested row means the reel is not done.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
import re

import numpy as np
from PIL import Image, ImageSequence

# Cursor moves, mode switches and colour resets carry no information a viewer can see.
# A stream of these is what makes a naive idle-gap check unable to fail.
_NOISE = re.compile(rb"^(?:\x1b\[[\d;]*[HfABCDGdm]|\x1b\[\?\d+[hl]|\x1b\[[0-2]?K|\s)*$")

MIN_UNIQUE_FRAME_RATIO = 0.80
# Hard floor. agg emits a frame per content change, so this measures how often the reel
# actually changes while it is moving. 12fps was shipped once and read as visibly choppy.
# 15 is the measured ceiling for a stream-driven renderer against this app, not a
# preference. agg emits one frame per distinct timestamp and Textual flushes a repaint
# roughly every 60ms under load, so ~16fps is the most the pipeline can produce; typing
# at 35ms, 20ms and 10ms all yield the same rate. Reaching the 25fps the VHS assets show
# requires fixed-rate PIXEL capture, which samples cursor blink and intermediate paint
# states the byte stream never carries (bb-83t3f). The floor exists to catch the 12.5fps
# regression that shipped once, not to encode an unreachable target.
MIN_MOTION_FPS = 15.0
# A frame that stays on screen longer than this is a dwell the reel asked for.
HOLD_MS = 300
# One typed question is the smallest driver span a reel legitimately has: about 30
# characters against a ~60ms repaint is a dozen frames. Asking for more than that
# marks honest reels untested rather than catching anything.
MIN_MOTION_FRAMES = 12
MAX_CONTENT_GAP = 4.0
MAX_STALL_SHARE = 0.20
MAX_GIF_MB = 10.0
MIN_STROKE_MEAN = 2.0
NEAR_WHITE = (225, 225, 243)
NEAR_WHITE_TOLERANCE = 45


@dataclasses.dataclass
class Row:
    name: str
    status: str          # PASS | FAIL | UNTESTED
    detail: str

    def __str__(self) -> str:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "UNTESTED": "----"}[self.status]
        return f"  [{mark}] {self.name}: {self.detail}"


def _events(cast: pathlib.Path) -> list[tuple[float, str]]:
    lines = cast.read_text(errors="ignore").splitlines()
    header = json.loads(lines[0])
    out, clock = [], 0.0
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e[1] != "o":
            continue
        # v3 stores deltas, v2 stores absolute time.
        t = (clock := clock + e[0]) if header.get("version") == 3 else e[0]
        out.append((t, e[2]))
    return out


def cast_gate(cast: pathlib.Path, *, must: tuple[str, ...] = (),
              forbid: tuple[str, ...] = ("Traceback", "not ready yet", "Error 1213"),
              window: tuple[float, float | None] | None = None) -> list[Row]:
    """Everything checkable from the byte stream, before a frame is ever rendered.

    ``window`` restricts every check to the span that actually ships. Without it the boot
    banner counts: a four-second wait for the model to load reads as dead air the viewer
    never sees, and a must-string that only ever appeared during startup passes.
    """
    events = _events(cast)
    if window:
        lo, hi = window
        events = [(t, d) for t, d in events if t >= lo and (hi is None or t <= hi)]
    text = "".join(e[1] for e in events)
    rows: list[Row] = []

    missing = [m for m in must if m not in text]
    rows.append(Row("must_strings", "FAIL" if missing else "PASS",
                    f"missing {missing}" if missing else f"all {len(must)} present"))

    hits = [f for f in forbid if f in text]
    rows.append(Row("no_error_text", "FAIL" if hits else "PASS",
                    f"found {hits}" if hits else "clean"))

    # Content-bearing gaps only. Measuring raw event gaps is the check that cannot fail:
    # the thinking header ticks at 0.1s, so there is always an event.
    #
    # Scored as a budget rather than a single worst gap. The renderer clamps long holds,
    # so "longest gap" would be satisfied by the clamp and stop meaning anything, while a
    # reel that spends a third of its time waiting for screens to mount is still a bad
    # reel however it is clamped. This asks what fraction of the take was stall.
    content = [t for t, d in events if not _NOISE.match(d.encode("utf-8", "replace"))]
    if len(content) < 2:
        rows.append(Row("no_dead_air", "UNTESTED", "fewer than two content events"))
    else:
        gaps = np.diff(content)
        stall = float(np.clip(gaps - MAX_CONTENT_GAP, 0, None).sum())
        span = content[-1] - content[0]
        share = stall / span if span else 0.0
        rows.append(Row("no_dead_air", "PASS" if share <= MAX_STALL_SHARE else "FAIL",
                        f"{stall:.1f}s of stall over {span:.1f}s = {share:.0%} "
                        f"(limit {MAX_STALL_SHARE:.0%}), longest gap {gaps.max():.1f}s"))
    return rows


def render_gate(gif: pathlib.Path,
                motion_idx: set[int] | None = None) -> list[Row]:
    """Everything measurable from the shipping gif. Never compare a gif to an mp4 frame."""
    im = Image.open(gif)
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    durs, im2 = [], Image.open(gif)
    try:
        while True:
            durs.append(im2.info.get("duration", 0))
            im2.seek(im2.tell() + 1)
    except EOFError:
        pass
    rows: list[Row] = []

    # The duplicate-padding row that used to live here has been retired. Two measured
    # facts killed it. GIF encoders always collapse byte-identical consecutive frames:
    # writing 344 padded frames reads back as 86, so the defect physically cannot survive
    # into a shipped gif and the row was measuring something impossible. And when a truly
    # padded capture IS fed in, motion_fps catches it exactly -- it reported 6.2fps, which
    # is the precise figure the weak-pod VHS capture measured and the reason for the
    # migration. One falsifiable row beats two, one of which could never go red.
    size_mb = gif.stat().st_size / 1e6
    rows.append(Row("gif_size", "PASS" if size_mb <= MAX_GIF_MB else "FAIL",
                    f"{size_mb:.2f}MB (cap {MAX_GIF_MB}MB)"))

    # Text metrics are medians over sampled frames, never one frame. Every single-frame
    # rule tried here picked the wrong frame on some reel: the middle frame is a coloured
    # progress bar on a launch reel (ink reported as orange) and is post-theme-cycle on a
    # palette reel (a colour regression that was not one); the frame with the most bright
    # pixels is a modal, whose filled border blocks read as 10px-wide "strokes".
    stroke_per_frame, colour_per_frame = [], []
    for f in frames[:: max(1, len(frames) // 16)]:
        a = np.asarray(f)
        lum = a.mean(axis=2)
        bright = a[lum > 170]
        if bright.size < 3 * 200:      # essentially blank; nothing to judge
            continue
        colour_per_frame.append(bright.reshape(-1, 3).mean(axis=0))
        runs: list[int] = []
        for row in lum > 140:
            n = 0
            for v in row:
                if v:
                    n += 1
                elif n:
                    runs.append(n)
                    n = 0
            if n:
                runs.append(n)
        if runs:
            stroke_per_frame.append(runs)

    if not stroke_per_frame:
        rows.append(Row("stroke_weight", "FAIL", "no bright text found in any sampled frame"))
    else:
        means = [float(np.mean(r)) for r in stroke_per_frame]
        med = float(np.median(means))
        common = collections.Counter(r for runs in stroke_per_frame for r in runs)
        rows.append(Row("stroke_weight", "PASS" if med >= MIN_STROKE_MEAN else "FAIL",
                        f"median run {med:.2f}px over {len(means)} frames "
                        f"(floor {MIN_STROKE_MEAN}px), top {common.most_common(2)}"))

    if not colour_per_frame:
        rows.append(Row("near_white_text", "FAIL", "no bright pixels in any sampled frame"))
    else:
        avg = np.median(np.array(colour_per_frame), axis=0)
        dist = float(np.abs(avg - np.array(NEAR_WHITE)).max())
        rows.append(Row("near_white_text", "PASS" if dist <= NEAR_WHITE_TOLERANCE else "FAIL",
                        f"median bright {tuple(int(x) for x in avg)} vs target {NEAR_WHITE} "
                        f"(max channel delta {dist:.0f}, tolerance {NEAR_WHITE_TOLERANCE}, "
                        f"{len(colour_per_frame)} frames)"))

    # Motion frame rate: the falsifiable version of "is it choppy". agg emits a frame per
    # content change, so a deliberate two-second dwell is a single frame with a two-second
    # duration and would otherwise be scored as 0.5fps. Frames held longer than HOLD_MS are
    # therefore excluded as pacing rather than dropped frames; what remains is the reel
    # while it is actually animating. A capture with too few animating frames is UNTESTED,
    # not a pass, because the median of a handful of frames measures nothing.
    # Frame rate, scored only where the driver was the thing producing motion: typing and
    # burst scrolls. Everywhere else the cadence is the app's -- an unpack bar ticks about
    # every 200ms, a token stream arrives when the model produces it -- and scoring those
    # failed a launch reel at 5fps for rendering a progress bar exactly as it happened.
    # The regression this row exists to catch is the opposite case: typing throttled to
    # 75ms rendered a whole reel at 12.5fps. Inside a driver span there is nothing but the
    # driver's own cadence, so the median measures precisely that.
    #
    # Every frame agg emits is already a content change, so duration alone separates
    # moving from dwelling. Requiring a mean pixel difference (an earlier version used
    # 0.5) silently excluded typing: one character in a 1404x907 frame moves about 0.03%
    # of the pixels, so the row measured only full-screen repaints.
    if motion_idx:
        moving = [d for i, d in enumerate(durs) if i in motion_idx and 0 < d <= HOLD_MS]
        scope = f"{len(motion_idx)} driver-motion frames"
    else:
        # No spans means the cast predates span recording. Scoring the whole reel instead
        # measures a different thing -- mostly the model's token cadence -- and calling
        # that the same row would let an old take pass a check it never took.
        rows.append(Row("motion_fps", "UNTESTED",
                        "no driver-motion spans in this take; re-record to score it"))
        rows.append(Row("duration", "PASS", f"{sum(durs)/1000:.1f}s, {len(frames)} frames, "
                                            f"{frames[0].size[0]}x{frames[0].size[1]}"))
        return rows
    if len(moving) < MIN_MOTION_FRAMES:
        rows.append(Row("motion_fps", "UNTESTED",
                        f"only {len(moving)} frames in {scope} (need {MIN_MOTION_FRAMES}); "
                        "nothing the driver does in this reel is long enough to measure"))
    else:
        med = float(np.median(moving))
        fps = 1000.0 / med
        rows.append(Row("motion_fps", "PASS" if fps >= MIN_MOTION_FPS else "FAIL",
                        f"{fps:.1f} fps across {scope} (floor {MIN_MOTION_FPS:.0f}); "
                        f"median {med:.0f}ms over {len(moving)} frames"))

    rows.append(Row("duration", "PASS", f"{sum(durs)/1000:.1f}s, {len(frames)} frames, "
                                        f"{frames[0].size[0]}x{frames[0].size[1]}"))
    return rows


def scorecard(name: str, rows: list[Row]) -> tuple[str, bool]:
    fails = [r for r in rows if r.status == "FAIL"]
    untested = [r for r in rows if r.status == "UNTESTED"]
    ok = not fails and not untested
    head = f"{name}: {'READY' if ok else 'NOT DONE'}"
    if fails:
        head += f"  ({len(fails)} failed)"
    if untested:
        head += f"  ({len(untested)} untested -- an untested row is not a pass)"
    return "\n".join([head, *(str(r) for r in rows)]), ok


def selftest(gif: pathlib.Path, cast: pathlib.Path) -> str:
    """Prove each threshold can go red. A gate that has never failed is decoration."""
    import tempfile

    out = ["falsification pass -- each row must go FAIL against a broken input"]
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)

        # 1. Choppy motion. This is the row that replaced the duplicate-padding check:
        # stretching every frame to 160ms is exactly the weak-capture regression that
        # shipped once, and it must go red.
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(gif))]
        choppy = d / "choppy.gif"
        frames[0].save(choppy, save_all=True, append_images=frames[1:], duration=160, loop=0)
        # Spans covering the whole reel: the row only scores driver-motion windows, so a
        # falsification input has to declare that everything in it is driver motion.
        r = next(x for x in render_gate(choppy, motion_idx=set(range(len(frames))))
                 if x.name == "motion_fps")
        out.append(f"  choppy motion     -> {r.status} ({r.detail})")

        # 2. Grey text: dim the frames toward the 1px-stem condition.
        greyed = d / "grey.gif"
        # 0.78 keeps pixels above the brightness floor so this tests the colour
        # target rather than trivially finding no bright text at all.
        dim = [Image.fromarray((np.array(f) * 0.78).astype("uint8")) for f in frames]
        dim[0].save(greyed, save_all=True, append_images=dim[1:], duration=40, loop=0)
        r = next(x for x in render_gate(greyed) if x.name == "near_white_text")
        out.append(f"  grey text         -> {r.status} ({r.detail})")

        # 3. Spliced dead air: a long gap between content events.
        spliced = d / "spliced.cast"
        lines = cast.read_text().splitlines()
        head, body = lines[0], lines[1:]
        mid = len(body) // 2
        gap = json.loads(body[mid])
        gap[0] = gap[0] + 30.0
        body[mid] = json.dumps(gap)
        spliced.write_text("\n".join([head, *body]) + "\n")
        r = next(x for x in cast_gate(spliced) if x.name == "no_dead_air")
        out.append(f"  spliced dead air  -> {r.status} ({r.detail})")

        # 4. Error text in the stream.
        broke = d / "broke.cast"
        broke.write_text(cast.read_text() + json.dumps([9999.0, "o", "Traceback (most recent call last)"]) + "\n")
        r = next(x for x in cast_gate(broke) if x.name == "no_error_text")
        out.append(f"  error text        -> {r.status} ({r.detail})")

        # 5. A must_string that is not there.
        r = next(x for x in cast_gate(cast, must=("this string is not in the reel",))
                 if x.name == "must_strings")
        out.append(f"  missing must      -> {r.status} ({r.detail})")

    reds = sum(1 for line in out[1:] if "-> FAIL" in line)
    out.append(f"  {reds}/5 gates went red on demand")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    kit = pathlib.Path(__file__).resolve().parent
    gif = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else kit / "out/tui-palette.gif"
    cast = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else kit / "out/tui-palette.cast"
    rows = cast_gate(cast, must=("Slash Commands", "Search for commands")) + render_gate(gif)
    text, ok = scorecard(gif.stem, rows)
    print(text)
    print()
    print(selftest(gif, cast))
    raise SystemExit(0 if ok else 1)
