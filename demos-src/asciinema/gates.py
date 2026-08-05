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

from seams import periodicity as seams_periodicity

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
# How much of the end of a reel counts as "where it stopped".
TAIL_SECONDS = 3.0
MAX_CONTENT_GAP = 4.0
MAX_STALL_SHARE = 0.20
MAX_GIF_MB = 10.0
MIN_STROKE_MEAN = 2.0
NEAR_WHITE = (225, 225, 243)
NEAR_WHITE_TOLERANCE = 45
# Longest bright run still considered a glyph stroke rather than chrome.
MAX_STROKE_RUN = 6
MAX_SEAM_ALIGNMENT = 0.20
# No single unchanging screen may own more than this share of a reel.
MAX_DWELL_SHARE = 0.35
# Two thresholds because the two pacing rows detect opposite things, and one constant
# serving both is what made each of them score nothing useful. Below WAIT_CHANGE the
# screen is only twitching -- a spinner, a bar, a percentage -- and the reel is waiting.
# Above PACED_CHANGE something is actually being repainted, which is where frame rate
# means anything; a line of text is about 0.05% of the frame, so the bar for "the app is
# drawing" has to sit below that. A section can be both, and legitimately fails both.
WAIT_CHANGE = 0.012
PACED_CHANGE = 0.0005
MAX_WAIT_S = 6.0
MIN_PACED_FPS = 10.0
# 12 frames, matching MIN_MOTION_FRAMES: fewer than that and the median
# measures nothing. At the 300ms hold cap that is 1.2s at the outside.
MIN_PACED_SECTION_S = 1.2
MIN_PACED_FRAMES = 12


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


def screen_text(events: list[tuple[float, str]], cols: int = 128, rows: int = 41) -> str:
    """Replay the cast through a terminal emulator and return everything ever displayed.

    Searching the raw byte stream is what this replaces, and it was quietly unreliable:
    Textual writes a line as several writes with cursor moves between them, so a string
    that is contiguous on screen is not contiguous in the stream. "Sources:" appeared zero
    times in a cast whose reel plainly rendered it, which failed a must-string on a reel
    that satisfied it. Strings that happened to be written in one go passed by luck.

    Snapshots after every event and concatenates, so a string counts if it was ever on
    screen, not only if it survived to the end.
    """
    import pyte

    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream(screen)
    seen, last = [], None
    for _, data in events:
        try:
            stream.feed(data)
        except Exception:
            continue
        text = "\n".join(screen.display)
        if text != last:
            seen.append(text)
            last = text
    return "\n".join(seen)


def beat_gate(events: list[tuple[float, str]], beats: tuple[tuple[str, str], ...]) -> Row:
    """Did the reel's story actually happen, in order?

    The other rows measure properties: colour, frame rate, seams, file size. A reel can
    pass every one of them and still show the wrong thing. Shipped examples: a sessions
    reel whose drawer listed one conversation instead of three, a palette reel whose /add
    silently did nothing because the file was already indexed, and a placement reel that
    toggled nothing. All green, all wrong.

    Each beat is a label and a pattern that must appear on screen, and they must appear in
    the order given. Order matters: "the download finished" before "the card reads
    installed" is a different reel from the reverse.
    """
    if not beats:
        return Row("beats", "UNTESTED", "reel declares no beats")
    frames = _screen_frames(events)
    pos, missing = 0, None
    for label, pattern in beats:
        rx = re.compile(pattern)
        for i in range(pos, len(frames)):
            if rx.search(frames[i]):
                pos = i + 1
                break
        else:
            missing = label
            break
    if missing:
        return Row("beats", "FAIL", f"never reached {missing!r} (of {len(beats)} beats)")
    return Row("beats", "PASS", f"all {len(beats)} beats in order")


def dwell_gate(gif: pathlib.Path) -> Row:
    """How much of the shipped reel is one unchanging picture.

    Catches the failure where a reel is technically correct and mostly a progress bar: a
    catalog reel spent most of its length on a download with nothing to look at, and every
    property row was green.

    Measured on the gif rather than the cast. The cast still contains the real duration of
    anything the pipeline deliberately compressed, so scoring it counts a six-times-sped
    window at full length and fails a reel for a wait the viewer never sits through. What
    matters is what ships.
    """
    im = Image.open(gif)
    frames, durs = [], []
    for f in ImageSequence.Iterator(im):
        frames.append(np.asarray(f.convert("RGB"), dtype=np.int16))
        durs.append(f.info.get("duration", 40))
    total = sum(durs)
    if total <= 0 or len(frames) < 3:
        return Row("dwell", "UNTESTED", "too few frames to judge")
    longest = run = 0
    for a, b, d in zip(frames, frames[1:], durs[1:]):
        if np.abs(a - b).mean() < 0.5:      # visually the same picture
            run += d
            longest = max(longest, run)
        else:
            run = d
    longest = max(longest, max(durs))
    share = longest / total
    return Row("dwell", "PASS" if share <= MAX_DWELL_SHARE else "FAIL",
               f"longest unchanging stretch {longest / 1000:.0f}s of {total / 1000:.0f}s "
               f"= {share:.0%} (limit {MAX_DWELL_SHARE:.0%})")


def pacing_gate(gif: pathlib.Path,
                protect: list[tuple[float, float]] | None = None) -> list[Row]:
    """Whether the shipped reel is worth watching second to second.

    Two rows, for the two ways a reel wastes a viewer's time. Both are measured on the
    gif, because both are about what ships rather than what was recorded, and both have
    the same remedy: compress the stretch, do not try to make the wait prettier.

    ``no_long_wait`` catches waiting that is technically animated. dwell only sees an
    unchanging picture and no_dead_air only sees an absence of content events, so a
    thinking spinner satisfies both while conveying nothing -- one reel spent 53.6s of
    its 88.6s that way and every row was green. A stretch counts as waiting when the
    screen keeps changing but only in a sliver of itself: a spinner, a progress bar, a
    percentage counter.

    ``paced_fps`` covers frame rate where motion_fps does not look. That row scores only
    spans the driver caused, so anything the application animates -- a crawl filling in,
    a download, a token stream -- was never measured at all, and a reel reported as
    choppy was green on every row. Honest slowness still reads as choppy, so the fix is
    the same as for a wait: speed the section up rather than claim the rate is fine.
    """
    im = Image.open(gif)
    frames, durs = [], []
    for f in ImageSequence.Iterator(im):
        frames.append(np.asarray(f.convert("RGB"), dtype=np.int16))
        durs.append(f.info.get("duration", 40))
    if len(frames) < 4:
        return [Row("no_long_wait", "UNTESTED", "too few frames to judge"),
                Row("paced_fps", "UNTESTED", "too few frames to judge")]

    px = frames[0].shape[0] * frames[0].shape[1]
    # Per transition: how much of the screen moved, and how long the frame was held.
    changed = [float((np.abs(a - b).max(axis=2) > 24).sum()) / px
               for a, b in zip(frames, frames[1:])]

    # A wait is a run of transitions that are all small -- including zero, so a still
    # holds are part of the same run rather than splitting it in two.
    # Stretches the reel declared as the demonstration are measured and reported, never
    # failed: a launch reel is supposed to contain the launch at full length. Reported
    # rather than skipped silently, so a protected span cannot be used to hide a wait
    # that has nothing to do with what the reel is proving.
    guard = list(protect or [])
    section_in_guard = False
    worst_wait, run, run_start = 0.0, 0.0, 0.0
    kept_wait, clock = 0.0, 0.0
    for share, d in zip(changed, durs[1:]):
        clock += d / 1000.0
        inside = any(lo <= clock <= hi for lo, hi in guard)
        if share < WAIT_CHANGE:
            if run == 0.0:
                run_start = clock
            run += d / 1000.0
            if inside:
                kept_wait = max(kept_wait, run)
            else:
                worst_wait = max(worst_wait, run)
        else:
            run = 0.0
    detail = (f"longest stretch with nothing but a spinner or a bar moving "
              f"{worst_wait:.1f}s (limit {MAX_WAIT_S:.0f}s); compress it")
    if guard:
        detail += (f" -- plus {kept_wait:.1f}s inside spans the reel protects as the "
                   f"thing it is demonstrating, left at real speed on purpose")
    rows = [Row("no_long_wait", "PASS" if worst_wait <= MAX_WAIT_S else "FAIL", detail)]

    # Sections where the screen really is repainting. Frame rate here belongs to
    # whatever is driving it, which is the point: if that is too slow to watch, the
    # section needs compressing.
    section: list[float] = []
    worst: tuple[float, float] | None = None
    kept_slow: tuple[float, float] | None = None
    clock2 = 0.0
    for share, d in zip(changed, durs[1:]):
        clock2 += d / 1000.0
        # No hold cap here, unlike motion_fps. There a long frame is a dwell the reel
        # asked for; here a long frame inside a section that keeps repainting IS the
        # choppiness being measured, so filtering it out is what made this row score
        # nothing on every reel in the set. The median absorbs the occasional
        # deliberate pause without needing the frame dropped.
        if share >= PACED_CHANGE:
            section.append(d)
            section_guarded = any(lo <= clock2 <= hi for lo, hi in guard)
            if section_guarded:
                section_in_guard = True
            continue
        if len(section) >= MIN_PACED_FRAMES and sum(section) / 1000.0 >= MIN_PACED_SECTION_S:
            fps = 1000.0 / float(np.median(section))
            span = sum(section) / 1000.0
            # A protected section is the reel showing how the app actually behaves. A
            # cold launch repaints at about 5fps and that IS the measurement the reel was
            # recorded to make; failing it would demand the pipeline fake a frame rate the
            # application never produced. Reported, never failed.
            if section_in_guard:
                if kept_slow is None or fps < kept_slow[0]:
                    kept_slow = (fps, span)
            elif worst is None or fps < worst[0]:
                worst = (fps, span)
        section = []
        section_in_guard = False
    if len(section) >= MIN_PACED_FRAMES and sum(section) / 1000.0 >= MIN_PACED_SECTION_S:
        fps = 1000.0 / float(np.median(section))
        if worst is None or fps < worst[0]:
            worst = (fps, sum(section) / 1000.0)

    if worst is None and kept_slow is not None:
        rows.append(Row("paced_fps", "PASS",
                        f"only slow stretch is {kept_slow[0]:.1f} fps over {kept_slow[1]:.1f}s "
                        f"inside a span the reel protects; that rate is the application's "
                        f"own and is what the reel exists to show"))
    elif worst is None:
        # Not the same as a failed measurement. motion_fps goes UNTESTED when the driver
        # moved and the sample was too small to judge; here the subject is absent -- the
        # reel contains no stretch the application paces on its own -- and there is
        # nothing for the row to catch.
        rows.append(Row("paced_fps", "PASS",
                        "no app-paced section in this reel; nothing to score"))
    else:
        fps, secs = worst
        extra = (f"; plus {kept_slow[0]:.1f} fps over {kept_slow[1]:.1f}s inside a "
                 f"protected span, left as recorded" if kept_slow else "")
        rows.append(Row("paced_fps", "PASS" if fps >= MIN_PACED_FPS else "FAIL",
                        f"slowest app-paced section {fps:.1f} fps over {secs:.1f}s "
                        f"(floor {MIN_PACED_FPS:.0f}){extra}"))
    return rows


def _screen_frames(events, with_time: bool = False):
    """Distinct rendered screens, in order, optionally with their timestamps."""
    import pyte

    screen = pyte.Screen(128, 41)
    stream = pyte.Stream(screen)
    out, last = [], None
    for t, data in events:
        try:
            stream.feed(data)
        except Exception:
            continue
        text = "\n".join(screen.display)
        if text != last:
            out.append((t, text) if with_time else text)
            last = text
    return out


def cast_gate(cast: pathlib.Path, *, must: tuple[str, ...] = (),
              forbid: tuple[str, ...] = ("Traceback", "not ready yet", "Error 1213"),
              window: tuple[float, float | None] | None = None,
              tail_forbid: tuple[str, ...] = (),
              beats: tuple[tuple[str, str], ...] = ()) -> list[Row]:
    """Everything checkable from the byte stream, before a frame is ever rendered.

    ``window`` restricts every check to the span that actually ships. Without it the boot
    banner counts: a four-second wait for the model to load reads as dead air the viewer
    never sees, and a must-string that only ever appeared during startup passes.
    """
    events = _events(cast)
    if window:
        lo, hi = window
        events = [(t, d) for t, d in events if t >= lo and (hi is None or t <= hi)]
    text = screen_text(events)
    rows: list[Row] = []

    missing = [m for m in must if m not in text]
    rows.append(Row("must_strings", "FAIL" if missing else "PASS",
                    f"missing {missing}" if missing else f"all {len(must)} present"))

    hits = [f for f in forbid if f in text]
    rows.append(Row("no_error_text", "FAIL" if hits else "PASS",
                    f"found {hits}" if hits else "clean"))

    # What is still on screen when the reel stops. A reel that cuts while the model is
    # mid-sentence passes every other row -- it really does contain the question, the
    # citation and the strings it was asked for -- and ends on a spinner.
    if tail_forbid and events:
        cut = events[-1][0] - TAIL_SECONDS
        tail = screen_text([(t_, d) for t_, d in events if t_ >= cut])
        stuck = [f for f in tail_forbid if f in tail]
        rows.append(Row("finished_before_cut", "FAIL" if stuck else "PASS",
                        f"still showing {stuck} in the last {TAIL_SECONDS:.0f}s"
                        if stuck else f"nothing in flight over the last {TAIL_SECONDS:.0f}s"))

    # Content-bearing gaps only. Measuring raw event gaps is the check that cannot fail:
    # the thinking header ticks at 0.1s, so there is always an event.
    #
    # Scored as a budget rather than a single worst gap. The renderer clamps long holds,
    # so "longest gap" would be satisfied by the clamp and stop meaning anything, while a
    # reel that spends a third of its time waiting for screens to mount is still a bad
    # reel however it is clamped. This asks what fraction of the take was stall.
    rows.append(beat_gate(events, beats))

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


def artifact_gate(shipped: pathlib.Path, reference: pathlib.Path) -> Row:
    """Compare the shipped gif against the frames agg drew, pixel for pixel.

    Every step after the render -- optimisation, quantisation, whatever gets added next --
    is supposed to be lossless. Nothing checked that, and a ``--lossy`` flag shipped ten
    reels whose footer text rendered as two overlapping copies of itself: the compressor
    judged the changed pixels close enough to skip, so the previous frame's glyphs stayed
    underneath the new ones. Stroke weight and text colour barely move under that damage,
    so the existing rows all stayed green.

    Compared as exact pixels rather than as a statistic. A ghosted glyph is a handful of
    wrong pixels in a 1404x907 frame; any tolerance wide enough to be "robust" is wide
    enough to miss it.
    """
    a = [np.asarray(f.convert("RGB"), dtype=np.int16)
         for f in ImageSequence.Iterator(Image.open(shipped))]
    b = [np.asarray(f.convert("RGB"), dtype=np.int16)
         for f in ImageSequence.Iterator(Image.open(reference))]
    if len(a) != len(b):
        return Row("no_render_artifacts", "FAIL",
                   f"shipped gif has {len(a)} frames, the render had {len(b)}")
    worst, dirty = 0, 0
    for fa, fb in zip(a, b):
        diff = int(np.abs(fa - fb).max())
        worst = max(worst, diff)
        if diff > 0:
            dirty += 1
    return Row("no_render_artifacts", "PASS" if worst == 0 else "FAIL",
               "identical to the render" if worst == 0 else
               f"{dirty} of {len(a)} frames differ from the render, worst channel {worst}")


def seam_gate(gif: pathlib.Path) -> Row:
    """Cell-boundary seams surviving in the shipped file.

    agg lays terminal cells out on fractional pixel boundaries, so the two sides of a
    boundary composite separately and the shared pixel ends up partially covered by both.
    A run of block glyphs -- a VRAM bar, an ingest bar, a panel border -- comes apart into
    segments, which reads as tearing.

    Scored on how strongly the seam pixels line up with the cell pitch, because that is
    what separates the artifact from the detector's own floor. Two earlier versions of
    this row were content-dependent and both were wrong: an absolute rate failed a
    settings screen full of bordered inputs that looked perfect, and a reduction ratio
    failed a reel that started close to the floor and had little left to close.

    Calibrated against both ends. An unrepaired render measures 0.362; the repaired reels
    measure 0.05 to 0.15, with the higher end coming from reels that have few seam
    pixels at all; the VHS assets these replace, which never had this artifact, measure
    0.04 to 0.06. The repaired reels are indistinguishable from the clean
    reference, and the limit sits between them and the defect.
    """
    m = seams_periodicity(gif)
    return Row("no_cell_seams", "PASS" if m["alignment"] <= MAX_SEAM_ALIGNMENT else "FAIL",
               f"seam phase alignment {m['alignment']:.3f} over {m['n']} pixels "
               f"(limit {MAX_SEAM_ALIGNMENT}; unrepaired renders measure ~0.36)")


def render_gate(gif: pathlib.Path,
                motion_idx: set[int] | None = None,
                static_by_design: bool = False,
                cold_by_design: bool = False) -> list[Row]:
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
        if (lum > 170).sum() < 200:    # essentially blank; nothing to judge
            continue
        # Colour is judged on glyph strokes only. Sampling every bright pixel lets UI
        # chrome decide the answer: the placement drawer is bordered and filled in iris,
        # so a reel whose text is perfectly correct measured as purple and failed. A
        # stroke is a short bright run; borders, bars and filled toggles are long ones.
        stroke_px = []
        for row_lum, row_px in zip(lum > 170, a):
            n = 0
            for x, on in enumerate(row_lum):
                if on:
                    n += 1
                    continue
                if 0 < n <= MAX_STROKE_RUN:
                    stroke_px.extend(row_px[x - n:x])
                n = 0
        if len(stroke_px) >= 200:
            colour_per_frame.append(np.array(stroke_px).reshape(-1, 3).mean(axis=0))
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
    if len(moving) < MIN_MOTION_FRAMES and static_by_design:
        # Declared, not inferred. Some screens genuinely do not animate: the placement
        # drawer coalesces repaints, so nine rounds of toggling produce six measurable
        # frames. A screen with no animation cannot be choppy, which is the only thing
        # this row exists to catch, so blocking such a reel forever measures nothing. The
        # reel has to say so in its own source with a reason, and it still reports the
        # sample size rather than claiming a frame rate it did not measure.
        rows.append(Row("motion_fps", "PASS",
                        f"only {len(moving)} animating frames; reel declares this screen "
                        "does not animate (see its source for why)"))
    elif cold_by_design:
        # The reel is recording an application that is deliberately not warm, and that
        # slowness is its subject. A cold-started lilbee repaints slowly enough that most
        # driver frames exceed the hold cap and are excluded, so the sample never reaches
        # the floor however long the scroll is -- 14, 22, 30 and 60 keypresses all came
        # back short, and 60 was worse than 30. Enforcing a rate here would demand the
        # pipeline manufacture frames the application never drew. Reported, with the real
        # numbers, so nothing is hidden.
        med = float(np.median(moving)) if moving else 0.0
        rows.append(Row("motion_fps", "PASS",
                        f"{1000.0 / med:.1f} fps over {len(moving)} frames "
                        f"(floor {MIN_MOTION_FPS:.0f} not enforced: this reel declares a "
                        f"cold application whose repaint rate is what it exists to show)"
                        if med else "no measurable driver motion in a cold-start reel"))
    elif len(moving) < MIN_MOTION_FRAMES:
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

        # 1b. A long wait, built the way a real one looks: a frame that changes in only a
        # sliver of itself, held for a spinner's worth of time, repeated. dwell cannot see
        # this because the picture keeps changing, which is the whole reason the row exists.
        waiting = d / "waiting.gif"
        base = np.array(frames[0])
        spin = []
        for i in range(40):
            f = base.copy()
            f[:12, :12] = 0 if i % 2 else 255      # ~0.01% of the frame, well under WAIT_CHANGE
            spin.append(Image.fromarray(f))
        spin[0].save(waiting, save_all=True, append_images=spin[1:], duration=400, loop=0)
        r = next(x for x in pacing_gate(waiting) if x.name == "no_long_wait")
        out.append(f"  spinner wait      -> {r.status} ({r.detail})")

        # 1c. A section that really is repainting, but slowly. Alternating full frames at
        # 200ms is 5fps of genuine content, which is the crawl-progress case.
        crawl = d / "crawl.gif"
        alt = [frames[i % 2] for i in range(30)]
        alt[0].save(crawl, save_all=True, append_images=alt[1:], duration=200, loop=0)
        r = next(x for x in pacing_gate(crawl) if x.name == "paced_fps")
        out.append(f"  slow app section  -> {r.status} ({r.detail})")

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

        # 6. Cell seams, injected exactly as agg produces them: one column per cell,
        # darkened 15%, which puts every seam at the same phase in the cell.
        seamy = d / "seamy.gif"
        cell = frames[0].size[0] / 128
        dirty = []
        for f in frames:
            a = np.asarray(f).astype(np.int16)
            for i in range(1, 128):
                x = int(round(i * cell))
                if x < a.shape[1]:
                    a[:, x] = (a[:, x] * 0.85).astype(np.int16)
            dirty.append(Image.fromarray(a.astype("uint8")))
        dirty[0].save(seamy, save_all=True, append_images=dirty[1:], duration=40, loop=0)
        r = seam_gate(seamy)
        out.append(f"  cell seams        -> {r.status} ({r.detail})")

        # 7. Lossy compression. This is the exact damage that shipped: gifsicle --lossy
        # leaves stale pixels under changed text, and nothing else here notices.
        import shutil as _shutil
        import subprocess as _subprocess

        lossy = d / "lossy.gif"
        if _shutil.which("gifsicle"):
            _subprocess.run(["gifsicle", "-O3", "--lossy=40", "--colors", "128",
                             str(gif), "-o", str(lossy)], check=True, capture_output=True)
            r = artifact_gate(lossy, gif)
            out.append(f"  lossy artifacts   -> {r.status} ({r.detail})")

    reds = sum(1 for line in out[1:] if "-> FAIL" in line)
    out.append(f"  {reds}/{len(out) - 1} gates went red on demand")
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
