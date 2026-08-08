#!/usr/bin/env python3
"""Record, render, trim and gate one reel, then write its scorecard.

One entry point so no reel can skip a step. The order is fixed: record the full cast,
render the whole thing, then trim in the frame domain (see frametrim). Marks recorded by
the driver are timestamps against the session clock, and agg preserves cast timing, so
frame offsets and mark offsets are the same clock.

Usage: make_reel.py <name> [--no-record]
``--no-record`` re-renders and re-gates the cast already in out/, which is what to use
when only the trim window or the gate thresholds changed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import shutil
import sys
import time

KIT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
OUT = KIT / "out"

import agg_finish  # noqa: E402
import frametrim  # noqa: E402
import deseam  # noqa: E402
import gates  # noqa: E402
import seams  # noqa: E402


def build(name: str, *, record: bool = True,
          record_only: bool = False) -> tuple[str, bool]:
    mod = importlib.import_module(f"reels.{name.replace('-', '_')}")
    cast = OUT / f"{name}.cast"
    marks_path = OUT / f"{name}.marks.json"
    OUT.mkdir(exist_ok=True)

    if record:
        # Record which build this reel actually shows. A reel is evidence about a version,
        # and the version was ambiguous once already: the recording environment reported
        # dev724 from a stale install marker while the code it ran was current main, which
        # cost a round trip to establish. Stamping both removes the question.
        provenance = _provenance()
        print("build " + str(provenance))
        t0 = time.monotonic()
        timings = mod.record(cast)
        timings["wall"] = time.monotonic() - t0
        timings["build"] = provenance
        marks_path.write_text(json.dumps(timings, indent=2))
        print(f"recorded {cast} in {timings['wall']:.0f}s")
        if record_only:
            # Phase boundary. Everything below is a deterministic function of the cast
            # and the marks, so it can run for several reels at once; recording cannot,
            # because the frame rate being measured is the application's own repaint
            # cadence and contention would change the reading. Recording is 26% of the
            # wall time and rendering is 74%, so splitting here is where the speed is.
            return name, True
    timings = json.loads(marks_path.read_text())
    marks = timings.get("marks", {})

    full = OUT / f"{name}-full"
    agg_finish.render(cast, full)
    gif = OUT / f"{name}.gif"
    # Head-trim to the mark, not to zero: the boot banner and shell prompt are not the
    # reel. A reel with no marks ships whole rather than guessing a window.
    # Compress the generation window if the reel marked one. On a laptop most of a chat
    # reel can be a progress bar, and the answer is the part worth watching.
    speedup = []
    for base in getattr(mod, "SPEED_WINDOWS", ("gen",)):
        lo, hi = marks.get(f"{base}_start"), marks.get(f"{base}_end")
        if lo is not None and hi is not None:
            speedup.append((lo, hi, getattr(mod, "SPEED_FACTOR", 6)))
    # Close agg's cell-boundary seams before anything else looks at the file. This is a
    # renderer repair, not a content edit: the terminal drew a solid bar and agg split it.
    #
    # Ahead of the trim, which is what the comment always claimed and the order did not.
    # Repairing seams erases small pixel differences, so a compressor working on the
    # unrepaired frames scores transitions the gate will not: waits it measured as two
    # short stretches arrive at the gate welded into one long one, and four reels failed
    # a 6s limit at 6.4-7.3s with no threshold that fixed it. Raising the compressor's
    # margin twice did not close the gap because the inputs genuinely differ. Deseaming
    # first costs frames that later get dropped, and buys both stages the same pixels.
    seam_info = deseam.repair_gif(full.with_suffix(".gif"),
                                  window=(marks.get("boot_end", 0.0),
                                          marks.get("payload_end")))
    print("deseamed " + str(seam_info))
    # Spans this reel declares untouchable: recorded at real speed, holds intact.
    #
    # Driver motion joins them automatically. Typing and scrolls are the only stretches
    # motion_fps can score, and a cold-started app repaints slowly enough that the
    # compressor classified a scroll as a slow section and thinned it -- leaving 11 frames
    # against a floor of 12, so the row went untested and the reel could never pass.
    # Compressing the thing being measured is self-defeating: the driver's cadence is
    # deliberate, and a burst typed at 45ms is content, not a wait.
    protect = []
    for base in getattr(mod, "PROTECT_WINDOWS", ()):
        lo, hi = marks.get(f"{base}_start"), marks.get(f"{base}_end")
        if lo is not None and hi is not None:
            protect.append((lo, hi))
    info = frametrim.trim_gif(full.with_suffix(".gif"), gif,
                              start=marks.get("boot_end", 0.0),
                              # A reel may name an earlier stopping point. The default
                              # tail runs to payload_end, which on the placement reels
                              # trails several seconds of a static answer after the
                              # protected hold has ended -- deliberate on camera, but
                              # unprotected dead air to the gate. Ending on the beat that
                              # matters is better than padding a threshold to allow it.
                              end=marks.get(getattr(mod, "END_MARK", "payload_end"),
                                            marks.get("payload_end")),
                              protect=protect or None,
                              motion=[(lo, hi) for lo, hi in timings.get("motion_spans", [])],
                              speedup=speedup or None)
    print("trimmed " + str({k: v for k, v in info.items() if k != "kept_starts"}))
    # Verify the written file rather than trusting the in-memory pass: saving quantises
    # to a 256-colour palette, which can merge waits the compressor handled separately.
    again = frametrim.compress_waits(gif, limit=gates.MAX_WAIT_S,
                                     protect=info.get("protected"))
    if again.get("windows"):
        print("recompressed " + str(again))
    for stale in (full.with_suffix(".gif"), full.with_suffix(".mp4"), full.with_suffix(".png")):
        stale.unlink(missing_ok=True)
    # Keep what the renderer produced, before anything optimises it, so the shipped file
    # can be compared against it rather than trusted.
    reference = OUT / f"{name}.reference.gif"
    shutil.copy(gif, reference)
    _optimize(gif)
    _derive(gif)

    forbid = ("Traceback", "not ready yet", "Error 1213", "No space left on device",
              *getattr(mod, "FORBID_STRINGS", ()))
    cast_protect = []
    for base in getattr(mod, "PROTECT_WINDOWS", ()):
        lo, hi = marks.get(f"{base}_start"), marks.get(f"{base}_end")
        if lo is not None and hi is not None:
            cast_protect.append((lo, hi))
    rows = gates.cast_gate(cast, must=tuple(getattr(mod, "MUST_STRINGS", ())), forbid=forbid,
                           protect=cast_protect or None,
                           window=(marks.get("boot_end", 0.0), marks.get("payload_end")),
                           tail_forbid=getattr(mod, "TAIL_FORBID", ()),
                           beats=getattr(mod, "BEATS", ()))
    # Map driver-motion spans onto frame indices using each frame's ORIGINAL time. Hold
    # clamping shortens frames, so a position in the finished gif no longer says when that
    # frame happened; matching on output timing put every span in the wrong place and
    # scored zero frames.
    shift = marks.get("boot_end", 0.0)
    spans = [(lo - shift, hi - shift) for lo, hi in timings.get("motion_spans", [])]
    starts = [t - info["kept_starts"][0] for t in info["kept_starts"]]
    motion_idx = {i for i, t in enumerate(starts)
                  if any(lo <= t <= hi for lo, hi in spans)}
    rows += gates.render_gate(gif, motion_idx=motion_idx or None,
                              static_by_design=getattr(mod, "STATIC_BY_DESIGN", False),
                              cold_by_design=getattr(mod, "COLD_BY_DESIGN", False))
    rows.append(gates.artifact_gate(gif, reference))
    rows.append(gates.seam_gate(gif))
    rows.append(gates.dwell_gate(gif))
    rows += gates.pacing_gate(gif, protect=info.get('protected'))
    reference.unlink(missing_ok=True)
    # A strip across the reel, not its last frame. Reviewing the ending is what let a
    # sessions reel with one conversation, a palette reel whose add did nothing and a
    # placement reel that toggled nothing all ship: each was correct at the end.
    sheet = _contact_sheet(gif, OUT / f"{name}-contact.png")
    text, ok = gates.scorecard(name, rows)
    text += f"\n  review: {sheet}"
    (OUT / f"{name}.score.txt").write_text(text + "\n")
    print(text)
    return text, ok


def _contact_sheet(gif: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """Six frames spread across the reel, stacked, for a human to actually look at."""
    from PIL import Image, ImageSequence

    frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(gif))]
    picks = [frames[min(len(frames) - 1, int(len(frames) * p))]
             for p in (0.08, 0.25, 0.42, 0.60, 0.78, 0.96)]
    w, h = picks[0].size
    scale = 0.5
    sheet = Image.new("RGB", (int(w * scale), int(h * scale) * len(picks)))
    for i, f in enumerate(picks):
        sheet.paste(f.resize((int(w * scale), int(h * scale))), (0, int(h * scale) * i))
    sheet.save(out)
    return out


def _provenance() -> dict:
    """What is being recorded: the app version, and the source it actually runs."""
    import subprocess

    def run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except Exception:
            return "?"

    src = run("python3", "-c", "import lilbee,os;print(os.path.dirname(lilbee.__file__))")
    head = run("git", "-C", src or ".", "rev-parse", "--short", "HEAD")
    return {"version": run("lilbee", "--version"), "source": src, "head": head}


def _optimize(gif: pathlib.Path) -> None:
    """Shrink the gif with gifsicle, losslessly.

    ``--lossy`` is not an option here and never was. It decides some changed pixels are
    close enough to skip, so those pixels keep whatever the previous frame left in them,
    and text that changes between frames renders as two overlapping copies of itself --
    the footer hint in every reel came out looking struck through. It shipped once because
    the gates measured stroke weight and text colour, which lossy quantisation barely
    moves, and nothing compared the shipped frames against the ones agg drew. That
    comparison is now its own gate row.

    A full-screen scroll changes nearly every pixel, so inter-frame deltas save little and
    a scrolling reel lands large. If that ever exceeds the size cap the answer is fewer
    beats, not a dirtier picture.
    """
    import shutil
    import subprocess

    if not shutil.which("gifsicle"):
        return
    tmp = gif.with_suffix(".opt.gif")
    subprocess.run(["gifsicle", "-O3", str(gif), "-o", str(tmp)],
                   check=True, capture_output=True)
    if tmp.stat().st_size < gif.stat().st_size:
        tmp.replace(gif)
    else:
        tmp.unlink(missing_ok=True)


def _derive(gif: pathlib.Path) -> None:
    """mp4 and poster come from the shipped gif so they cannot disagree with it."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return
    mp4 = gif.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-i", str(gif), "-movflags", "faststart",
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-sseof", "-0.3", "-i", str(mp4), "-vframes", "1",
                    str(gif.with_suffix(".png"))], check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--record-only", action="store_true")
    a = ap.parse_args()
    _, ok = build(a.name, record=not a.no_record, record_only=a.record_only)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
