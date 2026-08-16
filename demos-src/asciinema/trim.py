#!/usr/bin/env python3
"""SUPERSEDED by frametrim.py -- kept only for the v3-to-v2 normalisation notes.

Trim a cast to its payload window, and hold the last frame.

Boot and teardown are not the reel. The shipped tapes hid them with VHS ``Hide``/``Show``;
agg has no equivalent and ``--idle-time-limit`` cannot substitute (measured: a strict
no-op above 0.5s, and below it every real pause compresses, breaking the 1x rule).

Events are never deleted from the middle of a window. Steady-state Textual output is
cursor-addressed diffs with no full clear, so dropping an interior event leaves stale
cells on screen until exit -- and that still passes a must_string check, because the text
is all present in the cast. Only the head and tail are cut, and the head cut is only safe
because the driver forces a full repaint at the boundary (see ``Session.repaint``), so the
first surviving event paints the whole screen rather than a diff against frames that are
no longer there.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _screen_at(header: dict, before: list) -> str:
    """Replay `before` through a terminal emulator and re-emit the resulting screen.

    Returns an ANSI blob that paints the whole screen in one event, so the trimmed cast
    opens on a complete frame rather than on whatever cells happen to survive the cut.
    """
    if not before:
        return ""
    import pyte

    cols = header.get("width") or header.get("term", {}).get("cols", 80)
    rows = header.get("height") or header.get("term", {}).get("rows", 24)
    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream(screen)
    for e in before:
        stream.feed(e[2])

    out = ["\x1b[2J\x1b[H"]
    for y in range(rows):
        line, prev = [], None
        for x in range(cols):
            ch = screen.buffer[y][x]
            style = (ch.fg, ch.bg, ch.bold, ch.reverse)
            if style != prev:
                sgr = ["0"]
                if ch.bold:
                    sgr.append("1")
                if ch.reverse:
                    sgr.append("7")
                if ch.fg != "default":
                    sgr.append(f"38;2;{int(ch.fg[0:2],16)};{int(ch.fg[2:4],16)};{int(ch.fg[4:6],16)}"
                               if len(ch.fg) == 6 else "39")
                if ch.bg != "default":
                    sgr.append(f"48;2;{int(ch.bg[0:2],16)};{int(ch.bg[2:4],16)};{int(ch.bg[4:6],16)}"
                               if len(ch.bg) == 6 else "49")
                line.append("\x1b[" + ";".join(sgr) + "m")
                prev = style
            line.append(ch.data or " ")
        out.append(f"\x1b[{y+1};1H" + "".join(line))
    # Restore the cursor to where it actually was. Without this the repaint leaves it
    # wherever the last cell was written (bottom-right), and agg draws a block cursor
    # there for the rest of the reel -- a second cursor floating in empty space.
    out.append("\x1b[0m")
    out.append(f"\x1b[{screen.cursor.y + 1};{screen.cursor.x + 1}H")
    # Always hide it. Textual draws its own cursor as a styled cell and keeps the real
    # terminal cursor hidden; re-showing it in the preface paints a block on top of
    # whatever occupies that position (it landed on the INSERT chip and garbled it).
    # The app re-enables the cursor itself if it ever wants one.
    out.append("\x1b[?25l")
    return "".join(out)


def trim(cast: pathlib.Path, out: pathlib.Path, *, start: float | None = None,
         end: float | None = None, freeze: float = 2.5) -> dict:
    lines = cast.read_text().splitlines()
    header = json.loads(lines[0])
    events = []
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # v3 stores the interval since the previous event; v2 stores absolute time. Trim
    # windows are wall-clock offsets from the driver, so normalise to absolute first --
    # comparing a window against deltas silently matches nothing.
    v3 = header.get("version") == 3
    if v3:
        clock = 0.0
        absolute = []
        for e in events:
            clock += e[0]
            absolute.append([round(clock, 6), e[1], e[2]])
        events = absolute
        header = {**header, "version": 2,
                  "width": header.get("term", {}).get("cols", 80),
                  "height": header.get("term", {}).get("rows", 24)}
        header.pop("term", None)

    kept = [e for e in events
            if (start is None or e[0] >= start) and (end is None or e[0] <= end)]
    if not kept:
        raise SystemExit(f"trim window [{start}, {end}] kept no events of {len(events)}")

    # Prepend the screen as it stood at the cut, reconstructed by replaying everything
    # before it. Without this the reel opens on a blank frame: Textual repaints with
    # cursor-home plus line writes and never emits a clear-screen, so there is no paint
    # boundary to cut on, and the dropped head contains the only copy of most cells.
    # A blank first frame is the worst possible opening for a README gif that autoplays
    # silently mid-scroll.
    preface = _screen_at(header, [e for e in events if start is not None and e[0] < start])

    base = kept[0][0]
    rebased = [[round(e[0] - base, 6), e[1], e[2]] for e in kept]
    if preface:
        rebased.insert(0, [0.0, "o", preface])
        rebased = [[round(e[0] + (0.05 if i else 0), 6), e[1], e[2]]
                   for i, e in enumerate(rebased)]
    # Hold the final frame. One source of the tail only: the pipeline used to add a hold
    # in two places and the freezes stacked into five seconds of dead air.
    rebased.append([round(rebased[-1][0] + freeze, 6), "o", ""])

    out.write_text("\n".join([json.dumps(header)] + [json.dumps(e) for e in rebased]) + "\n")
    return {"events_in": len(events), "events_out": len(rebased),
            "duration": rebased[-1][0], "dropped_head": sum(1 for e in events if e[0] < (start or 0)),
            "dropped_tail": sum(1 for e in events if end is not None and e[0] > end)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cast", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--freeze", type=float, default=2.5)
    a = ap.parse_args()
    print(trim(a.cast, a.out, start=a.start, end=a.end, freeze=a.freeze))


if __name__ == "__main__":
    main()
