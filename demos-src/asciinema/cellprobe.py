#!/usr/bin/env python3
"""Find a font size whose cell width is a whole number of pixels.

A progress bar is a run of block glyphs in adjacent cells. When the advance width at the
chosen font size is fractional, agg rounds each glyph origin independently and the run
comes apart into segments with a one-pixel seam at every cell boundary -- the placement
bar and the Task Center bar both rendered that way at size 18, where 128 columns produced
1404 pixels, or 10.97 per cell.

Prints the rendered size, the implied cell width, and the number of interior background
pixels found inside a solid coloured run.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile

from PIL import Image

KIT = pathlib.Path(__file__).resolve().parent
FONT_DIR = KIT / "fonts-extrabold"
COLS = 128
PALETTE = ":".join(
    "#" + c for c in (
        "26233a eb6f92 31748f f6c177 9ccfd8 c4a7e7 ebbcba e0def4 "
        "6e6a86 eb6f92 31748f f6c177 9ccfd8 c4a7e7 ebbcba e0def4").split())


def build_cast(path: pathlib.Path) -> None:
    header = {"version": 2, "width": COLS, "height": 8,
              "theme": {"fg": "#e0def4", "bg": "#191724", "palette": PALETTE}}
    esc = chr(27)
    bar = f"{esc}[38;2;156;207;216m" + "█" * 40 + f"{esc}[0m"
    path.write_text("\n".join([json.dumps(header),
                               json.dumps([0.1, "o", bar]),
                               json.dumps([0.6, "o", " "])]) + "\n")


def seams(gif: pathlib.Path) -> tuple[int, int, int]:
    im = Image.open(gif).convert("RGB")
    w, h = im.size
    px = im.load()
    row = h // 3
    xs = [x for x in range(w) if px[x, row][1] > 150]
    if not xs:
        return w, h, -1
    gaps = sum(1 for x in range(min(xs), max(xs)) if px[x, row][1] <= 150)
    return w, h, gaps


def main() -> None:
    d = pathlib.Path(tempfile.mkdtemp())
    cast = d / "blocks.cast"
    build_cast(cast)
    for size in range(12, 31):
        out = d / f"s{size}.gif"
        r = subprocess.run(
            ["agg", "--font-dir", str(FONT_DIR), "--font-family", "Lilbee Reel Mono",
             "--font-size", str(size), "--line-height", "1.2", "--fps-cap", "25",
             str(cast), str(out)], capture_output=True)
        if r.returncode:
            print(f"size {size:2d}  render failed")
            continue
        w, h, gaps = seams(out)
        print(f"size {size:2d}  {w}x{h}  cell {w / COLS:7.3f}  "
              f"{'whole' if w % COLS == 0 else 'fractional'}  seams={gaps}")


if __name__ == "__main__":
    main()
