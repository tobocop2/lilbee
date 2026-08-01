#!/usr/bin/env python3
"""Render an asciinema cast to gif/mp4/png at the reel render spec.

Three things here are load-bearing and were each established by measurement; see
kit/RENDER-SPEC.md for the evidence.

  * agg's ``--theme custom`` is advertised by ``--help`` and rejected by the parser, so
    rose-pine arrives through the cast header instead. Every colour needs a ``#``; agg
    fails the whole file with "not a v1, v2, v3 asciicast" if any lacks one.
  * The theme header only survives in v2, so v3 recordings are converted first.
  * ``--font-dir`` is additive, not exclusive. Asking for "JetBrains Mono" on a machine
    with the family installed resolves to Regular and renders byte-identically to using
    no font dir at all. The face is therefore renamed to a family that exists nowhere
    else, and the resolved weight is asserted rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

KIT = pathlib.Path(__file__).resolve().parent
FONT_DIR = KIT / "fonts-extrabold"
FONT_FAMILY = "Lilbee Reel Mono"
FONT_FILE = FONT_DIR / "LilbeeReelMono-ExtraBold.ttf"
EXPECT_WEIGHT = 800

FONT_SIZE = 18
FPS_CAP = 25
# agg defaults to 1.4, which makes the cell taller than the glyph box. Block-element
# glyphs (U+2580-259F, what lilbee draws panel edges with) then cannot reach the next
# row, so every border renders as dashes with gaps rather than a continuous line.
# Measured: 1.2 and below tile cleanly, 1.4 segments. 1.2 is the loosest that still
# tiles, so it keeps the most line spacing. Nothing to do with font weight -- ExtraBold
# and Bold segment identically at 1.4.
LINE_HEIGHT = 1.2

# rose-pine (main), matching what lilbee paints with LILBEE_THEME=rose-pine, so the
# terminal background and the app background are the same colour and leave no seam.
THEME = {
    "fg": "#e0def4",
    "bg": "#191724",
    "palette": ":".join(
        "#" + c
        for c in (
            "26233a eb6f92 31748f f6c177 9ccfd8 c4a7e7 ebbcba e0def4 "
            "6e6a86 eb6f92 31748f f6c177 9ccfd8 c4a7e7 ebbcba e0def4"
        ).split()
    ),
}


def preflight() -> None:
    """Refuse to render rather than emit grey text nobody notices until review."""
    for tool in ("agg", "asciinema"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not on PATH")
    if not FONT_FILE.exists():
        sys.exit(f"missing {FONT_FILE}")
    faces = list(FONT_DIR.glob("*.ttf")) + list(FONT_DIR.glob("*.otf"))
    if len(faces) != 1:
        sys.exit(f"font dir must hold exactly one face, found {len(faces)}: {faces}")
    from fontTools.ttLib import TTFont

    font = TTFont(FONT_FILE)
    weight = font["OS/2"].usWeightClass
    if weight != EXPECT_WEIGHT:
        sys.exit(f"font resolves to weight {weight}, expected {EXPECT_WEIGHT}")
    # Name records are UTF-16BE on the Windows platform, so decode rather than str().
    family = {r.toUnicode() for r in font["name"].names if r.nameID == 1}
    if FONT_FAMILY not in family:
        sys.exit(f"font family is {family}, expected {FONT_FAMILY!r}")


def to_v2_themed(cast: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """v3 -> v2, then stamp the theme into the header."""
    tmp = out.with_suffix(".v2.cast")
    header = json.loads(cast.read_text().splitlines()[0])
    if header.get("version") == 2:
        shutil.copy(cast, tmp)
    else:
        subprocess.run(["asciinema", "convert", "-f", "asciicast-v2", str(cast), str(tmp)],
                       check=True, capture_output=True)
    lines = tmp.read_text().splitlines()
    head = json.loads(lines[0])
    head["theme"] = THEME
    tmp.write_text("\n".join([json.dumps(head)] + lines[1:]) + "\n")
    return tmp


def render(cast: pathlib.Path, stem: pathlib.Path, *, speed: float = 1.0) -> dict:
    preflight()
    themed = to_v2_themed(cast, stem)
    gif = stem.with_suffix(".gif")
    subprocess.run(
        ["agg", "--font-dir", str(FONT_DIR), "--font-family", FONT_FAMILY,
         "--font-size", str(FONT_SIZE), "--fps-cap", str(FPS_CAP), "--line-height", str(LINE_HEIGHT),
         "--speed", str(speed), str(themed), str(gif)],
        check=True, capture_output=True,
    )
    out = {"gif": gif}
    # mp4 and png derive from the gif, so they cannot disagree with the shipped frames.
    if shutil.which("ffmpeg"):
        mp4 = stem.with_suffix(".mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(gif), "-movflags", "faststart", "-pix_fmt", "yuv420p",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)],
            check=True, capture_output=True,
        )
        out["mp4"] = mp4
        png = stem.with_suffix(".png")
        subprocess.run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(mp4),
                        "-vframes", "1", str(png)], check=True, capture_output=True)
        out["png"] = png
    themed.unlink(missing_ok=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cast", type=pathlib.Path)
    ap.add_argument("stem", type=pathlib.Path, help="output path without extension")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    for kind, path in render(a.cast, a.stem, speed=a.speed).items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
