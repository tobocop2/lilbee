#!/usr/bin/env python3
"""Widen the box-drawing and block-element glyphs so adjacent cells touch.

A progress bar, a panel border and a sparkline are all runs of glyphs in adjacent cells,
and they came out segmented: a hairline of background at every cell boundary, which reads
as tearing.

The cause is a rounding mismatch, not the renderer -- resvg and fontdue produce identical
seams. FULL BLOCK carries an advance of 0.600 em and ink spanning exactly [0, 0.600], so
in font units it fills its cell perfectly. FreeType then reports the advance in 1/64ths of
a pixel and rounds 0.600 em up to 39/64 = 0.609375 em, so the cell agg lays out is wider
than the ink it puts in it. At font size 18 that is 10.969px of cell against 10.8px of
glyph: 0.17px of background per boundary.

Rounding up the advance would need the font's own advance to be a whole number of 1/64ths
at every size, which no single value satisfies. Extending the ink does: these glyphs are
solid shapes whose whole purpose is to tile, so a little overhang past the advance is
invisible except where it closes the seam. Terminal emulators that care about this draw
box characters themselves for the same reason.

The overhang is proportional, so one factor works at every font size.
"""
from __future__ import annotations

import pathlib
import shutil

from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

KIT = pathlib.Path(__file__).resolve().parent
SRC = KIT / "fonts-extrabold/LilbeeReelMono-ExtraBold.ttf"

# Box drawing (U+2500-257F) and block elements (U+2580-259F). Braille and the powerline
# range are left alone: they are not meant to tile edge to edge.
TILING = range(0x2500, 0x25A0)

# 1/64 em is the worst-case rounding error, so 1.5/64 of the advance covers it at any
# size with a margin, and is still under a fifth of a pixel at the sizes reels use.
OVERHANG = 1.5 / 64


def widen(src: pathlib.Path = SRC, factor: float = 1.0 + OVERHANG) -> dict:
    font = TTFont(src)
    glyf, hmtx, cmap = font["glyf"], font["hmtx"], font.getBestCmap()
    glyph_set = font.getGlyphSet()
    touched, skipped = [], []

    for cp in TILING:
        name = cmap.get(cp)
        if not name or name not in glyf:
            continue
        glyph = glyf[name]
        if not glyph.numberOfContours or glyph.isComposite():
            skipped.append(name)
            continue
        advance, lsb = hmtx[name]
        rec = RecordingPen()
        glyph_set[name].draw(rec)
        pen = TTGlyphPen(glyph_set)
        # Scale about the cell centre so the overhang is split between both edges and a
        # run stays centred on the same pixels it was.
        cx = advance / 2.0
        rec.replay(TransformPen(pen, (factor, 0, 0, 1, cx * (1 - factor), 0)))
        glyf[name] = pen.glyph()
        glyf[name].recalcBounds(glyf)
        hmtx[name] = (advance, glyf[name].xMin)
        touched.append(name)

    out = src.with_name(src.stem + ".tiled.ttf")
    font.save(out)
    return {"out": out, "widened": len(touched), "skipped_composite": len(skipped)}


def main() -> None:
    info = widen()
    out = info.pop("out")
    backup = SRC.with_suffix(".ttf.orig")
    if not backup.exists():
        shutil.copy(SRC, backup)
    shutil.move(out, SRC)
    print({**info, "installed": str(SRC), "backup": str(backup)})


if __name__ == "__main__":
    main()
