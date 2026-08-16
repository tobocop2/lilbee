#!/usr/bin/env python3
"""Solve per-class 2x canvas dims so terminal cols/rows EXACTLY match 1x.

Naive doubling drifts by one col/row (xterm.js rounds fractional cell sizes:
measured 149x44 vs 148x45 for the flagship class), which shifts wrap points
against the proven choreographies. This solver renders geometry probes,
measures, and nudges Width/Height (a few px, invisible after the gif
downscale to the fixed display size) until both dimensions match.

Runs inside the render environment (golden Docker image or prep pod).
Writes geometry_cal.json: {class: {width, height, cols, rows}} consumed by
retape.py. Exit 1 if any class fails to converge.

Usage: geo_calibrate.py reels.yaml geometry_cal.json
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import yaml

MAX_ITER = 5


def render_probe(cls_name: str, cls: dict, scale: int, width: int, height: int,
                 font_family: str, workdir: pathlib.Path) -> tuple[int, int] | None:
    out = f"/tmp/geo-cal-{cls_name}-{scale}.txt"
    pathlib.Path(out).unlink(missing_ok=True)
    lines = [
        f'Set Shell "{cls["shell"]}"',
        f"Set Width {width}",
        f"Set Height {height}",
        f"Set FontSize {cls['fontsize'] * scale}",
        f'Set FontFamily "{font_family}"',
        f"Set LetterSpacing {scale}",
        f"Set Padding {cls['padding'] * scale}",
        "Set Framerate 25",
    ]
    if cls.get("windowbar"):
        lines += [f"Set WindowBar {cls['windowbar']}", f"Set WindowBarSize {30 * scale}"]
    lines += [
        "Output geo-cal.mp4",  # relative: VHS's parser rejects tmpdir absolute paths
        f'Type "stty size > {out}"',
        "Enter",
        "Sleep 2s",
    ]
    tape = workdir / "geo-cal.tape"
    tape.write_text("\n".join(lines) + "\n")
    r = subprocess.run(["vhs", str(tape)], capture_output=True, timeout=300, cwd=workdir)
    try:
        rows, cols = pathlib.Path(out).read_text().split()
        return int(cols), int(rows)
    except (FileNotFoundError, ValueError):
        print(f"  probe render failed for {cls_name} scale {scale}: {r.stderr.decode()[-200:]}")
        return None


def main() -> None:
    manifest = yaml.safe_load(open(sys.argv[1]))
    g = manifest["globals"]
    ss = g["supersample"]
    cal, failed = {}, []
    with tempfile.TemporaryDirectory() as td:
        wd = pathlib.Path(td)
        for cls_name, cls in manifest["classes"].items():
            target = render_probe(cls_name, cls, 1, cls["width"], cls["height"],
                                  g["font_family"], wd)
            if not target:
                failed.append(cls_name)
                continue
            tcols, trows = target
            w, h = cls["width"] * ss, cls["height"] * ss
            got = None
            for _ in range(MAX_ITER):
                got = render_probe(cls_name, cls, ss, w, h, g["font_family"], wd)
                if not got:
                    break
                cols, rows = got
                if (cols, rows) == (tcols, trows):
                    break
                # estimate cell size from this measurement and nudge
                pad = cls["padding"] * ss
                bar = 30 * ss if cls.get("windowbar") else 0
                cell_w = (w - 2 * pad) / cols
                cell_h = (h - 2 * pad - bar) / rows
                w += round((tcols - cols) * cell_w)
                h += round((trows - rows) * cell_h)
            if got == (tcols, trows):
                cal[cls_name] = {"width": w, "height": h, "cols": tcols, "rows": trows}
                print(f"{cls_name}: {w}x{h} -> {tcols}x{trows} (nominal {cls['width']*ss}x{cls['height']*ss})")
            else:
                failed.append(cls_name)
                print(f"{cls_name}: FAILED to converge (target {tcols}x{trows}, last {got})")
    json.dump(cal, open(sys.argv[2], "w"), indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
