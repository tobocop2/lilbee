#!/usr/bin/env python3
"""Generate per-class geometry probe tapes at 1x and 2x.

Each tape writes `stty size` to /tmp/geo-<class>-<scale>.txt inside the pty.
qualgate.sh compares the files: cols/rows MUST be identical between scales,
otherwise every proven choreography breaks (wrap points, drawer layout).

Usage: gen_geometry_probes.py reels.yaml <out-dir>
"""

import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def tape(cls_name: str, cls: dict, scale: int, font_family: str, framerate: int) -> str:
    if scale > 1:
        from retape import calibrated_dims
        width, height = calibrated_dims(cls_name, cls, scale)
    else:
        width, height = cls["width"], cls["height"]
    lines = [
        f'Set Shell "{cls["shell"]}"',
        f"Set Width {width}",
        f"Set Height {height}",
        f"Set FontSize {cls['fontsize'] * scale}",
        f'Set FontFamily "{font_family}"',
        f"Set LetterSpacing {scale}",
        f"Set Padding {cls['padding'] * scale}",
        f"Set Framerate {framerate}",
    ]
    if cls.get("windowbar"):
        lines += [f"Set WindowBar {cls['windowbar']}", f"Set WindowBarSize {30 * scale}"]
    lines += [
        f"Output geo-{cls_name}-{scale}x.mp4",
        "Hide",
        f'Type "stty size > /tmp/geo-{cls_name}-{scale}x.txt"',
        "Enter",
        "Sleep 2s",
        "Show",
        "Sleep 1s",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    manifest = yaml.safe_load(open(sys.argv[1]))
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    g = manifest["globals"]
    for cls_name, cls in manifest["classes"].items():
        for scale in (1, g["supersample"]):
            path = out / f"geo-{cls_name}-{scale}x.tape"
            path.write_text(tape(cls_name, cls, scale, g["font_family"], g["framerate"]))
    print(f"wrote {2 * len(manifest['classes'])} geometry probes to {out}")


if __name__ == "__main__":
    main()
