#!/usr/bin/env python3
"""Emit a 2x-supersampled tape from a proven source tape + manifest class.

Header is generated entirely from reels.yaml (never trusted from the source);
the body (choreography) is passed through verbatim. This is how the proven
v8 bodies on the volume become 2x tapes without hand-copy errors.

Usage: retape.py reels.yaml <reel-name> <source-tape-path> <out-tape-path>
"""

import json
import pathlib
import sys

import yaml

HEADER_PREFIXES = ("Set ", "Output ", "Require ")


def calibrated_dims(cls_name: str, cls: dict, ss: int) -> tuple[int, int]:
    """2x dims from geometry_cal.json (solved so cols/rows == 1x exactly);
    nominal doubling until calibration exists."""
    cal_path = pathlib.Path(__file__).parent / "geometry_cal.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        if cls_name in cal:
            w, h = cal[cls_name]["width"], cal[cls_name]["height"]
        else:
            w, h = cls["width"] * ss, cls["height"] * ss
    else:
        w, h = cls["width"] * ss, cls["height"] * ss
    # H.264 requires EVEN width/height or ffmpeg emits "Conversion failed!" and
    # a 0-byte mp4 (hit the drawer=3839 and mcp=2893 classes). A 1px trim never
    # drops a column/row (cells are >=16px) so the geometry gate still holds.
    return w - (w % 2), h - (h % 2)


def build_header(manifest: dict, reel_name: str) -> list[str]:
    g = manifest["globals"]
    reel = manifest["reels"][reel_name]
    cls = manifest["classes"][reel["class"]]
    ss = g["supersample"]
    width, height = calibrated_dims(reel["class"], cls, ss)
    lines = [
        f'Set Shell "{cls["shell"]}"',
        f"Set Width {width}",
        f"Set Height {height}",
        f"Set FontSize {cls['fontsize'] * ss}",
        f'Set FontFamily "{g["font_family"]}"',
        f"Set LetterSpacing {ss}",  # VHS letter spacing is px and does not scale with FontSize
        f"Set Padding {cls['padding'] * ss}",
        'Set TypingSpeed 35ms',
        f"Set Framerate {g['framerate']}",
        "Set PlaybackSpeed 1",
    ]
    if cls.get("theme"):
        lines.append(f'Set Theme "{cls["theme"]}"')
    if cls.get("windowbar"):
        lines.append(f"Set WindowBar {cls['windowbar']}")
        lines.append(f"Set WindowBarSize {30 * ss}")
    lines += [
        'Env COLORTERM "truecolor"',
        'Env TERM "xterm-256color"',
        f"Output {reel_name}.mp4",  # mp4 only; gifs+posters are made locally at gather
    ]
    return lines


def body_lines(source_path: str, reel_name: str, reel: dict) -> list[str]:
    windows = reel.get("windows") or {}
    subs = {
        "__REEL__": reel_name,
        "__BOOT__": str(windows.get("boot", 85)),
        "__GEN__": str(windows.get("gen", 60)),
        # pod-private path map: volume-era bodies -> local-NVMe layout
        "source /root/k5_env.sh": "source /root/kit/env.sh",
        "source /workspace/kit/k5_env.sh": "source /root/kit/env.sh",
        "source /workspace/kit/take_env.sh": "source /root/kit/env.sh",
        "/workspace/cv-manual.pdf": "/root/cv-manual.pdf",
        "/workspace/corpus": "/root/corpus",
        "/workspace/godot/doc/classes": "/root/godot/doc/classes",
        "cd /workspace/godot": "cd /root/godot",
        # agent reels run opencode inside the checkout; pod-local copy so two
        # groups never share opencode project state on the volume
        "cd /workspace/lilbee": "cd /root/lilbee",
    }
    out = []
    for line in open(source_path):
        stripped = line.rstrip("\n")
        if stripped.startswith(HEADER_PREFIXES):
            continue
        if stripped.startswith("Env "):
            continue  # header owns env
        for k, v in subs.items():
            stripped = stripped.replace(k, v)
        out.append(stripped)
    while out and not out[0].strip():
        out.pop(0)
    residue = [l for l in out if "/workspace/" in l or "k5_env" in l]
    if residue:
        sys.exit(f"retape: unmapped volume-era references in {source_path}: {residue[:3]}")
    return out


def main() -> None:
    manifest_path, reel_name, source, dest = sys.argv[1:5]
    manifest = yaml.safe_load(open(manifest_path))
    if reel_name not in manifest["reels"]:
        sys.exit(f"unknown reel {reel_name}")
    reel = manifest["reels"][reel_name]
    lines = build_header(manifest, reel_name) + [""] + body_lines(source, reel_name, reel)
    with open(dest, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
