#!/usr/bin/env python3
"""Pull the proven per-model boot/gen windows out of the rendered v1 tapes on
the volume (/workspace/reels/permodel-<key>.tape) and write them into
reels.yaml, replacing the conservative defaults. Runs on the prep pod.

Usage: reconcile_windows.py reels.yaml /workspace/reels
"""

import pathlib
import re
import sys

import yaml


def windows_from(tape: pathlib.Path) -> tuple[int, int] | None:
    sleeps = [int(m.group(1)) for m in re.finditer(r"^Sleep (\d+)s$", tape.read_text(), re.M)]
    big = [s for s in sleeps if s >= 20]
    if len(big) >= 2:
        return big[0], big[-1]  # first long sleep = boot, last = generation
    return None


def main() -> None:
    manifest_path, reels_dir = sys.argv[1], pathlib.Path(sys.argv[2])
    text = pathlib.Path(manifest_path).read_text()
    manifest = yaml.safe_load(text)
    changed = 0
    for name, reel in manifest["reels"].items():
        if not name.startswith("pm-"):
            continue
        key = reel["model"]
        tape = reels_dir / f"permodel-{key}.tape"
        if not tape.exists():
            print(f"no volume tape for {name} ({tape}), keeping defaults")
            continue
        w = windows_from(tape)
        if not w:
            continue
        new = f"windows: {{boot: {w[0]}, gen: {w[1]}}}"
        pat = re.compile(rf"({re.escape(name)}:.*?)windows: \{{boot: \d+, gen: \d+\}}")
        text2 = pat.sub(rf"\1{new}", text, count=1)
        if text2 != text:
            text, changed = text2, changed + 1
            print(f"{name}: {new}")
    pathlib.Path(manifest_path).write_text(text)
    yaml.safe_load(text)
    print(f"reconciled {changed} per-model windows")


if __name__ == "__main__":
    main()
