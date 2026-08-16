#!/usr/bin/env python3
"""Copy finished reels into a gh-pages checkout. Only reels whose scorecard is clean.

A reel with a FAIL or an UNTESTED row does not ship. That is the whole point of the
scorecard, so the copy step enforces it rather than trusting whoever runs it to check.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

KIT = pathlib.Path(__file__).resolve().parent
OUT = KIT / "out"


def ready(name: str) -> tuple[bool, str]:
    score = OUT / f"{name}.score.txt"
    if not score.exists():
        return False, "no scorecard"
    text = score.read_text()
    bad = [line.strip() for line in text.splitlines()
           if line.strip().startswith(("[FAIL]", "[----]"))]
    return (not bad), (bad[0] if bad else "clean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", type=pathlib.Path, help="gh-pages checkout root")
    ap.add_argument("--force", action="store_true", help="copy despite a red scorecard")
    a = ap.parse_args()
    demos = a.dest / "demos"
    if not demos.is_dir():
        sys.exit(f"{demos} is not a gh-pages checkout")

    shipped, held = [], []
    for gif in sorted(OUT.glob("*.gif")):
        name = gif.stem
        if name.startswith("live-") or name.endswith(("-contact", "-full")):
            continue
        ok, why = ready(name)
        if not ok and not a.force:
            held.append((name, why))
            continue
        for ext in (".gif", ".mp4", ".png"):
            src = OUT / f"{name}{ext}"
            if src.exists():
                shutil.copy2(src, demos / src.name)
        shipped.append(name)

    print(f"shipped {len(shipped)}: {', '.join(shipped)}")
    for name, why in held:
        print(f"held    {name}: {why}")


if __name__ == "__main__":
    main()
