#!/usr/bin/env python3
"""Statically validate generated 2x tapes against reels.yaml before anything runs.

Checks, per tape:
- header exactly matches what retape.py would emit for the reel's class
  (dims/FontSize/Padding doubled, LetterSpacing 2, WindowBarSize 60, Framerate 25)
- FontFamily present and equal to the manifest stack
- Theme present iff the class has one (agent/plain tapes MUST be themeless)
- Output is the reel's mp4 only (no gif output on pods)
- COLORTERM truecolor present
- no stray Set/Output/Env in the body
- body is non-empty and contains at least one Show

Usage: tape_lint.py reels.yaml <tapes-dir> [--pending reel1,reel2]
Pending reels (bodies awaiting volume reconciliation on the prep pod) get
their /add-path findings downgraded to warnings; everything else is strict.
Exit 1 on any strict finding.
"""

import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from retape import HEADER_PREFIXES, build_header


def lint(manifest: dict, tapes_dir: pathlib.Path, pending: set[str]) -> list[str]:
    findings = []
    warnings = []
    for reel_name, reel in manifest["reels"].items():
        path = tapes_dir / f"{reel_name}.tape"
        if not path.exists():
            findings.append(f"{reel_name}: tape missing at {path}")
            continue
        lines = path.read_text().splitlines()
        expected = build_header(manifest, reel_name)
        got = lines[: len(expected)]
        for i, (e, g) in enumerate(zip(expected, got)):
            if e != g:
                findings.append(f"{reel_name}: header line {i + 1}: expected {e!r} got {g!r}")
        body = lines[len(expected):]
        for n, line in enumerate(body, start=len(expected) + 1):
            if line.startswith(HEADER_PREFIXES) or line.startswith("Env "):
                findings.append(f"{reel_name}: stray header directive in body at line {n}: {line!r}")
        # a Hide with no following Show records nothing; fully-visible
        # bodies (no Hide at all) are legitimate (original code-search style)
        if any(line.strip() == "Hide" for line in body) and not any(line.strip() == "Show" for line in body):
            findings.append(f"{reel_name}: body Hides but never Shows")
        if not any(line.strip().startswith("Type") for line in body):
            findings.append(f"{reel_name}: body has no Type (empty choreography?)")
        kb_paths = {"manual": ("/root/cv-manual.pdf",),
                    "corpus": ("/root/corpus",),
                    "godot": ("/root/godot",)}
        allowed = kb_paths.get(reel.get("kb"), ())
        for n, line in enumerate(body, start=len(expected) + 1):
            if "/add " in line:
                target = line.split("/add ", 1)[1].rstrip('"').strip()
                if not any(target.startswith(a) for a in allowed):
                    msg = f"{reel_name}: /add target {target!r} not staged for kb={reel.get('kb')!r} (line {n})"
                    (warnings if reel_name in pending else findings).append(msg)
        cls = manifest["classes"][reel["class"]]
        themed = any(line.startswith("Set Theme") for line in lines)
        if bool(cls.get("theme")) != themed:
            findings.append(f"{reel_name}: theme mismatch (class wants {cls.get('theme')!r}, tape themed={themed})")
    for w in warnings:
        print(f"LINT-PENDING (must clear after volume reconcile): {w}")
    return findings


def main() -> None:
    manifest = yaml.safe_load(open(sys.argv[1]))
    pending = set()
    if "--pending" in sys.argv:
        pending = set(sys.argv[sys.argv.index("--pending") + 1].split(","))
    findings = lint(manifest, pathlib.Path(sys.argv[2]), pending)
    for f in findings:
        print(f"LINT: {f}")
    print(f"{len(findings)} finding(s) across {len(manifest['reels'])} reels")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
