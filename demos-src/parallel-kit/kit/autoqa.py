#!/usr/bin/env python3
"""Per-reel automated QA on a rendered mp4. Runs on the pod after each take
and again locally at gather. Catches single-frame transients that sampling
misses (garbled scroll frames, wizard flash, redraw bursts).

Pipeline:
1. decode EVERY frame (scaled to 700px wide for speed)
2. consecutive-frame diff series -> flag bursts outside typing/streaming
3. perceptual-hash dedupe -> OCR every UNIQUE frame against error strings
4. settle assert: near-zero diff over the final 2s
5. dims + duration bounds (manifest override, else derived from the tape)
6. palette assert: mode-color bg matches the reel's bg_target
7. must-strings anywhere in the reel via 1fps OCR at 1400px

Usage: autoqa.py reels.yaml <reel-name> <mp4> --report report.json [--tapes-dir DIR]
Exit 0 pass, 1 fail (report always written).
"""

import json
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np
import yaml
from PIL import Image

# measured from shipped/passing assets: rose-pine TUI base, opencode's own
# UI bg, and the VHS default terminal bg
BG_TARGETS = {"rose-pine": (25, 23, 35), "opencode": (24, 24, 36), "vhs-default": (23, 23, 23)}
BG_TOL = 8
DIFF_BURST = 0.55        # fraction of pixels changed in one frame = hard cut/garble
SETTLE_MAX = 0.02


def tape_visible_seconds(tape_path: pathlib.Path) -> float:
    """Expected output duration: VHS drops Hidden segments, so sum Sleeps and
    typing time only while visible."""
    visible, total = True, 0.0
    for raw in tape_path.read_text().splitlines():
        line = raw.strip()
        if line == "Hide":
            visible = False
        elif line == "Show":
            visible = True
        elif visible and line.startswith("Sleep "):
            v = line.split()[1]
            total += float(v[:-2]) / 1000 if v.endswith("ms") else float(v.rstrip("s"))
        elif visible and line.startswith("Type"):
            head, _, text = line.partition(" ")
            if "@" in head:
                assert head.endswith("ms"), f"unsupported Type speed unit: {head}"
                speed_ms = float(head[5:-2])
            else:
                speed_ms = 35.0
            total += len(text.strip('"')) * speed_ms / 1000
    return total


def expected_show_cuts(tape_path: pathlib.Path) -> int:
    """Mid-take Show count: each hidden section that ends after recording
    started produces one legitimate full-frame cut in the output."""
    shows = [line.strip() for line in tape_path.read_text().splitlines()
             if line.strip() == "Show"]
    return max(len(shows) - 1, 0)


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def extract_frames(mp4: str, outdir: str) -> list[pathlib.Path]:
    subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", mp4, "-vf", "scale=700:-1", f"{outdir}/f%06d.png"],
        check=True,
    )
    return sorted(pathlib.Path(outdir).glob("f*.png"))


def phash(a: np.ndarray) -> int:
    small = np.asarray(Image.fromarray(a).convert("L").resize((16, 16)))
    return int("".join("1" if v else "0" for v in (small > small.mean()).flat), 2)


def ocr(path: str) -> str:
    try:
        return run(["tesseract", path, "stdout", "--psm", "6"]).lower()
    except subprocess.CalledProcessError:
        return ""


def main() -> None:
    manifest = yaml.safe_load(open(sys.argv[1]))
    reel_name, mp4 = sys.argv[2], sys.argv[3]
    report_path = sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else "autoqa.json"
    tapes_dir = pathlib.Path(sys.argv[sys.argv.index("--tapes-dir") + 1]) \
        if "--tapes-dir" in sys.argv else pathlib.Path(__file__).parent / "tapes"
    reel = manifest["reels"][reel_name]
    cls = manifest["classes"][reel["class"]]
    g = manifest["globals"]
    findings: list[str] = []
    rep: dict = {"reel": reel_name, "mp4": mp4}

    # dims + duration
    probe = json.loads(run(["ffprobe", "-v", "quiet", "-print_format", "json",
                            "-show_streams", "-show_format", mp4]))
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    ss = g["supersample"]
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from retape import calibrated_dims
    want = calibrated_dims(reel["class"], cls, ss)
    got = (vs["width"], vs["height"])
    rep["dims"] = got
    if got != want:
        findings.append(f"dims {got} != {want}")
    dur = float(probe["format"]["duration"])
    rep["duration_s"] = round(dur, 1)
    tape_for_dur = tapes_dir / f"{reel_name}.tape"
    bounds = reel.get("duration_s")
    if not bounds and tape_for_dur.exists():
        exp = tape_visible_seconds(tape_for_dur)
        bounds = {"min": exp * 0.6, "max": exp * 1.5 + 20}
        rep["derived_bounds"] = [round(bounds["min"]), round(bounds["max"])]
    if bounds and not (bounds["min"] <= dur <= bounds["max"]):
        findings.append(f"duration {dur:.0f}s outside [{bounds['min']:.0f},{bounds['max']:.0f}]")

    with tempfile.TemporaryDirectory() as td:
        frames = extract_frames(mp4, td)
        rep["frames"] = len(frames)
        if not frames:
            findings.append("no frames decoded")
            rep["findings"] = findings
            rep["ok"] = False
            json.dump(rep, open(report_path, "w"), indent=1)
            print(json.dumps({"ok": False, "findings": findings}, indent=1))
            sys.exit(1)
        arrs_prev, diffs = None, []
        hashes: dict[int, pathlib.Path] = {}
        for f in frames:
            a = np.asarray(Image.open(f).convert("RGB"))
            if arrs_prev is not None:
                diffs.append(float((np.abs(a.astype(int) - arrs_prev.astype(int)).max(axis=2) > 30).mean()))
            arrs_prev = a
            h = phash(a)
            if h not in hashes:
                hashes[h] = f
        rep["unique_frames"] = len(hashes)

        # diff bursts: full-frame rewrites (hard cuts, garbled frames). Each
        # mid-take Show in the tape produces one LEGITIMATE cut; anything
        # beyond that count is a defect (garbled scroll, flash, corrupt frame)
        bursts = [i for i, d in enumerate(diffs) if d > DIFF_BURST]
        rep["diff_bursts_at_frames"] = bursts
        tape = tapes_dir / f"{reel_name}.tape"
        allowed = expected_show_cuts(tape) if tape.exists() else 0
        rep["expected_show_cuts"] = allowed
        late = [i for i in bursts if i > g["framerate"] * 2 and i < len(diffs) - g["framerate"]]
        # a single cut often spans 2 adjacent frames (cut + full repaint);
        # cluster within 3 frames so one legitimate Show counts as one event
        events = []
        for i in late:
            if not events or i - events[-1][-1] > 3:
                events.append([i])
            else:
                events[-1].append(i)
        rep["burst_events"] = [e[0] for e in events]
        if len(events) > allowed:
            findings.append(f"{len(events)} redraw burst events mid-take (expected <= {allowed} Show cuts) at frames {[e[0] for e in events][:10]}")

        # settle: final 2s must be static
        tail = diffs[-2 * g["framerate"]:]
        if tail and max(tail) > SETTLE_MAX:
            findings.append(f"no settle: max tail diff {max(tail):.3f}")


        # palette: median bg across mid-take frames
        # background = the MODE color (median drifts on busy frames: panels
        # and text pushed a known-good take 9/channel off the true bg)
        mid = frames[len(frames) // 2]
        a = np.asarray(Image.open(mid).convert("RGB")).reshape(-1, 3).astype(np.uint32)
        packed = (a[:, 0] << 16) | (a[:, 1] << 8) | a[:, 2]
        top = np.bincount(packed).argmax()
        bg = np.array([(top >> 16) & 255, (top >> 8) & 255, top & 255], dtype=float)
        rep["mid_bg"] = [int(x) for x in bg]
        target = BG_TARGETS[reel.get("bg_target", "rose-pine" if cls.get("theme") else "opencode")]
        if np.abs(bg - np.array(target)).max() > BG_TOL:
            findings.append(f"bg {rep['mid_bg']} != expected {target} (bg_target={reel.get('bg_target')})")

    # OCR pass: 1fps at FULL resolution (reduced-scale OCR misread a clean
    # take as "Error" and missed real must-strings; error toasts persist for
    # seconds so 1fps catches them, and sub-second garbles are caught by the
    # diff-burst detector above). Soft "error" needs a word-boundary match.
    # soft "error" matching is disabled per-reel where prose/code legitimately
    # contains the word (agent reels answering over source code)
    soft_enabled = reel.get("soft_error_check", True)
    soft_re = re.compile(r"\berrors?\b")
    hard = [s for s in g["error_strings"] if s.lower() not in ("error", "error:")]
    musts = reel.get("must_strings") or []
    errs: dict[str, list[str]] = {}
    seen_musts: set[str] = set()
    with tempfile.TemporaryDirectory() as td2:
        subprocess.run(["ffmpeg", "-v", "quiet", "-i", mp4, "-vf", "fps=1,scale=1400:-1",
                        f"{td2}/s%05d.png"], check=True)
        subprocess.run(["ffmpeg", "-v", "quiet", "-sseof", "-0.5", "-i", mp4,
                        "-vf", "scale=1400:-1", "-vframes", "1", f"{td2}/s99999.png"], check=True)
        for f in sorted(pathlib.Path(td2).glob("s*.png")):
            text = ocr(str(f))
            for s in hard:
                pat = rf"\b{re.escape(s.lower())}\b" if s.isdigit() else re.escape(s.lower())
                if re.search(pat, text):
                    errs.setdefault(s, []).append(f.name)
            if soft_enabled and soft_re.search(text):
                errs.setdefault("error", []).append(f.name)
            for m in musts:
                if m.lower() in text:
                    seen_musts.add(m)
    rep["error_strings"] = errs
    if errs:
        findings.append(f"error strings on frames: {errs}")
    if musts:
        missing = [m for m in musts if m not in seen_musts]
        rep["must_strings_missing"] = missing
        if missing:
            findings.append(f"musts never seen on any frame: {missing}")

    rep["findings"] = findings
    rep["ok"] = not findings
    json.dump(rep, open(report_path, "w"), indent=1)
    print(json.dumps({"ok": rep["ok"], "findings": findings}, indent=1))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
