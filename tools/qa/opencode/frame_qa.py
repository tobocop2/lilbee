"""Automated frame-by-frame validation of a recorded opencode reel.

The matrix verdict (``scenarios.py``) grades the live tmux pane. The reel is a
separate VHS recording, and a reel can look dead on camera even when the model
answered: a final-frame screenshot passes while the video arc is a boot screen
with a blinking cursor (the failure mode that shipped a broken reel before).

This module closes that gap. ``reelrun.sh`` extracts the reel's unique frames as
``frames/t<sec>.png``; here we OCR each frame and assert, across the whole
timeline, the things a developer must actually see:

* the prompt was typed (the task is visible),
* ``lilbee_search`` was dispatched (the tool fired, not narrated),
* the expected answer / code landed in a late frame (it solved the problem),
* no frame shows an error (no "no models", "chat failed", 5xx, traceback, OOM),
* the reel is not a dead screen (enough distinct frames over its span).

A reel that fails any hard check is not publishable. Run as a CLI from
``reelrun.sh`` after frame extraction; exit code 0 means publishable.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_FRAME_GLOB = "t*.png"
_FRAME_SECONDS_RE = re.compile(r"t(\d+)\.png$")

# Error text that must not appear in ANY frame. opencode/lilbee surface these on
# a failed turn; one is enough to make the reel unpublishable.
_ERROR_MARKERS: tuple[str, ...] = (
    "no models installed",
    "no models",
    "chat failed",
    "internal error",
    "traceback (most recent call last)",
    "model_not_found",
    "connection refused",
    "out of memory",
    "cudamalloc",
    "context_length_exceeded",
    "provider error",
    "503 service",
    "500 internal",
    "rate limit",
)

# A real tool turn renders the tool name beside opencode's gear glyph (U+2699,
# which OCR drops); match the name itself, the reliable signal.
_DISPATCH_MARKER = "lilbee_search"

# Expected developer-visible answer per tier. The fixed-fact tiers assert the
# known correct token (the search returns it; a hallucinated answer omits it).
# The code tiers assert that GDScript was actually written on camera.
_PROMPT_MARKERS: dict[str, tuple[str, ...]] = {
    "small": ("astargrid2d", "get_id_path"),
    "mid": ("object.connect", "connect_deferred"),
    "coder": ("level_generator", "procedural level"),
    "giant": ("level_generator", "procedural level"),
}
_ANSWER_MARKERS: dict[str, tuple[str, ...]] = {
    # AStarGrid2D.get_id_path returns a PackedVector2Array (the grounded fact).
    "small": ("packedvector2array",),
    # CONNECT_DEFERRED == 1, CONNECT_ONE_SHOT == 4; the signature names a Callable.
    "mid": ("connect_deferred", "callable"),
    # The agent must write real GDScript, not narrate it.
    "coder": ("func ", "extends ", "set_cell"),
    "giant": ("func ", "extends ", "set_cell"),
}

# Dead-screen guard. Frames are extracted at 1fps then de-duplicated, so a live
# streaming reel keeps a new frame most seconds while a frozen boot reel collapses
# to a handful. We grade the kept-frame ratio (unique / span) rather than an
# absolute gap so a giant's legitimate silent prompt-eval doesn't read as dead.
_MIN_UNIQUE_FRAMES = 6
_MIN_UNIQUE_FRAME_RATIO = 0.2


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class FrameQAReport:
    family: str
    tier: str
    frames: int
    span_s: int
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def _ocr(frame: Path) -> str:
    """OCR one frame to lowercase text via the tesseract binary (no python dep).

    Returns ``""`` if tesseract is unavailable or errors on the frame, so a
    single unreadable frame degrades that frame's evidence rather than crashing
    the whole gate.
    """
    try:
        out = subprocess.run(
            ["tesseract", str(frame), "stdout", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.lower()


def _frame_seconds(frame: Path) -> int:
    match = _FRAME_SECONDS_RE.search(frame.name)
    return int(match.group(1)) if match else 0


def _load_frames(frames_dir: Path) -> list[tuple[int, Path]]:
    frames = sorted(frames_dir.glob(_FRAME_GLOB), key=_frame_seconds)
    return [(_frame_seconds(f), f) for f in frames]


def _any_marker(texts: list[str], markers: tuple[str, ...]) -> bool:
    return any(any(m in t for m in markers) for t in texts)


def _unique_frame_ratio(unique_frames: int, span_s: int) -> float:
    """Kept frames per extracted second (1fps extraction => span+1 extracted)."""
    return unique_frames / (span_s + 1)


def validate_texts(family: str, tier: str, frame_texts: list[tuple[int, str]]) -> FrameQAReport:
    """Apply the timeline checks to already-OCR'd frame text.

    Split from :func:`validate` so the assertion logic is unit-testable without
    tesseract or real PNGs.
    """
    texts = [t for _, t in frame_texts]
    seconds = [s for s, _ in frame_texts]
    span_s = max(seconds, default=0)

    prompt_markers = _PROMPT_MARKERS.get(tier, _PROMPT_MARKERS["small"])
    answer_markers = _ANSWER_MARKERS.get(tier, _ANSWER_MARKERS["small"])

    error_frames = [s for s, t in frame_texts if any(marker in t for marker in _ERROR_MARKERS)]
    ratio = _unique_frame_ratio(len(frame_texts), span_s)

    checks = [
        Check(
            "prompt_visible",
            _any_marker(texts, prompt_markers),
            f"a frame shows the task ({'/'.join(prompt_markers)})",
        ),
        Check(
            "dispatch_visible",
            _any_marker(texts, (_DISPATCH_MARKER,)),
            f"a frame shows the {_DISPATCH_MARKER} tool call",
        ),
        Check(
            "answer_visible",
            _any_marker(texts, answer_markers),
            f"a frame shows the expected answer/code ({'/'.join(answer_markers)})",
        ),
        Check(
            "no_error_frame",
            not error_frames,
            "no frame carries an error marker"
            if not error_frames
            else f"error marker(s) at t={error_frames}s",
        ),
        Check(
            "not_dead_screen",
            len(frame_texts) >= _MIN_UNIQUE_FRAMES and ratio >= _MIN_UNIQUE_FRAME_RATIO,
            f"{len(frame_texts)} distinct frames over {span_s}s "
            f"(ratio {ratio:.2f}, need >={_MIN_UNIQUE_FRAME_RATIO} and "
            f">={_MIN_UNIQUE_FRAMES} frames)",
        ),
    ]
    return FrameQAReport(
        family=family, tier=tier, frames=len(frame_texts), span_s=span_s, checks=checks
    )


def validate(family: str, tier: str, frames_dir: Path) -> FrameQAReport:
    frames = _load_frames(frames_dir)
    frame_texts = [(sec, _ocr(frame)) for sec, frame in frames]
    return validate_texts(family, tier, frame_texts)


def _write_artifacts(report: FrameQAReport, out_dir: Path) -> None:
    (out_dir / "frame_qa.json").write_text(json.dumps(asdict(report), indent=2))
    lines = [
        f"# Frame QA: {report.family} ({report.tier})",
        "",
        f"{report.frames} distinct frames over {report.span_s}s. "
        f"Verdict: {'PASS' if report.ok else 'FAIL'}",
        "",
        "| Check | Result | Detail |",
        "|-------|--------|--------|",
    ]
    for c in report.checks:
        lines.append(f"| {c.name} | {'PASS' if c.ok else 'FAIL'} | {c.detail} |")
    (out_dir / "frame_qa.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir", type=Path, help="dir of t<sec>.png unique frames")
    parser.add_argument("tier", help="small|mid|coder|giant")
    parser.add_argument("--family", default="model", help="family label for the report")
    args = parser.parse_args()

    report = validate(args.family, args.tier, args.frames_dir)
    _write_artifacts(report, args.frames_dir.parent)
    for c in report.checks:
        print(f"[frame_qa] {c.name}: {'PASS' if c.ok else 'FAIL'} - {c.detail}")
    if report.ok:
        print(f"[frame_qa] {args.family}: PASS ({report.frames} frames / {report.span_s}s)")
        return 0
    print(f"[frame_qa] {args.family}: FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
