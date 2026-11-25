"""Processes that OCR at once on an empty cache must each read the page."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

# Two processes lose the race now and then; eight lose it in most rounds,
# and three rounds make a miss rare.
_PROCESSES = 8
_ROUNDS = 3
_WORD = "tessdata"
_MIN_TRAINEDDATA_BYTES = 1_000_000


def _png(path: Path) -> None:
    image = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(image).text((10, 40), _WORD, fill="black")
    image.save(path, format="PNG")


def _child_env(home: Path, cache: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in {"TESSDATA_PREFIX", "XBERG_CACHE_DIR"}}
    env["HOME"] = str(home)
    env["XBERG_CACHE_DIR"] = str(cache)
    return env


def _race(root: Path) -> list[dict[str, str]]:
    """Start the processes together on an empty cache under *root*; return each one's report."""
    png = root / "race.png"
    _png(png)
    start = root / "start"
    env = _child_env(root / "home", root / "cache")
    children = [
        subprocess.Popen(
            [sys.executable, "-m", "tests._tessdata_child", str(png), str(start)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(_PROCESSES)
    ]
    while len(list(root.glob(f"{start.name}.*"))) < _PROCESSES:
        time.sleep(0.01)
    start.touch()
    reports = []
    for child in children:
        out, _ = child.communicate(timeout=300)
        assert child.returncode == 0, out
        reports.append(json.loads(out.strip().splitlines()[-1]))
    return reports


@pytest.mark.slow
@pytest.mark.parametrize("round_number", range(_ROUNDS))
def test_processes_that_ocr_at_once_on_an_empty_cache_each_read_the_page(tmp_path, round_number):
    reports = _race(tmp_path)
    assert [r["error"] for r in reports] == [None] * _PROCESSES
    assert all(_WORD in r["text"].lower() for r in reports), reports
    files = sorted((tmp_path / "cache" / "tessdata").iterdir())
    assert [f.name for f in files] == ["eng.traineddata"]
    assert files[0].stat().st_size > _MIN_TRAINEDDATA_BYTES
