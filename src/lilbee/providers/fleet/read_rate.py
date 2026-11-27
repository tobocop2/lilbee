"""How fast the volume that holds a model reads, from a short timed read of the model."""

from __future__ import annotations

import functools
import time
from pathlib import Path

# The sample stops at this many bytes or after this many seconds, whichever comes
# first, so a plan on a slow volume pays at most the budget once per file.
_SAMPLE_BYTES = 64 * 1024 * 1024
_SAMPLE_BUDGET_S = 1.0
_CHUNK_BYTES = 1024 * 1024


@functools.lru_cache(maxsize=64)
def measured_read_rate(model_path: Path) -> int:
    """Bytes per second that a short read of *model_path* takes; 0 when it is not measured.

    A model folder is measured on its largest file. A path that does not open, and a
    file that ends before the sample does, give 0: a file read whole inside the
    budget loads well within the shortest cold-load wait.
    """
    sample = _sample_file(model_path)
    if sample is None:
        return 0
    chunk = bytearray(_CHUNK_BYTES)
    done = 0
    try:
        with sample.open("rb", buffering=0) as handle:
            start = time.monotonic()
            while True:
                read = handle.readinto(chunk)
                if not read:
                    return 0
                done += read
                elapsed = time.monotonic() - start
                if done >= _SAMPLE_BYTES or elapsed >= _SAMPLE_BUDGET_S:
                    return int(done / elapsed) if elapsed > 0 else 0
    except OSError:
        return 0


def _sample_file(model_path: Path) -> Path | None:
    """The file to time: *model_path* itself, or the largest file in a model folder."""
    if not model_path.is_dir():
        return model_path
    files = [path for path in model_path.rglob("*") if path.is_file()]
    return max(files, key=lambda path: path.stat().st_size, default=None)
