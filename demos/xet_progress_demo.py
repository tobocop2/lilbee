#!/usr/bin/env python3
"""Demo: real-time download progress using a custom tqdm_class bridge.

Shows fine-grained progress for xet-backed downloads without any visible
tqdm bar. Instead, updates are forwarded to a plain callback function.

Usage:
    python demos/xet_progress_demo.py
"""

import sys
import time

from tqdm.auto import tqdm

from huggingface_hub import hf_hub_download


class CallbackProgressBar(tqdm):
    """A tqdm subclass that forwards updates to a plain callback.

    This bridges huggingface_hub's tqdm-based progress into any custom
    progress system (TUI widgets, websockets, logging, etc.) without
    depending on tqdm's terminal rendering.
    """

    # Set by the caller before passing as tqdm_class
    _callback = None

    def __init__(self, *args, **kwargs):
        # Strip HF-internal kwargs that vanilla tqdm doesn't understand
        kwargs.pop("name", None)
        # Never render a visible bar
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
        self._cumulative = 0

    def update(self, n=1):
        # disable=True makes super().update() a no-op, so track manually
        self._cumulative += n
        if self._callback is not None:
            self._callback(int(self._cumulative), self.total)


def make_progress_class(callback):
    """Create a tqdm_class that forwards progress to the given callback.

    Args:
        callback: callable(downloaded_bytes: int, total_bytes: int | None)
    """

    class _Cls(CallbackProgressBar):
        _callback = staticmethod(callback)

    return _Cls


# ---- demo usage ----


def my_progress(downloaded: int, total: int | None) -> None:
    """Example callback that prints a simple progress line."""
    if total and total > 0:
        pct = downloaded / total * 100
        mb_done = downloaded / 1_000_000
        mb_total = total / 1_000_000
        bar_width = 40
        filled = int(bar_width * downloaded // total)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\r[{bar}] {pct:5.1f}%  {mb_done:,.1f} / {mb_total:,.1f} MB", end="", flush=True)
    else:
        mb_done = downloaded / 1_000_000
        print(f"\r  {mb_done:,.1f} MB downloaded...", end="", flush=True)


def main():
    # Use a known xet-backed repo with a reasonably sized file
    repo_id = "unsloth/gemma-3-1b-it-GGUF"
    filename = "gemma-3-1b-it-Q4_K_M.gguf"

    print(f"Downloading {repo_id}/{filename}")
    print(f"Using custom callback (no tqdm bar)\n")

    start = time.time()
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        tqdm_class=make_progress_class(my_progress),
        force_download=True,
    )
    elapsed = time.time() - start

    print()  # newline after progress
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
