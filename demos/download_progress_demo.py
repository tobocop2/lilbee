#!/usr/bin/env python3
"""Demo script to visually verify download progress callbacks.

Run:
    uv run python demos/download_progress_demo.py

Downloads a tiny (~3 MB) model and prints each progress callback,
proving real-time progress updates work end-to-end.
"""

import sys
import time

from lilbee.catalog import CatalogModel, download_model
from lilbee.config import cfg

_TINY_MODEL = CatalogModel(
    name="tinystories",
    tag="260k",
    display_name="TinyStories 260K",
    hf_repo="karpathy/tinyllamas",
    gguf_filename="stories260K.gguf",
    size_gb=0.003,
    min_ram_gb=0.5,
    description="Tiny model for testing download progress",
    featured=False,
    downloads=0,
    task="chat",
)


def main() -> None:
    calls: list[tuple[int, int]] = []
    start = time.monotonic()

    def on_progress(downloaded: int, total: int) -> None:
        calls.append((downloaded, total))
        elapsed = time.monotonic() - start
        if total > 0:
            pct = min(downloaded * 100 / total, 100)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            bar_width = 40
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(
                f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB  ({elapsed:.1f}s)",
                end="",
                flush=True,
            )
        else:
            print(f"\r  Downloaded {downloaded} bytes ({elapsed:.1f}s)", end="", flush=True)

    print(f"Downloading {_TINY_MODEL.display_name} from {_TINY_MODEL.hf_repo}...")
    print(f"  Models dir: {cfg.models_dir}")
    print()

    try:
        path = download_model(_TINY_MODEL, on_progress=on_progress)
        print()
        print()
        print(f"  ✓ Downloaded to: {path}")
        print(f"  ✓ Total callbacks: {len(calls)}")
        if len(calls) >= 2:
            print("  ✓ Progress was incremental (multiple callbacks received)")
        else:
            print("  ⚠ Only 1 callback — progress may have jumped to 100%")
            print("    (This is expected if the file was already cached)")
    except Exception as exc:
        print()
        print(f"  ✗ Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
