#!/usr/bin/env python3
"""Manual verification script for fine-grained download progress.

Downloads Qwen3 4B (~2.5 GB) via lilbee's own ``download_model`` and prints
a live progress line plus a summary of callback density. Used to verify the
HF_HUB_DISABLE_XET workaround for HF issue #4058 — under broken xet progress
the bar jumps 3-4 times, on the HTTP path it updates continuously.

Usage:
    uv run python demos/verify_qwen3_4b_progress.py

The model is downloaded to cfg.models_dir. Delete it first if you want a
fresh download:
    rm "$(uv run python -c 'from lilbee.config import cfg; print(cfg.models_dir)')"/*Qwen3-4B*
"""

from __future__ import annotations

import os
import sys
import time

# Importing lilbee sets HF_HUB_DISABLE_XET=1 before any HF submodule loads.
from lilbee.catalog import find_catalog_entry, download_model


def main() -> int:
    entry = find_catalog_entry("qwen3:4b")
    if entry is None:
        print("qwen3:4b not found in catalog", file=sys.stderr)
        return 1

    print(f"HF_HUB_DISABLE_XET = {os.environ.get('HF_HUB_DISABLE_XET')!r}")
    print(f"Downloading {entry.display_name} ({entry.hf_repo})")
    print(f"Expected size: ~{entry.size_gb} GB")
    print()

    calls: list[tuple[float, int, int]] = []
    start = time.monotonic()

    def on_progress(downloaded: int, total: int) -> None:
        now = time.monotonic() - start
        calls.append((now, downloaded, total))
        if total > 0:
            pct = downloaded / total * 100
            mb_done = downloaded / 1_000_000
            mb_total = total / 1_000_000
            bar_width = 40
            filled = int(bar_width * downloaded // total)
            bar = "=" * filled + "-" * (bar_width - filled)
            print(
                f"\r[{bar}] {pct:5.1f}%  {mb_done:,.1f} / {mb_total:,.1f} MB",
                end="",
                flush=True,
            )

    path = download_model(entry, on_progress=on_progress)
    elapsed = time.monotonic() - start
    print()

    print()
    print(f"Saved to: {path}")
    print(f"Elapsed:  {elapsed:.1f}s")
    print(f"Callbacks: {len(calls)}")

    # Diagnostic: under broken xet progress this is ~3-4. On the HTTP path
    # it is typically hundreds to low thousands for a 2.5 GB file.
    if len(calls) < 20:
        print()
        print("WARN: very few progress callbacks fired — xet path may still be active")
        print("      or the file was already cached.")
        return 2

    # Sanity: monotonic increase in downloaded bytes.
    prev = 0
    for _t, d, _total in calls:
        if d < prev:
            print(f"ERROR: non-monotonic progress: {prev} -> {d}")
            return 3
        prev = d

    final_d, final_t = calls[-1][1], calls[-1][2]
    if final_d != final_t:
        print(f"ERROR: final callback not 100%: {final_d} / {final_t}")
        return 4

    print("OK: fine-grained, monotonic, reached 100%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
