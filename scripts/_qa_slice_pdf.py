"""Slice a PDF to a page range. Companion helper for qa-wiki-rerank.sh."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input PDF")
    parser.add_argument("output", type=Path, help="Output sliced PDF")
    parser.add_argument(
        "page_range",
        help="1-based page range, inclusive (e.g. '1-5' keeps pages 1 through 5)",
    )
    args = parser.parse_args()

    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print(
            "pypdf is required (ships transitively via kreuzberg).",
            file=sys.stderr,
        )
        return 2

    start, end = _parse_range(args.page_range)
    reader = PdfReader(str(args.source))
    if start < 1 or end > len(reader.pages):
        print(
            f"Page range {args.page_range} is out of bounds for a "
            f"{len(reader.pages)}-page document.",
            file=sys.stderr,
        )
        return 2
    writer = PdfWriter()
    for i in range(start - 1, end):
        writer.add_page(reader.pages[i])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        writer.write(fh)
    print(f"Wrote {end - start + 1} pages to {args.output}")
    return 0


def _parse_range(raw: str) -> tuple[int, int]:
    if "-" in raw:
        start_s, end_s = raw.split("-", 1)
        return int(start_s), int(end_s)
    page = int(raw)
    return page, page


if __name__ == "__main__":
    sys.exit(main())
