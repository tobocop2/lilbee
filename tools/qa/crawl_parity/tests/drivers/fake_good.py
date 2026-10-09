"""The stand-in crawler with no defect."""

from __future__ import annotations

import sys

from _fake import crawl

if __name__ == "__main__":
    sys.exit(crawl())
