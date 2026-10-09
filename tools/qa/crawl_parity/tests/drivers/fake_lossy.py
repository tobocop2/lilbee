"""The stand-in crawler with two defects: it drops the word text and loses a page named two."""

from __future__ import annotations

import sys

from _fake import crawl

if __name__ == "__main__":
    sys.exit(crawl(drop_word="text", lose_path="/two"))
