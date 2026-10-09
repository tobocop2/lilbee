"""The stand-in converter with one defect: it drops the word text."""

from __future__ import annotations

from _fake import convert_forever

if __name__ == "__main__":
    convert_forever(drop_word="text")
