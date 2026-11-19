"""Write the licence notices that a web page bundle carries, for a wheel whose server embeds it.

Usage: embedded_ui_notices.py <bundle> <output>

<bundle> is the HTML page of a web UI that the server build compiles in, plain or gzip-compressed.
The notices are the block comments of the page (and of the scripts that it holds in strings) that
name a licence. <output> receives them, and the standard MIT permission text when one of them says
MIT. The script stops when the bundle holds no notice, so a bundle that changed cannot ship none.
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

MAX_NOTICE_CHARS = 4000
COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
MARKER = re.compile(
    r"@license|@licstart|[Ll]icensed under|MIT [Ll]icense|Permission is hereby granted"
)
MIT_NAMED = re.compile(r"MIT [Ll]icense")
ESCAPES = {r"\n": "\n", r"\r": "", r"\t": "\t", r"\"": '"', r"\'": "'"}
MIT_TEXT = """\
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def read_page(path: Path) -> str:
    """The text of the page, decompressed when the file is gzip."""
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="replace")


def unescape(comment: str) -> str:
    """A comment that sits in a script string has its line breaks as backslash sequences."""
    if "\n" in comment:
        return comment
    for escaped, plain in ESCAPES.items():
        comment = comment.replace(escaped, plain)
    return comment


def notices(page: str) -> list[str]:
    """The distinct comments of the page that name a licence, in order of appearance."""
    found: dict[str, None] = {}
    for match in COMMENT.finditer(page):
        text = match[0]
        if len(text) <= MAX_NOTICE_CHARS and MARKER.search(text):
            found[unescape(text).strip()] = None
    return list(found)


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        raise SystemExit(__doc__)
    bundle, output = Path(argv[1]), Path(argv[2])
    found = notices(read_page(bundle))
    if not found:
        raise SystemExit(f"{bundle} holds no licence notice")
    text = "\n\n".join(found) + "\n"
    if MIT_NAMED.search(text):
        text += f"\nThe MIT License, the permission text of the notices above:\n\n{MIT_TEXT}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
