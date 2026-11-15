"""Write the NVIDIA CUDA notice that a wheel with NVIDIA runtime files carries.

Usage: write_cuda_notice.py <CUDA toolkit version> <output file> <files that the wheel carries>

The notice states which NVIDIA files the wheel holds and repeats the End User License Agreement
text of that toolkit version, which the NVIDIA documentation site publishes.
"""

from __future__ import annotations

import html.parser
import sys
import urllib.request

_BREAKS = ("p", "div", "li", "h1", "h2", "h3", "h4", "br", "tr")
_MIN_BODY = 20000


class _Text(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in _BREAKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def eula_url(version: str) -> str:
    return f"https://docs.nvidia.com/cuda/archive/{version}/eula/index.html"


def eula_text(page: str, url: str) -> str:
    """The plain text of the EULA page; stops when the page looks cut short."""
    parser = _Text()
    parser.feed(page)
    body = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    if "Attachment A" not in body or len(body) < _MIN_BODY:
        raise SystemExit(f"the EULA text from {url} looks incomplete")
    return body


def notice(version: str, files: str, body: str, url: str) -> str:
    short = ".".join(version.split(".")[:2])
    head = (
        f"This wheel carries NVIDIA CUDA {short} runtime files: {files}\n"
        "They are named in Attachment A of the NVIDIA CUDA Toolkit End User License Agreement\n"
        "(https://docs.nvidia.com/cuda/eula/index.html#attachment-a).\n"
        "The files are not modified. The EULA terms apply to these files. The GPU driver\n"
        "(libcuda, nvcuda) is not part of this wheel.\n\n"
        f"NVIDIA CUDA Toolkit EULA, version {version}, from {url}\n\n"
    )
    return head + body + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    version, out, files = argv[1:]
    url = eula_url(version)
    page = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(notice(version, files, eula_text(page, url), url))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
