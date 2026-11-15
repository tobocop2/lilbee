"""Write the NVIDIA CUDA notice of a pack: a short statement followed by the EULA text.

Usage: write_cuda_notice.py OUTPUT

The text comes from the CUDA 12.4.1 EULA page. The script fails when the page looks incomplete,
so a changed page never ships a notice with a missing EULA.
"""

from __future__ import annotations

import html.parser
import sys
import urllib.request

URL = "https://docs.nvidia.com/cuda/archive/12.4.1/eula/index.html"
MIN_BODY_CHARS = 20000
HEAD = (
    "This wheel carries NVIDIA CUDA 12.4 runtime files: the CUDA Runtime (cudart) and the\n"
    "CUDA BLAS libraries (cublas, cublasLt). They are named in Attachment A of the NVIDIA\n"
    "CUDA Toolkit End User License Agreement (https://docs.nvidia.com/cuda/eula/index.html#attachment-a).\n"
    "The files are not modified. The EULA terms apply to these files. The GPU driver\n"
    "(libcuda, nvcuda) is not part of this wheel.\n\n"
    "NVIDIA CUDA Toolkit EULA, version 12.4.1, from " + URL + "\n\n"
)
BLOCK_TAGS = ("p", "div", "li", "h1", "h2", "h3", "h4", "br", "tr")


class Text(html.parser.HTMLParser):
    """The visible text of a page, one line for each block element."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def main(output: str) -> None:
    page = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
    parser = Text()
    parser.feed(page)
    body = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    if "Attachment A" not in body or len(body) < MIN_BODY_CHARS:
        raise SystemExit("the EULA text from " + URL + " looks incomplete")
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(HEAD + body + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
