"""Write the NVIDIA CUDA notice of the llama.cpp packs: a header and the text of the EULA page.

Usage: llama_cuda_notice.py <notice-file>
"""

import html.parser
import sys
import urllib.request
from pathlib import Path

URL = "https://docs.nvidia.com/cuda/archive/12.4.1/eula/index.html"
HEAD = (
    "This wheel carries NVIDIA CUDA 12.4 runtime files: the CUDA Runtime (cudart) and the\n"
    "CUDA BLAS libraries (cublas, cublasLt). They are named in Attachment A of the NVIDIA\n"
    "CUDA Toolkit End User License Agreement"
    " (https://docs.nvidia.com/cuda/eula/index.html#attachment-a).\n"
    "The files are not modified. The EULA terms apply to these files. The GPU driver\n"
    "(libcuda, nvcuda) is not part of this wheel.\n\n"
    f"NVIDIA CUDA Toolkit EULA, version 12.4.1, from {URL}\n\n"
)
MIN_EULA_CHARS = 20000


class Text(html.parser.HTMLParser):
    """Collect the text of a page, with a line break at each block."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "br", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def main(notice: Path) -> None:
    page = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")  # noqa: S310
    parser = Text()
    parser.feed(page)
    lines = "".join(parser.parts).splitlines()
    body = "\n".join(line.strip() for line in lines if line.strip())
    if "Attachment A" not in body or len(body) < MIN_EULA_CHARS:
        raise SystemExit(f"the EULA text from {URL} looks incomplete")
    notice.write_text(HEAD + body + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
