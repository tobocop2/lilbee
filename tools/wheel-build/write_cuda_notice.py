"""Write NOTICE-NVIDIA-CUDA with the text of the NVIDIA CUDA Toolkit EULA.

Usage: write_cuda_notice.py <output> <cuda-version> <cargo-build-log | sha256-list>

<cuda-version> is a full toolkit version such as 12.9.1; the EULA text is read from the archive
page of that version. The last argument is the output of `cargo build -vv`: the notice names the
NVIDIA libraries that the log shows linked. It can also be the sha256sum list of the shared
libraries that a wheel carries unchanged (such as libcudart.so.12): the notice names those. The
script stops when Attachment A of the EULA does not list one of them (cuDNN, for example).
"""

from __future__ import annotations

import html.parser
import re
import sys
import urllib.request

MIN_EULA_CHARS = 20000
LINK_LINE = re.compile(r"cargo::?rustc-link-lib=(?:(static|dylib)(?::[^=]*)?=)?([\w.+-]+)")
FILE_NAME = re.compile(r"\b(lib[\w.+-]*?\.(?:so|a))\b")
NVIDIA_LIB = re.compile(
    r"cud|cublas|curand|cufft|cusparse|cusolver|culibos|nvrtc|nvptx|nvJit|nvvm|nccl|npp"
)
SHIPPED_LINE = re.compile(r"[0-9a-f]{64} \*?(lib[\w+-]+)\.so(?:\.\d+)*")
DRIVER_LIBS = frozenset({"cuda"})


class Text(html.parser.HTMLParser):
    """The visible text of an HTML page, one block element for each line."""

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


def eula_text(version: str) -> tuple[str, str]:
    """The URL and the plain text of the EULA of one CUDA Toolkit version."""
    url = f"https://docs.nvidia.com/cuda/archive/{version}/eula/index.html"
    page = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    parser = Text()
    parser.feed(page)
    body = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    if "Attachment A" not in body or len(body) < MIN_EULA_CHARS:
        raise SystemExit(f"the EULA text from {url} looks incomplete")
    return url, body


def attachment_a_files(body: str) -> set[str]:
    """The file names that Attachment A of the EULA text allows an application to carry."""
    start = body.rfind("The following CUDA Toolkit files may be distributed")
    if start < 0:
        raise SystemExit("the EULA text has no Attachment A file list")
    end = body.find("Attachment B", start)
    section = body[start : end if end > start else len(body)]
    return set(FILE_NAME.findall(section))


def linked_nvidia_files(build_log: str) -> list[str]:
    """The NVIDIA library files that the build log shows cargo linking, as file names."""
    files: set[str] = set()
    for kind, name in LINK_LINE.findall(build_log):
        if name in DRIVER_LIBS or not NVIDIA_LIB.search(name):
            continue
        files.add(f"lib{name}.a" if kind == "static" else f"lib{name}.so")
    return sorted(files)


def shipped_nvidia_files(sha_list: str) -> list[str]:
    """The NVIDIA shared libraries of a sha256sum list, as the .so names Attachment A uses."""
    files: set[str] = set()
    for line in sha_list.splitlines():
        match = SHIPPED_LINE.fullmatch(line.strip())
        if match and NVIDIA_LIB.search(match[1]):
            files.add(f"{match[1]}.so")
    return sorted(files)


def disallowed(files: list[str], allowed: set[str]) -> list[str]:
    """The files in *files* that Attachment A does not list."""
    return [name for name in files if name not in allowed]


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        raise SystemExit(__doc__)
    output, version, log_path = argv[1:]
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    shipped = shipped_nvidia_files(text)
    files = shipped or linked_nvidia_files(text)
    if not files:
        raise SystemExit(f"{log_path} shows no NVIDIA library linked")
    url, body = eula_text(version)
    refused = disallowed(files, attachment_a_files(body))
    if refused:
        raise SystemExit(f"Attachment A of the EULA does not list: {', '.join(refused)}")
    if shipped:
        description = (
            f"This wheel carries NVIDIA CUDA {version} shared libraries without change: "
            f"{', '.join(files)}."
        )
    else:
        description = (
            f"This wheel links NVIDIA CUDA {version} libraries into the router as static or "
            f"shared libraries: {', '.join(files)}."
        )
    if "libcublas_static.a" in files:
        description += " The cublas library keeps the GPU code of this build's architecture only."
    head = (
        f"{description}\n"
        "They are named in Attachment A of the NVIDIA CUDA Toolkit End User License Agreement\n"
        "(https://docs.nvidia.com/cuda/eula/index.html#attachment-a). The EULA terms apply to\n"
        "these files. The GPU driver (libcuda, nvcuda) is not part of this wheel.\n\n"
        f"NVIDIA CUDA Toolkit EULA, version {version}, from {url}\n\n"
    )
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(head + body + "\n")


if __name__ == "__main__":
    main(sys.argv)
