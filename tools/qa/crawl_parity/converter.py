"""The converter layer: one long-lived converter process for each side, fed HTML, never a URL."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import TracebackType

from tools.qa.crawl_parity.corpus import Corpus, Response
from tools.qa.crawl_parity.model import Layer, Page
from tools.qa.crawl_parity.sides import DRIVERS_DIR, SideConfig, neutral_origin

DEFAULT_CHARSET = "utf-8"
_HEADER_CHARSET = re.compile(r"charset=([\w-]+)", re.IGNORECASE)
_META_CHARSET = re.compile(rb"<meta[^>]+charset=[\"']?([\w-]+)", re.IGNORECASE)
PROCESS_EXITED = "the converter process exited"


def declared_charset(response: Response) -> str:
    """The charset a response declares in its header, else in a meta tag, else UTF-8."""
    for name, value in response.headers:
        match = _HEADER_CHARSET.search(value) if name.lower() == "content-type" else None
        if match:
            return match.group(1)
    meta = _META_CHARSET.search(response.body)
    return meta.group(1).decode("ascii") if meta else DEFAULT_CHARSET


def decode(response: Response) -> str:
    """The body as text under its declared charset; an unknown charset reads as UTF-8."""
    try:
        return response.body.decode(declared_charset(response), errors="replace")
    except LookupError:
        return response.body.decode(DEFAULT_CHARSET, errors="replace")


class Converter:
    """A converter driver process that answers one conversion at a time."""

    def __init__(self, config: SideConfig, stderr_path: Path) -> None:
        self._command = [
            str(config.interpreters[Layer.CONVERTER]),
            str(DRIVERS_DIR / config.converter_driver),
        ]
        self._stderr_path = stderr_path
        self._process: subprocess.Popen[str] | None = None

    def __enter__(self) -> Converter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.wait()

    def _running(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.poll() is not None:
            self._stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_path.open("ab"),
                text=True,
                encoding="utf-8",
                cwd=self._stderr_path.parent,
            )
        return self._process

    def convert(self, url: str, html: str) -> Page:
        """The page the converter makes of *html*; a failure is the page's error."""
        process = self._running()
        assert process.stdin is not None  # opened as pipes above
        assert process.stdout is not None
        try:
            process.stdin.write(json.dumps({"html": html, "base_url": url}) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        except (BrokenPipeError, OSError):
            line = ""
        if not line:
            return Page(url, None, PROCESS_EXITED)
        answer = json.loads(line)
        if answer["markdown"] and answer["markdown"].strip():
            return Page(url, answer["markdown"])
        return Page(url, None, answer["error"] or "No content extracted")


def _neutral(page: Page, origin: str) -> Page:
    return Page(page.url, neutral_origin(page.markdown, origin), page.error)


def convert_corpus(config: SideConfig, corpus: Corpus, origin: str, work: Path) -> dict[str, Page]:
    """Every HTML page of *corpus* through one side's converter, by corpus path."""
    with Converter(config, work / "converter-stderr.txt") as converter:
        return {
            path: _neutral(converter.convert(origin + path, decode(record.response)), origin)
            for path, record in corpus.html_pages().items()
            if not any(name.lower() == "content-encoding" for name, _ in record.response.headers)
        }
