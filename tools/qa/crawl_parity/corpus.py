"""The records a replay server serves, and the loader of the synthetic corpus."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from tools.qa.crawl_parity.model import Mode

SYNTHETIC_DIR = Path(__file__).parent / "corpus" / "synthetic"
CORPUS_FILE = "corpus.toml"
ORIGIN_PLACEHOLDER = b"{{ORIGIN}}"
STATUS_OK = 200
# A table as tomllib returns it.
_Table = dict[str, Any]


class AlternateWhen(StrEnum):
    """The request conditions under which a record answers with its alternate response."""

    NO_COOKIE = "no-cookie"
    FIRST_REQUESTS = "first-requests"


@dataclass(frozen=True)
class Response:
    """One recorded answer: status, headers in order, and the body bytes as sent on the wire."""

    status: int = STATUS_OK
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes = b""


@dataclass(frozen=True)
class Alternate:
    """A second answer of a record and the condition that selects it."""

    when: AlternateWhen
    response: Response
    cookie: str = ""
    count: int = 0


@dataclass(frozen=True)
class Record:
    """Everything replay needs to answer one path of one host."""

    path: str
    response: Response
    delay_ms: int = 0
    substitute_origin: bool = False
    alternate: Alternate | None = None


@dataclass(frozen=True)
class Seed:
    """One crawl start: the path of an index page and the modes it is crawled in."""

    name: str
    path: str
    modes: tuple[Mode, ...]


@dataclass(frozen=True)
class Corpus:
    """The records of one site and its seeds."""

    name: str
    records: dict[str, Record]
    seeds: tuple[Seed, ...] = field(default=())

    def seed(self, name: str) -> Seed:
        """The seed called *name*."""
        return next(seed for seed in self.seeds if seed.name == name)

    def html_pages(self) -> dict[str, Record]:
        """The records that answer 200 with an HTML content type."""
        return {
            path: record
            for path, record in self.records.items()
            if record.response.status == STATUS_OK and _is_html(record.response)
        }


def _is_html(response: Response) -> bool:
    return any(
        name.lower() == "content-type" and "html" in value.lower()
        for name, value in response.headers
    )


def _response(table: _Table, root: Path) -> Response:
    """The response a TOML table describes; ``file`` is read as bytes from *root*."""
    headers = tuple((str(name), str(value)) for name, value in table.get("headers", []))
    file_name = table.get("file")
    body = (root / str(file_name)).read_bytes() if file_name else b""
    return Response(int(table.get("status", STATUS_OK)), headers, body)


def _alternate(table: _Table, root: Path) -> Alternate:
    return Alternate(
        when=AlternateWhen(str(table["when"])),
        response=_response(table, root),
        cookie=str(table.get("cookie", "")),
        count=int(table.get("count", 0)),
    )


def _record(table: _Table, root: Path) -> Record:
    alternate: _Table | None = table.get("alternate")
    return Record(
        path=str(table["path"]),
        response=_response(table, root),
        delay_ms=int(table.get("delay_ms", 0)),
        substitute_origin=bool(table.get("substitute_origin", False)),
        alternate=_alternate(alternate, root) if alternate is not None else None,
    )


def load_synthetic(root: Path = SYNTHETIC_DIR) -> Corpus:
    """The synthetic corpus under *root*."""
    with (root / CORPUS_FILE).open("rb") as handle:
        document = tomllib.load(handle)
    records = {str(table["path"]): _record(table, root) for table in document["record"]}
    seeds = tuple(
        Seed(str(seed["name"]), str(seed["path"]), tuple(Mode(mode) for mode in seed["modes"]))
        for seed in document["seed"]
    )
    return Corpus(root.name, records, seeds)
