"""The rule for the server file of an engine wheel; the standard library only.

The engine store and ``tools/qa/assert_engine_bundle.py`` both load this file,
so a wheel that one accepts is a wheel that the other accepts.
"""

from __future__ import annotations

import re
import stat
import zipfile
from collections.abc import Iterable
from pathlib import PurePosixPath

_BIN_FILE = re.compile(r"[^/]+/bin/[^/]+")
_LIBRARY_SUFFIX = r"(\.[\d.]+)?\.(so|dylib|dll)(\.[\d.]+)?"


def server_entry(names: list[str], tool: str) -> str | None:
    """The first ``<package>/bin/`` entry of *names* named *tool* or ``<tool>.exe``."""
    found = sorted(
        n
        for n in names
        if _BIN_FILE.fullmatch(n) and PurePosixPath(n).name in (tool, f"{tool}.exe")
    )
    return found[0] if found else None


def _holds_library(package: str, names: list[str], library: str) -> bool:
    """True when a file of the *package* folder is the shared library *library*."""
    pattern = re.compile(rf"(lib)?{re.escape(library)}{_LIBRARY_SUFFIX}")
    return any(
        n.startswith(f"{package}/") and pattern.fullmatch(PurePosixPath(n).name) for n in names
    )


def server_problems(
    archive: zipfile.ZipFile, tool: str, libraries: Iterable[str] = ()
) -> list[str]:
    """What stops the server *tool* of *archive* from running; empty when nothing does.

    The server must be present and carry the owner execute bit unless it is an ``.exe``.
    Each of *libraries* must be a shared library somewhere in the package of the server.
    """
    names = archive.namelist()
    server = server_entry(names, tool)
    if server is None:
        return [f"has no {tool} in its bin directory"]
    problems = []
    mode = archive.getinfo(server).external_attr >> 16
    if not server.endswith(".exe") and not mode & stat.S_IXUSR:
        problems.append(f"holds {server!r} without the execute bit")
    package = server.split("/", 1)[0]
    problems.extend(
        f"has no library file for {library!r} in its {package} folder"
        for library in libraries
        if not _holds_library(package, names, library)
    )
    return problems
