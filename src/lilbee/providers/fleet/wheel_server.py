"""The rule for the server file of an engine wheel; the standard library only.

The engine store and ``tools/qa/assert_engine_bundle.py`` both load this file,
so a wheel that one accepts is a wheel that the other accepts.
"""

from __future__ import annotations

import re
import stat
import zipfile
from pathlib import PurePosixPath

_BIN_FILE = re.compile(r"[^/]+/bin/[^/]+")
_GGML_FILE = re.compile(r"(lib)?ggml.*")
_GGML_BASE = re.compile(r"(lib)?ggml-base(\.[\d.]+)?\.(so|dylib|dll)(\.[\d.]+)?")


def server_entry(names: list[str], tool: str) -> str | None:
    """The first ``<package>/bin/`` entry of *names* named *tool* or ``<tool>.exe``."""
    found = sorted(
        n
        for n in names
        if _BIN_FILE.fullmatch(n) and PurePosixPath(n).name in (tool, f"{tool}.exe")
    )
    return found[0] if found else None


def server_problems(archive: zipfile.ZipFile, tool: str) -> list[str]:
    """What stops the server *tool* of *archive* from running; empty when nothing does.

    The server must be present, must carry the owner execute bit unless it is an
    ``.exe``, and a wheel that ships ggml libraries must ship ``ggml-base`` with them.
    """
    names = archive.namelist()
    server = server_entry(names, tool)
    if server is None:
        return [f"has no {tool} in its bin directory"]
    problems = []
    mode = archive.getinfo(server).external_attr >> 16
    if not server.endswith(".exe") and not mode & stat.S_IXUSR:
        problems.append(f"holds {server!r} without the execute bit")
    folder = server.rsplit("/", 1)[0]
    siblings = [
        PurePosixPath(n).name
        for n in names
        if n.startswith(f"{folder}/") and _BIN_FILE.fullmatch(n)
    ]
    if any(_GGML_FILE.fullmatch(n) for n in siblings) and not any(
        _GGML_BASE.fullmatch(n) for n in siblings
    ):
        problems.append("ships ggml libraries but no ggml-base library")
    return problems
