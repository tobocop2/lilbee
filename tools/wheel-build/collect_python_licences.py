#!/usr/bin/env python3
"""Copy the licence texts of a frozen Python tree and name its terms.

Usage: collect_python_licences.py <site-packages> <out> <terms> [<supplied>]

Each ``*.dist-info`` of <site-packages> is read. Its licence files (the ``licenses`` folder and the
licence, copying and notice files at the top of the dist-info) are copied to <out>/<package>/. A
package whose wheel ships no licence file takes the file that <supplied> names for it: one line per
package, ``<name>-<version> <path>``, with the path relative to the folder of <supplied>. A line
``<name>-<version> <path> replaces <file>`` writes the supplied text over <file> of the dist-info,
in the tree as well as in the copy, for a wheel whose own file covers libraries it does not hold. The
SPDX terms of every package are written to <terms>, one for each line. The script stops when a
package names a GPL-family licence, names no licence that it can read, ships no licence file, or
holds a GNU licence heading. A licence file also stops it when its text anywhere holds the GNU
licence title that the engine store refuses, so a tree the store would refuse never becomes a
wheel. It also stops when the bundled interpreter licence, ``LICENSE.txt`` in
the parent of <site-packages>, is missing or is not the PSF licence. The term ``PSF-2.0`` joins the
terms for the interpreter. The script does not judge permissive against weak copyleft terms such as
MPL-2.0: the terms file states them, and the store reads the wheel metadata.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

COPYLEFT = re.compile(r"\b[AL]?GPL|GNU (Affero |Lesser |Library )?General Public", re.IGNORECASE)
GNU_HEADING = re.compile(r"GNU (AFFERO |LESSER |LIBRARY )?GENERAL PUBLIC LICENSE")
# The title test of the engine store (engine_store._COPYLEFT_TITLE), on the text with its spacing
# collapsed and its case folded.
STORE_TITLE = re.compile(r"gnu (?:(?:affero|lesser|library) )?general public license version")
PSF_TITLE = re.compile(r"PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2")
LICENCE_FILE = re.compile(r"(licen[cs]e|copying|notice)", re.IGNORECASE)
SHORT_FIELD = 200
HEADING_BYTES = 2000
INTERPRETER = "LICENSE.txt"
JOINER = re.compile(r"\s+(?:AND|OR)\s+")
# Names that packages write in place of an SPDX identifier, in lower case.
NAMES = {
    "mit license": "MIT",
    "apache software license": "Apache-2.0",
    "apache 2.0 license": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "3-clause bsd license": "BSD-3-Clause",
    "2-clause bsd license": "BSD-2-Clause",
    "isc license": "ISC",
    "isc license (iscl)": "ISC",
}
BSD_CLASSIFIER = "bsd license"


def _fields(metadata: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for line in metadata.split("\n\n", 1)[0].splitlines():
        field, _, value = line.partition(": ")
        found.setdefault(field, []).append(value.strip())
    return found


def _bsd_term(texts: list[str]) -> str:
    """BSD-3-Clause when a licence text holds the clause that bars the use of the author's name."""
    return "BSD-3-Clause" if any("neither the name" in t.lower() for t in texts) else "BSD-2-Clause"


def _terms(fields: dict[str, list[str]], texts: list[str]) -> list[str]:
    """The SPDX terms that a package declares; empty when none can be read."""
    if expression := next(iter(fields.get("License-Expression", [])), ""):
        return [t.strip("() ") for t in JOINER.split(expression) if t.strip("() ")]
    short = next(iter(fields.get("License", [])), "")
    if short and len(short) <= SHORT_FIELD:
        if short.lower() in NAMES:
            return [NAMES[short.lower()]]
        if re.fullmatch(r"[A-Za-z0-9.+-]+( (AND|OR) [A-Za-z0-9.+-]+)*", short):
            return JOINER.split(short)
    found = []
    for classifier in fields.get("Classifier", []):
        if not classifier.startswith("License ::"):
            continue
        name = classifier.rsplit("::", 1)[-1].strip().lower()
        found.append(_bsd_term(texts) if name == BSD_CLASSIFIER else NAMES.get(name, ""))
    return [t for t in found if t]


def _licence_files(dist: Path) -> list[Path]:
    top = [p for p in dist.iterdir() if p.is_file() and LICENCE_FILE.search(p.name)]
    folder = [p for p in (dist / "licenses").rglob("*") if p.is_file()]
    return sorted(top + folder)


def _supplied(path: Path | None) -> dict[str, tuple[Path, str | None]]:
    """The licence file that <supplied> names for each package, and the file it replaces, if any."""
    if path is None:
        return {}
    found: dict[str, tuple[Path, str | None]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match line.split():
            case [package, text]:
                found[package] = (path.parent / text, None)
            case [package, text, "replaces", replaced]:
                found[package] = (path.parent / text, replaced)
            case _:
                raise SystemExit(f"collect_python_licences: cannot read the supplied line {line!r}")
    return found


def _store_refuses(path: Path) -> bool:
    """True when the text of *path* holds the GNU licence title anywhere, as the store reads it."""
    text = path.read_bytes().decode("utf-8", errors="replace")
    return STORE_TITLE.search(" ".join(text.split()).casefold()) is not None


def _target(dist: Path, out: Path, name: str, path: Path) -> Path:
    if path.is_relative_to(dist / "licenses"):
        return out / name / path.relative_to(dist / "licenses")
    if path.is_relative_to(dist):
        return out / name / path.relative_to(dist)
    return out / name / path.name


def _problems(
    dist: Path, out: Path, terms: set[str], supplied: dict[str, tuple[Path, str | None]]
) -> list[str]:
    name = dist.name.removesuffix(".dist-info")
    metadata = dist / "METADATA"
    fields = _fields(metadata.read_text(encoding="utf-8", errors="replace")) if metadata.is_file() else {}
    files = _licence_files(dist)
    extra, replaced = supplied.pop(name, (None, None))
    copies = {path: _target(dist, out, name, path) for path in files}
    if extra is not None:
        if not extra.is_file():
            return [f"{name}: the supplied licence file {extra} does not exist"]
        if replaced is None and files:
            return [f"{name}: ships its own licence file, so the supplied one is not needed"]
        if replaced is not None:
            if dist / replaced not in copies:
                return [f"{name}: {replaced} is not a licence file of its dist-info"]
            copies[extra] = copies.pop(dist / replaced)
        else:
            copies[extra] = _target(dist, out, name, extra)
        files = sorted(copies)
    texts = [p.read_bytes()[:HEADING_BYTES].decode("utf-8", errors="replace") for p in files]
    found = _terms(fields, texts)
    problems = []
    declared = [v for k in ("License-Expression", "License", "Classifier") for v in fields.get(k, [])]
    problems += [f"{name}: {v} is copyleft" for v in declared if COPYLEFT.search(v) and len(v) <= SHORT_FIELD]
    if not found:
        problems.append(f"{name}: names no licence that this script reads")
    if not files:
        problems.append(f"{name}: ships no licence file")
    problems += [
        f"{name}: {path.name} holds a GNU licence text"
        for path, text in zip(files, texts, strict=True)
        if GNU_HEADING.search(text) or _store_refuses(path)
    ]
    if not problems:
        terms.update(found)
        if extra is not None and replaced is not None:
            shutil.copyfile(extra, dist / replaced)
        for path in files:
            target = copies[path]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return problems


def _interpreter_problems(site: Path) -> list[str]:
    path = site.parent / INTERPRETER
    if not path.is_file():
        return [f"the interpreter ships no {INTERPRETER} beside {site.name}"]
    text = path.read_text(encoding="utf-8", errors="replace")
    if GNU_HEADING.search(text):
        return [f"interpreter: {INTERPRETER} holds a GNU licence text"]
    if not PSF_TITLE.search(text.upper()):
        return [f"interpreter: {INTERPRETER} is not the PSF licence"]
    return []


def collect(site: Path, out: Path, supplied: Path | None = None) -> tuple[set[str], list[str]]:
    """The terms of the tree and the problems found; the copies are written only for clean packages."""
    dists = sorted(site.glob("*.dist-info"))
    if not dists:
        return set(), [f"{site} holds no package"]
    terms: set[str] = set()
    named = _supplied(supplied)
    problems = [p for dist in dists for p in _problems(dist, out, terms, named)]
    problems += [f"{package}: a licence file is supplied for a package the tree does not hold" for package in named]
    interpreter = _interpreter_problems(site)
    if not interpreter:
        terms.add("PSF-2.0")
    return terms, problems + interpreter


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    names, found = collect(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[4]) if len(sys.argv) == 5 else None)
    for problem in found:
        print(f"collect_python_licences: {problem}", file=sys.stderr)
    if found:
        raise SystemExit(1)
    Path(sys.argv[3]).write_text("".join(f"{t}\n" for t in sorted(names)), encoding="utf-8")
    print(f"{len(list(Path(sys.argv[1]).glob('*.dist-info')))} packages, terms: {' '.join(sorted(names))}")
