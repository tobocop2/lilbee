#!/usr/bin/env python3
"""Generate the lilbee.sh/compat/ PEP 503 index from forks' tagged GitHub releases.

Serves one index, ``lilbee.sh/compat/``, from one or more fork release
feeds: the pre-Haswell lancedb wheels (tobocop2/lancedb, releases tagged
``lancedb-v<version>+compat``) and, temporarily, the crawlberg preview
wheels (tobocop2/crawlberg, releases tagged with ``+lilbee``) -- the
crawlberg source goes once crawlberg 1.8.0 is on PyPI. Each ``--source``
writes its own project directory under ``<out>/compat/<project>/``; the
root ``<out>/compat/index.html`` then lists every project directory found
there, so any subset of sources can run without one clobbering another's
directory or its entry on the root page. Run by pages.yml at deploy time
alongside build_pep503_indexes.py.

A source's ``on-missing`` is ``fail`` (the default, used by lancedb) or
``warn`` (used by the crawlberg preview source): "fail" stops the build when the fork
has no matching release or ``gh api`` errors; "warn" emits a GitHub Actions
warning and skips that project's directory (exit 0) instead, so the
temporary preview source can never break the Pages deploy. A release with a
wheel asset that has no sha256 digest is a worse failure than either -- it
always aborts the build, regardless of on-missing. Two ``--source`` specs
naming the same ``project`` are rejected outright, so one source can never
silently overwrite another's project directory.

A source's ``project`` becomes a directory name (``<out>/compat/<project>/``)
and an ``<a href>`` target, so it must already be a PEP 503 normalized
project name: lowercase letters, digits, and ``-`` only, no leading,
trailing, or doubled ``-``. PEP 503 normalization (lowercasing and
collapsing runs of ``-_.``) is not applied to a bad name to make it legal --
an already-invalid ``project`` is rejected outright, since it is a constant
an operator writes into a ``pages.yml`` call site, not free-text a user
types; silently rewriting a typo would serve wheels from a directory nobody
asked for instead of failing the build where the typo was made.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import re
import subprocess
import sys
from enum import StrEnum
from pathlib import Path

_SOURCE_KEYS = frozenset({"repo", "tag-filter", "project", "on-missing"})
_SOURCE_REQUIRED_KEYS = frozenset({"repo", "tag-filter", "project"})

# The set of strings PEP 503's normalization (lowercase; collapse runs of
# -_. to a single -) leaves unchanged: lowercase alphanumeric segments
# joined by single hyphens, no leading/trailing hyphen. Also, incidentally,
# safe as a directory name (no `<>:"/\|?*` or other path-separator/control
# characters -- the Windows-illegal set that broke CI) and safe to place
# unescaped in HTML (no `<>&"'`). The two real project names, lancedb and
# crawlberg, don't collide with a reserved Windows device name (CON, NUL,
# COM1, ...); this pattern alone doesn't rule those out and isn't relied
# on to.
_NORMALIZED_PROJECT_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class OnMissing(StrEnum):
    """What build_source does when a fork has no matching release or gh api fails.

    FAIL (the default, used by lancedb) prints the message and fails the
    build. WARN (used by the crawlberg preview source) emits a GitHub
    Actions warning and skips that project's directory instead, so the
    temporary preview source can never break the Pages deploy.
    """

    FAIL = "fail"
    WARN = "warn"


@dataclasses.dataclass(frozen=True)
class Source:
    """One fork release feed to mirror into <out>/compat/<project>/."""

    repo: str
    tag_filter: str
    project: str
    on_missing: OnMissing = OnMissing.FAIL


def _parse_source(spec: str) -> Source:
    """Parse one --source spec: comma-separated key=value pairs.

    Recognized keys: repo, tag-filter, project (all required), on-missing
    (optional, "fail" or "warn", default "fail"). An unknown key, a key
    repeated within the same spec, or a spec missing a required key is
    rejected rather than silently accepted or last-value-wins. project must
    already be a PEP 503 normalized name (lowercase letters, digits, and
    "-" only); see _NORMALIZED_PROJECT_RE.
    """
    fields: dict[str, str] = {}
    for part in spec.split(","):
        key, sep, value = part.partition("=")
        key = key.strip()
        if not sep:
            raise argparse.ArgumentTypeError(f"--source entry {part!r} is not key=value")
        if key not in _SOURCE_KEYS:
            raise argparse.ArgumentTypeError(f"--source {spec!r}: unknown key {key!r}")
        if key in fields:
            raise argparse.ArgumentTypeError(f"--source {spec!r}: key {key!r} is repeated")
        fields[key] = value.strip()

    missing = _SOURCE_REQUIRED_KEYS - fields.keys()
    if missing:
        raise argparse.ArgumentTypeError(f"--source {spec!r} is missing {sorted(missing)}")

    project = fields["project"]
    if not _NORMALIZED_PROJECT_RE.fullmatch(project):
        raise argparse.ArgumentTypeError(
            f"--source {spec!r}: project {project!r} is not a PEP 503 normalized "
            "name (lowercase letters, digits, and '-' only)"
        )

    on_missing_raw = fields.get("on-missing", OnMissing.FAIL.value)
    try:
        on_missing = OnMissing(on_missing_raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--source {spec!r}: on-missing must be fail or warn"
        ) from None

    return Source(
        repo=fields["repo"],
        tag_filter=fields["tag-filter"],
        project=project,
        on_missing=on_missing,
    )


def _jq_filter(tag_filter: str) -> str:
    """Build the jq filter selecting wheel assets from releases whose tag contains tag_filter."""
    return (
        f".[] | select(.tag_name | contains({json.dumps(tag_filter)})) | .assets[] "
        '| select(.name | endswith(".whl")) '
        "| {name, url: .browser_download_url, digest}"
    )


def fetch_wheels(repo: str, tag_filter: str) -> list[tuple[str, str, str]]:
    """Return (filename, download_url, sha256) for every matching-release wheel asset."""
    # gh is a trusted CLI on PATH. Both arguments that reach it, `repo` (in the
    # API path) and `tag_filter` (json.dumps-escaped into the --jq program via
    # _jq_filter), are controlled constants hardcoded in each --source spec at
    # its pages.yml call site, not user input (S603/S607 are ignored for this
    # file in pyproject).
    out = subprocess.check_output(
        ["gh", "api", "--paginate", f"repos/{repo}/releases", "--jq", _jq_filter(tag_filter)],
        text=True,
    )
    wheels: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        digest = (a.get("digest") or "").removeprefix("sha256:")
        if not digest:
            raise SystemExit(f"asset {a['name']} has no sha256 digest from the API")
        wheels.append((a["name"], a["url"], digest))
    return wheels


def _missing(on_missing: OnMissing, project: str, message: str) -> int:
    """Report a no-release/unreadable-fork condition for one source's project.

    "fail" (the default, used by lancedb): print the message and return 1,
    failing the build. "warn" (used by the crawlberg preview source): emit a
    `::warning::` annotation and return 0 with the project directory left
    unwritten, so the build keeps the other sources instead of breaking.
    """
    if on_missing == OnMissing.WARN:
        print(f"::warning::{message}; skipping {project}", file=sys.stderr)
        return 0
    print(message, file=sys.stderr)
    return 1


def _write_project_index(out: Path, project: str, wheels: list[tuple[str, str, str]]) -> None:
    """Write <out>/<project>/index.html linking every wheel.

    project is used verbatim: _parse_source only ever hands this function a
    PEP 503 normalized name (_NORMALIZED_PROJECT_RE), which cannot contain
    an HTML metacharacter, so escaping it here would be a no-op. url and
    name come from the fork's GitHub releases, unvalidated, and stay
    escaped.
    """
    project_dir = out / project
    project_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f'    <a href="{html.escape(url)}#sha256={sha}">{html.escape(name)}</a><br/>'
        for name, url, sha in sorted(wheels)
    )
    (project_dir / "index.html").write_text(
        f"<!DOCTYPE html>\n<html><body>\n  <h1>Links for {project}</h1>\n{rows}\n</body></html>\n"
    )


def _write_root_index(out: Path) -> None:
    """Write <out>/index.html listing every project directory present in out.

    Scans the filesystem rather than tracking which sources this run wrote,
    so the page always reflects what is actually on disk -- a project
    another source skipped (on-missing warn) is simply absent, and a project
    this run did not touch but that is already there is still listed. Used
    verbatim, not escaped, for the same reason as in _write_project_index:
    the only writer of a directory under <out> is _write_project_index,
    which only ever receives a PEP 503 normalized project name.
    """
    projects = sorted(p.name for p in out.iterdir() if p.is_dir() and (p / "index.html").exists())
    rows = "\n".join(f'    <a href="{p}/">{p}</a><br/>' for p in projects)
    (out / "index.html").write_text(f"<!DOCTYPE html>\n<html><body>\n{rows}\n</body></html>\n")


def _duplicate_project(sources: list[Source]) -> str | None:
    """Return the first project name used by more than one source, or None.

    Two sources writing the same <out>/compat/<project>/ would have the
    second silently replace the first's directory -- the exact clobber the
    per-project-directory design exists to rule out. Checked before any
    source is fetched or written, so a collision aborts with nothing built.
    """
    seen: set[str] = set()
    for source in sources:
        if source.project in seen:
            return source.project
        seen.add(source.project)
    return None


def build_source(out: Path, source: Source) -> int:
    """Fetch one source's wheels and write its project directory. Returns an exit code."""
    try:
        wheels = fetch_wheels(source.repo, source.tag_filter)
    except subprocess.CalledProcessError as exc:
        return _missing(
            source.on_missing, source.project, f"gh api failed for {source.repo}: {exc}"
        )

    if not wheels:
        return _missing(
            source.on_missing,
            source.project,
            f"no {source.project} wheels found in {source.repo} releases",
        )

    _write_project_index(out, source.project, wheels)
    print(f"{source.project} index: {len(wheels)} wheels from {source.repo}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output site dir; writes <out>/compat/")
    ap.add_argument(
        "--source",
        type=_parse_source,
        action="append",
        required=True,
        metavar="repo=...,tag-filter=...,project=...[,on-missing=fail|warn]",
        help="a fork release feed to mirror into its own project dir; repeatable",
    )
    args = ap.parse_args()
    sources: list[Source] = args.source

    duplicate = _duplicate_project(sources)
    if duplicate:
        print(f"--source project {duplicate!r} is used by more than one source", file=sys.stderr)
        return 1

    out = Path(args.out) / "compat"
    out.mkdir(parents=True, exist_ok=True)

    for source in sources:
        rc = build_source(out, source)
        if rc != 0:
            return rc

    _write_root_index(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
