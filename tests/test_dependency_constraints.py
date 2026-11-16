"""Guard dependency specifiers against known-broken resolutions.

Every lilbee release is a pre-release, so the documented install commands
pass --pre / --prerelease=allow. uv and pip apply that strategy to the whole
resolution, not just lilbee, so an unbounded extra can resolve to an untested
pre-release of a dependency. These tests pin down ranges that are known to
break when that happens.

Floors that exist to keep a vulnerable transitive package out of a resolve are
guarded here too.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

from lilbee.providers.factory import sdk_library

SDK_DIST = sdk_library().dist
_PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"
_QA_REQUIREMENTS = Path(__file__).parents[1] / "tools" / "qa" / "requirements.txt"


def _base_requirement(name: str) -> Requirement:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    reqs = (Requirement(r) for r in data["project"]["dependencies"])
    return next(r for r in reqs if r.name == name)


def _extra_requirement(extra: str, name: str) -> Requirement:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    reqs = (Requirement(r) for r in data["project"]["optional-dependencies"][extra])
    return next(r for r in reqs if r.name == name)


def _qa_requirement(name: str) -> Requirement:
    lines = _QA_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    reqs = (Requirement(line) for line in lines if line.strip() and not line.startswith("#"))
    return next(r for r in reqs if r.name == name)


def test_providers_extra_stops_before_the_next_major() -> None:
    req = _extra_requirement("providers", SDK_DIST)
    assert not req.specifier.contains("3.0.0rc1", prereleases=True)
    assert not req.specifier.contains("3.0.0")


def test_providers_extra_admits_locked_stable() -> None:
    req = _extra_requirement("providers", SDK_DIST)
    assert req.specifier.contains("2.4.1")
    assert not req.specifier.contains("2.4.0")


def test_mcp_rejects_the_line_without_the_mcpserver_module() -> None:
    # The tool server subclasses mcp.server.mcpserver.MCPServer and configures
    # streamable-http on the app factory. 1.x has neither (it shipped
    # mcp.server.fastmcp.FastMCP and a settings object), so it cannot run this.
    req = _base_requirement("mcp")
    for version in ("1.26.0", "1.28.1"):
        assert not req.specifier.contains(version), version


def test_mcp_admits_the_current_major_and_stops_below_the_next() -> None:
    # 2.2.0 bounds StreamableHTTP sessions (CVE-2026-59951). <3 also keeps the
    # next major's prereleases off the beta channel, which resolves with
    # --prerelease=allow.
    req = _base_requirement("mcp")
    assert not req.specifier.contains("2.1.0")
    assert req.specifier.contains("2.2.0")
    assert req.specifier.contains("2.3.0")
    assert not req.specifier.contains("3.0.0b1", prereleases=True)


def test_qa_matrix_excludes_vulnerable_anyio() -> None:
    # 4.14.2 fixes GHSA-82r6-8w77-94w6 and GHSA-5p39-cfhj-2xmp.
    req = _qa_requirement("anyio")
    assert not req.specifier.contains("4.9.0")
    assert req.specifier.contains("4.14.2")


_ROOT = Path(__file__).parents[1]
_FORK_TAGS = (
    "macosx_11_0_arm64",
    "macosx_11_0_x86_64",
    "manylinux_2_28_aarch64",
    "manylinux_2_28_x86_64",
    "musllinux_1_2_aarch64",
    "musllinux_1_2_x86_64",
    "win_amd64",
)


def _fork_index() -> dict[str, Any]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    indexes = [i for i in data["tool"]["uv"]["index"] if i["name"] == "xberg-fork"]
    assert len(indexes) == 1
    index: dict[str, Any] = indexes[0]
    return index


def test_published_range_for_the_fork_dependency_resolves_on_pypi() -> None:
    # PyPI refuses a direct URL, and 1.4.0 is not on PyPI: the published range
    # stays at the newest release that is.
    req = _base_requirement("xberg")
    assert req.url is None
    assert req.specifier.contains("1.3.8")


def test_fork_index_is_flat_explicit_and_pinned_to_the_dependency() -> None:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    index = _fork_index()
    assert index["format"] == "flat"
    assert index["explicit"] is True
    assert data["tool"]["uv"]["sources"]["xberg"] == {"index": "xberg-fork"}


def test_lock_holds_the_fork_build_for_every_platform() -> None:
    lock = tomllib.loads((_ROOT / "uv.lock").read_text(encoding="utf-8"))
    (package,) = [p for p in lock["package"] if p["name"] == "xberg"]
    assert package["version"] == "1.4.0+lilbee"
    assert package["source"] == {"registry": _fork_index()["url"]}
    urls = [w["url"] for w in package["wheels"]]
    assert len(urls) == len(_FORK_TAGS)
    for tag in _FORK_TAGS:
        assert sum(url.endswith(f"-{tag}.whl") for url in urls) == 1, tag


def test_intel_mac_install_reads_the_fork_index() -> None:
    # `uv pip` ignores sources, so the Intel cell names the index itself.
    workflow = (_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert f"--find-links {_fork_index()['url']}" in workflow
