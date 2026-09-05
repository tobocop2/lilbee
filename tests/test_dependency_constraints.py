"""Guard dependency specifiers against known-broken resolutions.

Every lilbee release is a pre-release, so the documented install commands
pass --pre / --prerelease=allow. uv and pip apply that strategy to the whole
resolution, not just lilbee, so an unbounded extra can resolve to an untested
pre-release of a dependency. These tests pin down ranges that are known to
break when that happens.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

_PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def _base_requirement(name: str) -> Requirement:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    reqs = (Requirement(r) for r in data["project"]["dependencies"])
    return next(r for r in reqs if r.name == name)


def _extra_requirement(extra: str, name: str) -> Requirement:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    reqs = (Requirement(r) for r in data["project"]["optional-dependencies"][extra])
    return next(r for r in reqs if r.name == name)


def test_litellm_extra_stops_before_the_next_major() -> None:
    # 1.98.0 ships cp310-abi3 wheels for macOS (x86_64/arm64), Windows, and
    # manylinux, so the maturin-build wheel gap that capped the line at <1.92
    # is closed. The <1.99 cap only keeps the next line's prereleases off the
    # --prerelease=allow channel.
    req = _extra_requirement("litellm", "litellm")
    assert not req.specifier.contains("1.99.0rc1", prereleases=True)
    assert not req.specifier.contains("1.99.0")


def test_litellm_extra_admits_locked_stable() -> None:
    req = _extra_requirement("litellm", "litellm")
    assert req.specifier.contains("1.89.1")
    assert req.specifier.contains("1.91.0")
    assert req.specifier.contains("1.98.0")


def test_mcp_rejects_the_line_without_the_mcpserver_module() -> None:
    # The tool server subclasses mcp.server.mcpserver.MCPServer and configures
    # streamable-http on the app factory. 1.x has neither (it shipped
    # mcp.server.fastmcp.FastMCP and a settings object), so it cannot run this.
    req = _base_requirement("mcp")
    for version in ("1.26.0", "1.28.1"):
        assert not req.specifier.contains(version), version


def test_mcp_admits_the_current_major_and_stops_below_the_next() -> None:
    # <3 also keeps the next major's prereleases off the beta channel, which
    # resolves with --prerelease=allow.
    req = _base_requirement("mcp")
    assert req.specifier.contains("2.0.0")
    assert not req.specifier.contains("3.0.0b1", prereleases=True)
