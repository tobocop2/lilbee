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


def test_litellm_extra_stops_below_the_compiled_line() -> None:
    # litellm 1.92 moved to a maturin build. PyPI publishes no macOS wheel for
    # any release since, so a mac install compiles the Rust crate out of the
    # sdist or fails without a toolchain. 1.93.1 added win_amd64 but no macOS,
    # so the whole line stays out, prereleases included.
    req = _extra_requirement("litellm", "litellm")
    for version in ("1.92.0rc1", "1.92.0", "1.93.0", "1.93.1", "1.94.1"):
        assert not req.specifier.contains(version, prereleases=True), version


def test_litellm_extra_admits_locked_stable() -> None:
    req = _extra_requirement("litellm", "litellm")
    assert req.specifier.contains("1.89.1")
    assert req.specifier.contains("1.91.0")


def test_mcp_rejects_breaking_major_beta() -> None:
    # mcp 2.x is a breaking rewrite; every 2.x release so far is a pre-release,
    # so an uncapped range hands beta-channel installs an untested major.
    req = _base_requirement("mcp")
    assert not req.specifier.contains("2.0.0b1", prereleases=True)


def test_mcp_admits_current_stable_line() -> None:
    req = _base_requirement("mcp")
    assert req.specifier.contains("1.26.0")
    assert req.specifier.contains("1.28.1")
