"""The harness gives the stated tier 2 verdict for each stand-in engine binary and pin."""

from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.conformance.harness import Verdict, run_managed

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-in binary is a POSIX executable script"
)

SHA = "a" * 64
GOOD_PIN: dict[str, Any] = {
    "engine": "fake",
    "version": "1.2.3",
    "backend": None,
    "launch": ["{binary}", "--port", "{port}"],
    "platforms": {
        "linux-x86_64": {
            "url": "https://example.org/fake-linux.tar.gz",
            "sha256": SHA,
            "min_os": None,
        },
        "macos-arm64": {
            "url": "https://example.org/fake-mac.tar.gz",
            "sha256": SHA,
            "min_os": "14.0",
        },
    },
    "memory": {"sizing": ["{binary}", "--estimate", "{model}"], "rule": None},
    "licenses": ["MIT", "Apache-2.0", "BSD-3-Clause"],
}
P, F, K = Verdict.PASS, Verdict.FAIL, Verdict.FALLBACK


def pin_with(**changes: Any) -> dict[str, Any]:
    pin = copy.deepcopy(GOOD_PIN)
    for key, value in changes.items():
        if key == "platform":
            pin["platforms"].update(value)
        else:
            pin[key] = value
    return pin


def _platform(
    url: str = "https://e.org/x", sha: str = SHA, min_os: str | None = None
) -> dict[str, Any]:
    return {"url": url, "sha256": sha, "min_os": min_os}


# (name, binary overrides, pin, verdicts that differ from a clean pass)
CASES: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Verdict]]] = [
    ("ref", {}, GOOD_PIN, {}),
    ("no_devices_pin_backend", {"devices": "usage"}, pin_with(backend="metal"), {"M2": K}),
    ("no_devices_no_backend", {"devices": "usage"}, GOOD_PIN, {"M2": F}),
    ("launch_without_port", {}, pin_with(launch=["{binary}", "--serve"]), {"M1": F, "M6": F}),
    ("no_help", {"help_rc": 2}, GOOD_PIN, {"M1": F}),
    ("estimate_fails", {"estimate": False}, GOOD_PIN, {"M3": F}),
    (
        "memory_rule_only",
        {"estimate": False},
        pin_with(memory={"sizing": None, "rule": "file_size"}),
        {"M3": K},
    ),
    ("no_memory_rule", {}, pin_with(memory={"sizing": None, "rule": None}), {"M3": F}),
    ("pin_no_sha", {}, pin_with(platform={"linux-x86_64": _platform(sha="")}), {"M4": F}),
    (
        "pin_http_url",
        {},
        pin_with(platform={"linux-x86_64": _platform(url="http://e.org/x")}),
        {"M4": F},
    ),
    ("pin_mac_no_floor", {}, pin_with(platform={"macos-arm64": _platform()}), {"M4": F}),
    ("pin_no_platforms", {}, pin_with(platforms={}), {"M4": F}),
    ("licence_gpl", {}, pin_with(licenses=["MIT", "GPL-3.0-only"]), {"M5": F}),
    ("licence_agpl", {}, pin_with(licenses=["MIT", "AGPL-3.0-only"]), {"M5": F}),
    ("licence_lgpl_is_not_gpl", {}, pin_with(licenses=["MIT", "LGPL-2.1-only"]), {}),
    ("licence_empty_entry", {}, pin_with(licenses=["MIT", ""]), {"M5": F}),
    ("licence_none_listed", {}, pin_with(licenses=[]), {"M5": F}),
    ("ignores_sigterm", {"term": "ignore"}, GOOD_PIN, {"M6": F}),
]


@pytest.mark.parametrize(("name", "overrides", "pin", "differs"), CASES, ids=[c[0] for c in CASES])
def test_harness_gives_the_stated_managed_verdict(
    make_binary: Callable[..., Path],
    name: str,
    overrides: dict[str, Any],
    pin: dict[str, Any],
    differs: dict[str, Verdict],
) -> None:
    got = run_managed(make_binary(name, **overrides), pin=pin, timeout=15.0)
    want = {key: P for key in ("M1", "M2", "M3", "M4", "M5", "M6")}
    want.update(differs)
    assert got == want
