"""Publish a fixture pack digest for the cell the session host runs.

The table comes out of the record loader, from file names and hashes, as a release does.
"""

from __future__ import annotations

from typing import Any

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.engine_manifest import cell_key
from lilbee.providers.fleet.engine_record import build_manifests
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS

PREFIX = "https://index.test/packs/"
NEWEST_HOST_OS = "99.0"
_SLUGS = {
    EngineName.MISTRAL: "mistral_rs",
    EngineName.TEI: "tei",
    EngineName.IK_LLAMA: "ik_llama_cpp",
    EngineName.LLAMA: "llama_cpp",
    EngineName.MLX: "mlx",
}


def render_table(wheels: list[tuple[str, str, int]]) -> dict[str, Any]:
    """The table that the loader builds for *wheels*, as (file name, sha256, size) rows."""
    record = [
        {"file": name, "url": PREFIX + name, "sha256": sha, "size": size}
        for name, sha, size in wheels
    ]
    return dict(build_manifests(record))


def wheel_name(name: EngineName, version: str, key: str, min_os: str | None = None) -> str:
    """The file name of the wheel of engine *name* for the cell *key*."""
    os_name, arch, backend = key.split("-", 2)
    if os_name == "windows":
        platform = "win_amd64"
    elif os_name == "macos":
        platform = "macosx_14_0_arm64"
    else:
        platform = f"manylinux_{(min_os or '2.28').replace('.', '_')}_{arch}"
    build = f"1.{backend.replace('-', '_')}"
    return f"lilbee_engine_{_SLUGS[name]}-{version}-{build}-py3-none-{platform}.whl"


def publish(
    monkeypatch: pytest.MonkeyPatch,
    name: EngineName,
    sha: str,
    key: str | None = None,
    version: str | None = None,
    min_os: str | None = None,
    host_os: str = NEWEST_HOST_OS,
) -> None:
    """Make the table list *sha* for the pinned version of *name* in cell *key*.

    Without *key* the cell is the one the session host runs. The wheel has the OS
    floor *min_os* and the host has the OS version *host_os*, newer than any floor
    unless a test is about the floor.
    """
    pin = ADAPTERS[name].pin_record()
    if key is None:
        key = cell_key(runnable.session_host(), pin["backend"])
    assert key is not None
    file = wheel_name(name, version or pin["version"], key, min_os)
    table = render_table([(file, sha, 1)])
    monkeypatch.setattr(engine_manifest, "MANIFESTS", {**engine_manifest.MANIFESTS, **table})
    monkeypatch.setattr(engine_manifest, "host_os_version", lambda _os: host_os)
