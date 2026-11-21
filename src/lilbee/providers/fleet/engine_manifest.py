"""The pack digest of one engine on one host, read from the release manifests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lilbee.providers.fleet.host_caps import HostCaps

SCHEMA = 1

# The release manifests of the engine packs, one per engine, copied as published. An engine
# with no published pack has no entry. Each manifest keeps the layout of the release's
# manifest.json: schema, engine, release and cells, where cells maps <os>-<arch>-<backend>
# to the row of one wheel. Hand-owned until the generator of design 6 writes it.
MANIFESTS: Mapping[str, Mapping[str, Any]] = {}


def cell_key(host: HostCaps, backend: str | None) -> str | None:
    """The manifest key of the cell *host* runs, or ``None`` when a part of it is unknown."""
    if host.os is None or host.arch is None or backend is None:
        return None
    return f"{host.os.value}-{host.arch.value}-{backend}"


def pack_digest(engine: str, host: HostCaps, backend: str | None) -> str | None:
    """The sha256 of the published wheel of *engine* for *host*, or ``None`` without one."""
    manifest = MANIFESTS.get(engine)
    key = cell_key(host, backend)
    if manifest is None or key is None:
        return None
    if manifest["schema"] != SCHEMA:
        raise ValueError(f"The manifest of engine {engine} has schema {manifest['schema']}.")
    cell = manifest["cells"].get(key)
    return cell["sha256"] if cell is not None else None
