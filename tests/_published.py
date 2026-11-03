"""Publish a fixture pack digest for the cell the session host runs.

The table comes out of the generator, from file names and hashes, as a release does.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import engine_manifest
from lilbee.providers.fleet.engine_manifest import cell_key
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS

GENERATOR = Path(__file__).resolve().parents[1] / "tools" / "gen_engine_digests.py"
PREFIX = "https://index.test/packs/"
NEWEST_HOST_OS = "99.0"
_SLUGS = {
    EngineName.MISTRAL: "mistral_rs",
    EngineName.TEI: "tei",
    EngineName.IK_LLAMA: "ik_llama_cpp",
    EngineName.LLAMA: "llama_cpp",
    EngineName.MLX: "mlx",
}


def generator() -> ModuleType:
    """The table generator, loaded from tools/."""
    spec = importlib.util.spec_from_file_location("gen_engine_digests", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def render_table(wheels: list[tuple[str, str, int]]) -> dict[str, Any]:
    """The table the generator writes for *wheels*, as (file name, sha256, size) rows."""
    packages = [{"filename": name, "hash": f"sha256={sha}"} for name, sha, _ in wheels]
    sizes = {name: size for name, _, size in wheels}
    tree = ast.parse(generator().render(packages, sizes, PREFIX))
    assignment = next(node.value for node in tree.body if isinstance(node, ast.Assign))
    table: dict[str, Any] = ast.literal_eval(assignment)
    return table


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
