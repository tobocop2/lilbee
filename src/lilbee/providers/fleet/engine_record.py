"""The pack record: one entry per published engine wheel, read from package data."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from importlib import resources
from typing import Any

PACKAGES_URL = "https://github.com/tobocop2/lilbee/releases/download/"
RECORD_FILE = "engine_packs.json"
_FIELDS = ("file", "url", "sha256", "size")
_ENGINES = {
    "mistral_rs": "mistral.rs",
    "tei": "tei",
    "ik_llama_cpp": "ik_llama.cpp",
    "llama_cpp": "llama.cpp",
    "mlx": "mlx",
}
_WHEEL = re.compile(
    r"lilbee_engine_(?P<slug>[a-z0-9_]+)-(?P<version>[^-]+)-(?P<build>[^-]+)"
    r"-[^-]+-[^-]+-(?P<platform>[^-]+)\.whl"
)
_BUILD = re.compile(r"\d+\.(?P<backend>[a-z0-9_]+)")
_MANYLINUX = re.compile(r"manylinux_2_(?P<minor>\d+)_(?P<arch>x86_64|aarch64)")
_MACOS = re.compile(r"macosx_(?P<major>\d+)_(?P<minor>\d+)_(?P<arch>arm64)")
_DIGEST = re.compile(r"[0-9a-f]{64}")

Manifests = Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]


class PackRecordError(ValueError):
    """A row of the pack record cannot be used."""


def _host(label: str, platform: str) -> tuple[str, str, str | None]:
    if platform == "win_amd64":
        return "windows", "x86_64", None
    if found := _MANYLINUX.fullmatch(platform):
        return "linux", found["arch"], f"2.{found['minor']}"
    if found := _MACOS.fullmatch(platform):
        return "macos", found["arch"], f"{found['major']}.{found['minor']}"
    raise PackRecordError(f"{label}: the platform tag {platform!r} is not one an engine ships for.")


def _check_fields(label: str, entry: Mapping[str, Any]) -> None:
    for field in _FIELDS:
        if field not in entry:
            raise PackRecordError(f"{label}: the field {field!r} is missing.")
    for field in ("file", "url", "sha256"):
        if not isinstance(entry[field], str):
            raise PackRecordError(f"{label}: the field {field!r} is not a string.")
    size = entry["size"]
    if type(size) is not int or size <= 0:
        raise PackRecordError(f"{label}: the size {size!r} is not a positive whole number.")
    if _DIGEST.fullmatch(entry["sha256"]) is None:
        raise PackRecordError(
            f"{label}: the sha256 {entry['sha256']!r} is not 64 lowercase hex digits."
        )


def _cell(label: str, entry: Mapping[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    _check_fields(label, entry)
    name, url = entry["file"], entry["url"]
    wheel = _WHEEL.fullmatch(name)
    if wheel is None:
        raise PackRecordError(f"{label}: not the file name of an engine wheel.")
    engine = _ENGINES.get(wheel["slug"])
    if engine is None:
        raise PackRecordError(f"{label}: {wheel['slug']!r} is not an engine project.")
    build = _BUILD.fullmatch(wheel["build"])
    if build is None:
        raise PackRecordError(f"{label}: the build tag {wheel['build']!r} names no backend.")
    if not url.startswith("https://") or url.rsplit("/", 1)[-1] != name:
        raise PackRecordError(
            f"{label}: the url {url!r} is not an https url that ends with the file."
        )
    os_name, arch, min_os = _host(label, wheel["platform"])
    key = f"{os_name}-{arch}-{build['backend'].replace('_', '-')}"
    row = {"sha256": entry["sha256"], "url": url, "size": entry["size"], "min_os": min_os}
    return engine, wheel["version"], key, row


def build_manifests(record: Any) -> Manifests:
    """The table of the record: a row per engine, version and cell. Refuses a bad row by name."""
    if not isinstance(record, list):
        raise PackRecordError("the pack record is not a list of rows.")
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for index, entry in enumerate(record, start=1):
        if not isinstance(entry, dict):
            raise PackRecordError(f"row {index}: not an object.")
        label = f"row {index} ({entry.get('file', 'no file')})"
        engine, version, key, row = _cell(label, entry)
        cells = table.setdefault(engine, {}).setdefault(version, {})
        if key in cells:
            raise PackRecordError(f"{label}: {engine} {version} already has a wheel for {key}.")
        cells[key] = row
    return table


def load_manifests() -> Manifests:
    """Read the shipped record and build its table."""
    text = (
        resources.files("lilbee.providers.fleet").joinpath(RECORD_FILE).read_text(encoding="utf-8")
    )
    try:
        record = json.loads(text)
    except json.JSONDecodeError as error:
        raise PackRecordError(f"the pack record is not JSON: {error}.") from error
    return build_manifests(record)
