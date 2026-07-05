"""Shared helper for building minimal synthetic wheels in tests."""

from __future__ import annotations

import base64
import hashlib
import zipfile
from pathlib import Path


def make_wheel(
    path: Path,
    name: str,
    version: str,
    python: str = "py3",
    abi: str = "none",
    plat: str = "any",
    pkg_dir: str | None = None,
    pkg_files: dict[str, bytes] | None = None,
    metadata_extra: str = "",
) -> Path:
    """Create a minimal valid wheel zip at *path*."""
    tag = f"{python}-{abi}-{plat}"
    dist_info = f"{name}-{version}.dist-info"
    filename = f"{name}-{version}-{tag}.whl"
    path.mkdir(parents=True, exist_ok=True)
    whl_path = path / filename

    records: list[str] = []

    def _record_entry(arcname: str, data: bytes) -> str:
        digest = hashlib.sha256(data).digest()
        h = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return f"{arcname},sha256={h},{len(data)}"

    with zipfile.ZipFile(whl_path, "w", zipfile.ZIP_DEFLATED) as zf:
        meta = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{metadata_extra}"
        meta_bytes = meta.encode()
        arcname = f"{dist_info}/METADATA"
        zf.writestr(arcname, meta_bytes)
        records.append(_record_entry(arcname, meta_bytes))

        wheel_content = f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: {tag}\n"
        wheel_bytes = wheel_content.encode()
        arcname = f"{dist_info}/WHEEL"
        zf.writestr(arcname, wheel_bytes)
        records.append(_record_entry(arcname, wheel_bytes))

        if pkg_dir and pkg_files:
            for relname, data in pkg_files.items():
                arcname = f"{pkg_dir}/{relname}"
                zf.writestr(arcname, data)
                records.append(_record_entry(arcname, data))

        record_arcname = f"{dist_info}/RECORD"
        records.append(f"{record_arcname},,")
        record_bytes = ("\n".join(records) + "\n").encode()
        zf.writestr(record_arcname, record_bytes)

    return whl_path
