"""The server.json listing and the npm package it points at must agree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SERVER_JSON = ROOT / "server.json"
NPM_PACKAGE = ROOT / "packages" / "lilbee" / "package.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_server_name_equals_mcp_name() -> None:
    assert _load(SERVER_JSON)["name"] == _load(NPM_PACKAGE)["mcpName"]


def test_server_version_equals_npm_version() -> None:
    npm_version = _load(NPM_PACKAGE)["version"]
    server = _load(SERVER_JSON)
    assert server["version"] == npm_version
    assert [p["version"] for p in server["packages"]] == [npm_version]
