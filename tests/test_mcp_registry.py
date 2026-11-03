"""Tests for the MCP Registry listing: server.json, mcpName and the publish step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SERVER_JSON = ROOT / "server.json"
NPM_PACKAGE = ROOT / "packages" / "lilbee" / "package.json"
SCHEMA = ROOT / "tests" / "data" / "mcp-registry-server.schema.json"
WORKFLOW = ROOT / ".github" / "workflows" / "publish-packages.yml"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _npm_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["npm"]["steps"]


def _step_index(steps: list[dict[str, Any]], needle: str) -> int:
    hits = [i for i, step in enumerate(steps) if needle in step.get("run", "")]
    assert len(hits) == 1, f"expected one step running {needle!r}, found {len(hits)}"
    return hits[0]


def test_server_json_matches_registry_schema() -> None:
    schema = _load(SCHEMA)
    server = _load(SERVER_JSON)
    assert server["$schema"] == schema["$id"]
    jsonschema.validate(server, schema)


def test_schema_rejects_a_server_without_a_version() -> None:
    server = _load(SERVER_JSON)
    del server["version"]
    with pytest.raises(jsonschema.ValidationError, match="'version' is a required property"):
        jsonschema.validate(server, _load(SCHEMA))


def test_server_name_equals_mcp_name() -> None:
    assert _load(SERVER_JSON)["name"] == _load(NPM_PACKAGE)["mcpName"]


def test_server_version_equals_npm_version() -> None:
    npm_version = _load(NPM_PACKAGE)["version"]
    server = _load(SERVER_JSON)
    assert server["version"] == npm_version
    assert [p["version"] for p in server["packages"]] == [npm_version]


def test_package_entry_runs_the_npm_launcher_with_mcp() -> None:
    (package,) = _load(SERVER_JSON)["packages"]
    assert package["registryType"] == "npm"
    assert package["identifier"] == _load(NPM_PACKAGE)["name"]
    assert package["transport"] == {"type": "stdio"}
    assert package["packageArguments"] == [{"type": "positional", "value": "mcp"}]


def test_workflow_publishes_to_registry_after_npm_publish() -> None:
    steps = _npm_steps()
    npm_publish = _step_index(steps, "npm publish")
    login = _step_index(steps, "mcp-publisher login github-oidc")
    publish = _step_index(steps, "mcp-publisher publish")
    assert npm_publish < login < publish
    assert steps[publish]["if"] == steps[npm_publish]["if"]
    assert steps[login]["if"] == steps[npm_publish]["if"]
