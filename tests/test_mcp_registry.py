"""Tests for the MCP Registry listing: server.json, mcpName and the publish step."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SERVER_JSON = ROOT / "server.json"
NPM_PACKAGE = ROOT / "packages" / "lilbee" / "package.json"
SCHEMA = ROOT / "tests" / "data" / "mcp-registry-server.schema.json"
PUBLISH_SCRIPT = "scripts/publish_mcp_registry.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "publish-packages.yml"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_jsonschema_is_a_declared_dev_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = pyproject["dependency-groups"]["dev"]
    assert any(re.match(r"jsonschema\b", entry) for entry in dev)


def test_registry_step_runs_the_script_on_server_json_with_oidc() -> None:
    job = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["npm"]
    assert job["permissions"]["id-token"] == "write"
    assert job["timeout-minutes"] <= 30
    steps = job["steps"]
    publish_npm = next(
        i for i, s in enumerate(steps) if s.get("run") == "npm publish --access public"
    )
    (registry,) = [i for i, s in enumerate(steps) if PUBLISH_SCRIPT in s.get("run", "")]
    assert publish_npm < registry
    step = steps[registry]
    assert step["run"].split() == ["bash", PUBLISH_SCRIPT, "server.json"]
    assert step["if"] == "vars.NPM_TRUSTED_PUBLISHING == 'enabled'"
    assert 0 < step["timeout-minutes"] < job["timeout-minutes"]
    installer = [s for s in steps if s.get("uses", "").startswith("sigstore/cosign-installer@")]
    assert len(installer) == 1
    assert steps.index(installer[0]) < registry


def _derive_step() -> dict[str, Any]:
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["npm"]["steps"]
    derive = [
        i
        for i, s in enumerate(steps)
        if "package.json" in s.get("run", "") and "server.json" in s.get("run", "")
    ]
    (registry,) = [i for i, s in enumerate(steps) if PUBLISH_SCRIPT in s.get("run", "")]
    assert len(derive) == 1
    assert derive[0] < registry
    assert steps[derive[0]]["if"] == steps[registry]["if"]
    return steps[derive[0]]


@pytest.mark.skipif(sys.platform == "win32", reason="the publish step runs only on ubuntu-latest")
def test_publish_derives_server_versions_from_the_npm_version(tmp_path: Path) -> None:
    jq = shutil.which("jq")
    assert jq is not None
    (tmp_path / "packages" / "lilbee").mkdir(parents=True)
    package = _load(NPM_PACKAGE)
    package["version"] = "9.8.7"
    (tmp_path / "packages" / "lilbee" / "package.json").write_text(
        json.dumps(package), encoding="utf-8"
    )
    (tmp_path / "server.json").write_text(SERVER_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    subprocess.run(
        ["bash", "-c", _derive_step()["run"]],
        cwd=tmp_path,
        env={"PATH": str(Path(jq).parent) + ":/usr/bin:/bin", "RUNNER_TEMP": str(runner_temp)},
        check=True,
    )
    derived = _load(tmp_path / "server.json")
    assert derived["version"] == "9.8.7"
    assert [p["version"] for p in derived["packages"]] == ["9.8.7"]
    assert derived["name"] == _load(SERVER_JSON)["name"]
