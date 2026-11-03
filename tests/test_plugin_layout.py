"""The agent plugin directory holds the only copy of each skill and a valid Claude Code manifest."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "lilbee"
SKILL_DIRS = sorted(p for p in (PLUGIN / "skills").iterdir() if p.is_dir())


def _frontmatter(skill_md: Path) -> dict:
    _, block, _ = skill_md.read_text(encoding="utf-8").split("---\n", 2)
    return yaml.safe_load(block)


def test_both_skills_are_present():
    assert [p.name for p in SKILL_DIRS] == ["lilbee-mcp", "lilbee-mcp-wiki"]


@pytest.mark.parametrize("skill_dir", SKILL_DIRS, ids=lambda p: p.name)
def test_skill_name_matches_directory_and_description_fits(skill_dir):
    meta = _frontmatter(skill_dir / "SKILL.md")
    assert meta["name"] == skill_dir.name
    assert 1 <= len(meta["description"]) <= 1024


def test_wheel_maps_the_plugin_skill_to_the_installed_skill_path():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include == {"plugins/lilbee/skills/lilbee-mcp": "lilbee/skills/lilbee_mcp"}
    assert (ROOT / next(iter(force_include)) / "SKILL.md").is_file()


def test_manifest_and_marketplace_agree():
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "lilbee"
    assert manifest["version"]
    assert (
        manifest["version"]
        == json.loads((ROOT / "packages" / "lilbee" / "package.json").read_text(encoding="utf-8"))[
            "version"
        ]
    )
    entry = next(p for p in market["plugins"] if p["name"] == manifest["name"])
    assert entry["source"] == "./plugins/lilbee"
    assert (ROOT / entry["source"]).resolve() == PLUGIN


def test_mcp_server_runs_lilbee_mcp():
    config = json.loads((PLUGIN / ".mcp.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["lilbee"] == {"command": "lilbee", "args": ["mcp"]}


def test_worker_agent_names_itself():
    meta = _frontmatter(PLUGIN / "agents" / "lilbee-worker.md")
    assert meta["name"] == "lilbee-worker"
    assert meta["description"]
