"""The agent plugin directory holds the only copy of each skill and a valid Claude Code manifest."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

from lilbee.cli.launchers import pi as pi_launcher
from lilbee.cli.launchers import skill_install

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
    assert force_include == {
        "plugins/lilbee/skills/lilbee-mcp": "lilbee/skills/lilbee_mcp",
        "packages/lilbee/plugin": "lilbee/agent_plugin",
    }
    assert (ROOT / "plugins/lilbee/skills/lilbee-mcp" / "SKILL.md").is_file()
    assert (ROOT / "packages/lilbee/plugin" / "extensions" / "lilbee.js").is_file()


def test_launcher_wheel_and_binary_agree_on_the_skill_and_extension_locations():
    """The launcher finds the skill and the pi extension where the wheel and the binary put them."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    script = (ROOT / "tools" / "wheel-build" / "build_lilbee_binary.sh").read_text(encoding="utf-8")
    binary = dict(re.findall(r"--include-data-dir=(\S+)=(\S+)", script))
    pairs = [
        (skill_install._CHECKOUT_SKILL, skill_install._PACKAGED_SKILL),
        (skill_install._CHECKOUT_AGENT_PLUGIN, skill_install._PACKAGED_AGENT_PLUGIN),
    ]
    expected = {
        checkout.relative_to(ROOT).as_posix(): installed.relative_to(ROOT / "src").as_posix()
        for checkout, installed in pairs
    }
    assert wheel == expected
    assert {k: v for k, v in binary.items() if k in expected} == expected


def test_plugin_version_follows_the_lilbee_release_version():
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert manifest["version"] == pyproject["project"]["version"]


def test_manifest_and_marketplace_agree():
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "lilbee"
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


NPM_PACKAGE = ROOT / "packages" / "lilbee"


def test_npm_plugin_command_matches_the_claude_mcp_config():
    command = (NPM_PACKAGE / "plugin" / "command.js").read_text(encoding="utf-8")
    found = re.search(r"MCP_COMMAND = (\[.*?\])", command)
    assert found
    config = json.loads((PLUGIN / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]["lilbee"]
    assert json.loads(found.group(1)) == [config["command"], *config["args"]]


@pytest.mark.parametrize(
    "name",
    [pi_launcher._OPENAI_URL_ENV_VAR, pi_launcher._MCP_URL_ENV_VAR, pi_launcher._MODELS_ENV_VAR],
)
def test_pi_extension_reads_the_env_vars_the_pi_launcher_sets(name):
    extension = (ROOT / "packages" / "lilbee" / "plugin" / "extensions" / "lilbee.js").read_text(
        encoding="utf-8"
    )
    assert f'"{name}"' in extension

_INSTALL_DOCS = [ROOT / "README.md", ROOT / "docs" / "usage.md"]


def _install_commands(doc: Path) -> list[str]:
    """Every backticked command in the per-agent install table of one document."""
    text = doc.read_text(encoding="utf-8")
    table = text.split("| Agent | Install |", 1)[1].split("\n\n", 1)[0]
    return re.findall(r"`([^`]+)`", table)


def _manifest_names() -> tuple[str, str, str, str]:
    marketplace_file = ROOT / ".claude-plugin" / "marketplace.json"
    marketplace = json.loads(marketplace_file.read_text(encoding="utf-8"))
    package = json.loads((NPM_PACKAGE / "package.json").read_text(encoding="utf-8"))
    slug = re.search(r"github\.com/(.+?)\.git$", package["repository"]["url"]).group(1)
    return marketplace["name"], marketplace["plugins"][0]["name"], slug, package["name"]


@pytest.mark.parametrize("doc", _INSTALL_DOCS, ids=lambda p: p.name)
def test_install_table_commands_use_the_manifest_names(doc):
    marketplace, plugin, slug, npm_name = _manifest_names()
    commands = _install_commands(doc)
    assert f"claude plugin marketplace add {slug}" in commands
    assert f"claude plugin install {plugin}@{marketplace}" in commands
    assert f"opencode plugin add {npm_name}" in commands
    assert f"pi install npm:{npm_name}" in commands
    assert f"npx skills add {slug} -g -a hermes-agent" in commands
    assert f"npx skills add {slug} -g" in commands
    assert "lilbee mcp" in commands


@pytest.mark.parametrize("doc", _INSTALL_DOCS, ids=lambda p: p.name)
def test_skills_installs_are_global(doc):
    installs = [c for c in _install_commands(doc) if c.startswith("npx skills add")]
    assert len(installs) == 2
    for command in installs:
        assert "-g" in command.split(), f"{command} installs into the current project"


@pytest.mark.parametrize("doc", _INSTALL_DOCS, ids=lambda p: p.name)
def test_install_docs_name_the_node_version_skills_needs(doc):
    assert "The `npx skills` commands need Node 22.20 or later." in doc.read_text(encoding="utf-8")


@pytest.mark.parametrize("doc", _INSTALL_DOCS, ids=lambda p: p.name)
def test_hermes_snippet_registers_the_mcp_command(doc):
    text = doc.read_text(encoding="utf-8")
    snippet = re.search(r"```yaml\n(mcp_servers:\n.*?)```", text, re.S).group(1)
    entry = {"command": "lilbee", "args": ["mcp"]}
    assert yaml.safe_load(snippet) == {"mcp_servers": {"lilbee": entry}}


def test_install_table_agrees_between_readme_and_usage():
    assert _install_commands(ROOT / "docs" / "usage.md") == _install_commands(ROOT / "README.md")


@pytest.mark.parametrize(
    "doc", [ROOT / "README.md", ROOT / "docs" / "agent-models.md"], ids=lambda p: p.name
)
def test_launcher_lists_name_every_launcher(doc):
    text = doc.read_text(encoding="utf-8")
    for agent in ("opencode", "hermes", "pi", "claude"):
        assert f"`lilbee launch {agent}`" in text
