"""The agent plugin directory holds the only copy of each skill and a valid Claude Code manifest."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

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
    assert force_include == {"plugins/lilbee/skills/lilbee-mcp": "lilbee/skills/lilbee_mcp"}
    assert (ROOT / next(iter(force_include)) / "SKILL.md").is_file()


def _live_shell_code(script: Path) -> str:
    """The script text without whole-line comments and trailing comments."""
    lines = script.read_text(encoding="utf-8").splitlines()
    code = [re.sub(r"\s#.*$", "", line) for line in lines if not line.lstrip().startswith("#")]
    return "\n".join(code)


def test_a_comment_in_the_build_script_is_not_live_code(tmp_path):
    script = tmp_path / "build.sh"
    script.write_text(
        "# --include-data-dir=a=b\n"
        "  --include-data-dir=real=dest \\ # --include-data-dir=a=b\n"
        "echo done # --include-data-dir=c=d\n",
        encoding="utf-8",
    )
    found = re.findall(r"--include-data-dir=(\S+)=(\S+)", _live_shell_code(script))
    assert found == [("real", "dest")]


def test_launcher_wheel_and_binary_agree_on_the_skill_location():
    """The launcher reads the skill where the wheel and the frozen binary put it."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    script = _live_shell_code(ROOT / "tools" / "wheel-build" / "build_lilbee_binary.sh")
    binary = dict(re.findall(r"--include-data-dir=(\S+)=(\S+)", script))
    installed = skill_install._PACKAGED_SKILL.relative_to(ROOT / "src").as_posix()
    checkout = skill_install._CHECKOUT_SKILL.relative_to(ROOT).as_posix()
    expected = {checkout: installed}
    assert wheel == expected
    assert {k: v for k, v in binary.items() if k.startswith("plugins/")} == expected


def test_plugin_version_follows_the_lilbee_release_version():
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert manifest["version"] == pyproject["project"]["version"]


def test_claude_plugin_ci_job_pins_the_cli_version():
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    runs = [step.get("run", "") for step in ci["jobs"]["claude-plugin"]["steps"]]
    installs = [run for run in runs if "@anthropic-ai/claude-code" in run]
    assert len(installs) == 1
    assert re.search(r"@anthropic-ai/claude-code@\d+\.\d+\.\d+(\s|$)", installs[0])


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
