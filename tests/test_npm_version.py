"""The npm package carries the lilbee release version, mapped by one script."""

import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
NPM_PACKAGE = ROOT / "packages" / "lilbee"
VERSION_SCRIPT = ROOT / "scripts" / "release_version.sh"


def _npm_version(version: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(VERSION_SCRIPT), "npm-version", version],
        capture_output=True,
        text=True,
        check=False,
    )


def _lilbee_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["version"]


def _semver_key(version: str) -> tuple:
    core, _, pre = version.partition("-")
    # A release outranks its own prereleases; numeric identifiers compare as numbers.
    ids = [(0, int(i)) if i.isdigit() else (1, i) for i in pre.split(".")] if pre else []
    return tuple(int(n) for n in core.split(".")), not pre, ids


def test_npm_version_is_the_lilbee_version_in_semver_form():
    package = json.loads((NPM_PACKAGE / "package.json").read_text(encoding="utf-8"))
    version = _lilbee_version()
    mapped = _npm_version(version)
    assert mapped.returncode == 0, mapped.stderr
    assert package["version"] == mapped.stdout.strip()
    assert package["lilbee"]["release"] == f"v{version}"


def test_lock_file_carries_the_package_version():
    package = json.loads((NPM_PACKAGE / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((NPM_PACKAGE / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["version"] == package["version"]
    assert lock["packages"][""]["version"] == package["version"]


@pytest.mark.parametrize(
    ("lilbee", "npm"),
    [
        ("0.6.90b691", "0.6.90-beta.691"),
        ("0.6.90b1000", "0.6.90-beta.1000"),
        ("0.6.90", "0.6.90"),
        ("0.6.90b420.dev711", "0.6.90-beta.420.dev.711"),
    ],
)
def test_beta_number_becomes_a_numeric_prerelease_identifier(lilbee, npm):
    mapped = _npm_version(lilbee)
    assert mapped.returncode == 0, mapped.stderr
    assert mapped.stdout.strip() == npm


def test_four_digit_beta_sorts_after_three_digit_beta():
    older = _npm_version("0.6.90b691").stdout.strip()
    newer = _npm_version("0.6.90b1000").stdout.strip()
    final = _npm_version("0.6.90").stdout.strip()
    assert _semver_key(older) < _semver_key(newer) < _semver_key(final)


@pytest.mark.parametrize("version", ["0.7.0rc1", "0.7.0a1", "0.7.0b1.post1", "0.7", "v0.7.0"])
def test_a_version_with_no_mapping_is_refused(version):
    mapped = _npm_version(version)
    assert mapped.returncode == 1
    assert f"'{version}' has no npm version" in mapped.stderr


def test_every_stamped_file_exists_and_names_the_version_files():
    listed = subprocess.run(
        ["bash", str(VERSION_SCRIPT), "files"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert "packages/lilbee/package.json" in listed
    assert "packages/lilbee/package-lock.json" in listed
    assert [name for name in listed if not (ROOT / name).is_file()] == []


def _npm_job() -> dict:
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "publish-packages.yml").read_text(encoding="utf-8")
    )
    return workflow["jobs"]["npm"]


def test_publish_uses_the_latest_tag_explicitly():
    publish = [s["run"] for s in _npm_job()["steps"] if "npm publish" in s.get("run", "")]
    assert len(publish) == 1
    assert re.search(r"npm publish\b.*--tag latest", publish[0])


def test_publish_checks_the_version_against_the_tag_before_it_publishes():
    steps = _npm_job()["steps"]
    names = [s.get("name", "") for s in steps]
    check = steps[names.index("Check package.json matches the tag")]
    assert "release_version.sh npm-version" in check["run"]
    publish_at = next(i for i, s in enumerate(steps) if "npm publish" in s.get("run", ""))
    assert names.index(check["name"]) < publish_at
    checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["with"]["ref"] == "${{ needs.resolve.outputs.tag }}"


def test_publish_skips_dev_builds():
    assert "!contains(needs.resolve.outputs.tag, '.dev')" in _npm_job()["if"]
