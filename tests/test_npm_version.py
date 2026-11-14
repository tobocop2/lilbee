"""The npm package carries the lilbee release version, mapped by one script."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
NPM_PACKAGE = ROOT / "packages" / "lilbee"
VERSION_SCRIPT = ROOT / "scripts" / "release_version.sh"

# The release scripts run on the Linux and macOS runners only. On a Windows
# runner, `bash` is the WSL launcher with no distribution installed.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the release scripts run only on Linux and macOS"
)


def _script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(VERSION_SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _npm_version(version: str) -> str:
    mapped = _script("npm-version", version)
    assert mapped.returncode == 0, mapped.stderr
    return mapped.stdout.strip()


def _semver_key(version: str) -> tuple:
    core, _, pre = version.partition("-")
    # A release outranks its own prereleases; numeric identifiers compare as numbers.
    ids = [(0, int(i)) if i.isdigit() else (1, i) for i in pre.split(".")] if pre else []
    return tuple(int(n) for n in core.split(".")), not pre, ids


def test_package_version_is_the_lilbee_version_in_semver_form():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    package = json.loads((NPM_PACKAGE / "package.json").read_text(encoding="utf-8"))
    assert package["version"] == _npm_version(version)
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
    assert _npm_version(lilbee) == npm


def test_four_digit_beta_sorts_after_three_digit_beta():
    older, newer, final = (_npm_version(v) for v in ("0.6.90b691", "0.6.90b1000", "0.6.90"))
    assert _semver_key(older) < _semver_key(newer) < _semver_key(final)


@pytest.mark.parametrize("version", ["0.7.0rc1", "0.7.0a1", "0.7.0b1.post1", "0.7", "v0.7.0"])
def test_a_version_with_no_mapping_is_refused(version):
    mapped = _script("npm-version", version)
    assert mapped.returncode == 1
    assert f"'{version}' has no npm version" in mapped.stderr


def test_every_stamped_file_exists_and_includes_the_npm_package():
    listed = _script("files").stdout.split()
    assert "packages/lilbee/package.json" in listed
    assert "packages/lilbee/package-lock.json" in listed
    assert [name for name in listed if not (ROOT / name).is_file()] == []


def test_publish_checks_the_tag_then_publishes_with_the_latest_tag_and_skips_dev_builds():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "publish-packages.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["npm"]
    runs = [s.get("run", "") for s in job["steps"]]
    check = next(i for i, run in enumerate(runs) if "release_version.sh check" in run)
    publish = next(i for i, run in enumerate(runs) if "npm publish" in run)
    assert check < publish
    steps = job["steps"]
    checkout = next(s for s in steps if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout["with"]["ref"] == "${{ needs.resolve.outputs.tag }}"
    assert "--tag latest" in runs[publish]
    assert "!contains(needs.resolve.outputs.tag, '.dev')" in job["if"]
