"""The flatpakref is the only single-file installer that works for these apps.

Every channel's manifest uses extra-data, and `flatpak build-bundle` drops the
extra-data pointer from a bundle's detached metadata (flatpak#1334), so the
.flatpak assets releases used to carry could never be installed. A flatpakref
carries the remote instead of the payload, so what it must get right is the
pointer: the app id, the repo, the branch, and the key the repo is signed with.
"""

from __future__ import annotations

import base64
import configparser
import pathlib
import subprocess
import sys

import pytest

_RENDER = pathlib.Path(__file__).resolve().parents[1] / "packaging/tools/render_flatpakref.py"
_FLATPAK = pathlib.Path(__file__).resolve().parents[1] / "packaging/flatpak"

_GROUP = "Flatpak Ref"
_REPO_URL = "https://tobocop2.github.io/flatpak-lilbee/repo/"
_KEY = b"\x99\x02\x0dnot a real key\x00\xff"

_APP_IDS = [
    "io.github.tobocop2.lilbee",
    "io.github.tobocop2.lilbee.cuda",
    "io.github.tobocop2.lilbee.rocm",
    "io.github.tobocop2.lilbee.compat",
]


def _run(metainfo: pathlib.Path, key_file: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_RENDER),
            str(metainfo),
            "--repo-url",
            _REPO_URL,
            "--gpg-key-file",
            str(key_file),
            "--remote-name",
            "lilbee",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def key_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "key.gpg"
    path.write_bytes(_KEY)
    return path


def _render(app_id: str, key_file: pathlib.Path) -> configparser.ConfigParser:
    result = _run(_FLATPAK / f"{app_id}.metainfo.xml", key_file)
    assert result.returncode == 0, result.stderr
    parsed = configparser.ConfigParser()
    parsed.read_string(result.stdout)
    return parsed


@pytest.mark.parametrize("app_id", _APP_IDS)
def test_the_ref_names_the_app_it_was_rendered_for(app_id, key_file):
    """A ref carrying the wrong id installs the wrong channel, silently."""
    assert _render(app_id, key_file)[_GROUP]["Name"] == app_id


@pytest.mark.parametrize("app_id", _APP_IDS)
def test_the_ref_points_at_the_repo_and_the_branch_it_publishes(app_id, key_file):
    """build-update-repo writes the master branch; any other value is 'ref not found'."""
    ref = _render(app_id, key_file)[_GROUP]
    assert ref["Url"] == _REPO_URL
    assert ref["Branch"] == "master"


@pytest.mark.parametrize("app_id", _APP_IDS)
def test_the_advertised_key_is_the_signing_key(app_id, key_file):
    """A mangled key fails GPG verification on the user's machine, not in CI."""
    encoded = _render(app_id, key_file)[_GROUP]["GPGKey"]
    assert "\n" not in encoded
    assert base64.b64decode(encoded) == _KEY


@pytest.mark.parametrize("app_id", _APP_IDS)
def test_the_ref_brings_its_own_runtime_remote(app_id, key_file):
    """Without flathub the freedesktop runtime is unresolvable and the install fails."""
    ref = _render(app_id, key_file)[_GROUP]
    assert ref["RuntimeRepo"] == "https://dl.flathub.org/repo/flathub.flatpakrepo"
    assert ref["IsRuntime"] == "false"


@pytest.mark.parametrize("app_id", _APP_IDS)
def test_the_display_fields_come_from_the_metainfo(app_id, key_file):
    """One source of truth for the channel's name and summary, so they cannot drift."""
    metainfo = (_FLATPAK / f"{app_id}.metainfo.xml").read_text()
    ref = _render(app_id, key_file)[_GROUP]
    assert f"<name>{ref['Title']}</name>" in metainfo
    assert f"<summary>{ref['Comment']}</summary>" in metainfo
    assert f'<url type="homepage">{ref["Homepage"]}</url>' in metainfo


@pytest.mark.parametrize("app_id", _APP_IDS)
def test_the_suggested_remote_is_the_one_the_docs_name(app_id, key_file):
    """Installs from a ref and from the documented remote-add must not diverge."""
    assert _render(app_id, key_file)[_GROUP]["SuggestRemoteName"] == "lilbee"


def test_every_published_channel_is_covered_here():
    """A channel added without a case in this file would ship an untested installer."""
    on_disk = {path.name.removesuffix(".metainfo.xml") for path in _FLATPAK.glob("*.metainfo.xml")}
    assert on_disk == set(_APP_IDS)


@pytest.mark.parametrize("app_id", _APP_IDS)
def test_no_manifest_declares_a_branch_the_ref_would_not_know_about(app_id):
    """The rendered Branch is a constant; a manifest opting out of master breaks it silently."""
    manifest = (_FLATPAK / f"{app_id}.yml").read_text()
    assert "branch:" not in manifest
    assert "default-branch:" not in manifest


def test_a_metainfo_without_an_id_fails_instead_of_rendering_a_nameless_ref(tmp_path, key_file):
    """A ref with an empty Name is accepted by the renderer and rejected by every user."""
    metainfo = tmp_path / "broken.metainfo.xml"
    metainfo.write_text('<?xml version="1.0"?><component><name>x</name></component>')
    result = _run(metainfo, key_file)
    assert result.returncode != 0
    assert "id" in result.stderr


def test_an_empty_key_file_fails_instead_of_rendering_an_unverifiable_ref(tmp_path):
    """A failed `gpg --export` leaves an empty file, and GPGKey= reads as valid ini."""
    key_file = tmp_path / "empty.gpg"
    key_file.write_bytes(b"")
    result = _run(_FLATPAK / "io.github.tobocop2.lilbee.metainfo.xml", key_file)
    assert result.returncode != 0
    assert not result.stdout


def test_a_missing_key_file_fails_instead_of_rendering_an_unverifiable_ref(tmp_path):
    """An unsigned ref installs nothing: the repo it points at is signed."""
    result = _run(_FLATPAK / "io.github.tobocop2.lilbee.metainfo.xml", tmp_path / "absent.gpg")
    assert result.returncode != 0
    assert "absent.gpg" in result.stderr
    assert not result.stdout
