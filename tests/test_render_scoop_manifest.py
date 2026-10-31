"""The Scoop renderer pins each single-asset manifest to its own Windows exe."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_RENDER = _ROOT / "packaging/tools/render_scoop_manifest.py"
_SEED = _ROOT / "packaging/scoop/lilbee-compat-cuda.json"

_SHA = "a" * 64
_BASE_URL = "https://github.com/tobocop2/lilbee/releases/download/v9.9.9/"


def _render(manifest: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_RENDER), str(manifest), "--version", "9.9.9", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_the_compat_cuda_seed_renders_the_cu124_exe(tmp_path):
    manifest = tmp_path / "lilbee-compat-cuda.json"
    manifest.write_text(_SEED.read_text(encoding="utf-8"), encoding="utf-8")
    result = _render(manifest, "--compat-cuda", "--sha-windows-compat-cu124", _SHA)
    assert result.returncode == 0, result.stderr
    rendered = json.loads(manifest.read_text(encoding="utf-8"))
    assert rendered["version"] == "9.9.9"
    assert rendered["architecture"]["64bit"] == {
        "url": f"{_BASE_URL}lilbee-compat-windows-x86_64-cu124.exe#/lilbee.exe",
        "hash": _SHA,
    }
    assert rendered["bin"] == "lilbee.exe"


def test_the_compat_cuda_mode_requires_its_digest(tmp_path):
    manifest = tmp_path / "lilbee-compat-cuda.json"
    manifest.write_text(_SEED.read_text(encoding="utf-8"), encoding="utf-8")
    result = _render(manifest, "--compat-cuda")
    assert result.returncode != 0
    assert "--sha-windows-compat-cu124" in result.stderr


def test_a_compat_manifest_does_not_accept_the_compat_cuda_asset(tmp_path):
    """The compat and compat CUDA exes share a prefix; the URL pattern must not mix them."""
    manifest = tmp_path / "lilbee-compat.json"
    manifest.write_text(_SEED.read_text(encoding="utf-8"), encoding="utf-8")
    result = _render(manifest, "--compat", "--sha-windows-compat", _SHA)
    assert result.returncode != 0
    assert "compat download URL" in result.stderr


@pytest.mark.parametrize(
    ("mode", "sha_flag", "asset"),
    [
        ("--cuda", "--sha-windows-cu125", "lilbee-windows-x86_64-cu125.exe"),
        ("--compat", "--sha-windows-compat", "lilbee-compat-windows-x86_64.exe"),
    ],
)
def test_each_single_asset_mode_pins_its_own_exe(tmp_path, mode, sha_flag, asset):
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "0.0.1",
                "architecture": {
                    "64bit": {
                        "url": f"{_BASE_URL.replace('9.9.9', '0.0.1')}{asset}",
                        "hash": "0" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = _render(manifest, mode, sha_flag, _SHA)
    assert result.returncode == 0, result.stderr
    rendered = json.loads(manifest.read_text(encoding="utf-8"))
    assert rendered["architecture"]["64bit"] == {"url": f"{_BASE_URL}{asset}", "hash": _SHA}


def test_the_default_mode_requires_every_digest(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text("{}", encoding="utf-8")
    result = _render(manifest)
    assert result.returncode != 0
    assert "--sha-windows-cu124" in result.stderr
