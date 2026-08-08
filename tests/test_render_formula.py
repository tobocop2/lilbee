"""The Homebrew renderer owns the license, because nothing else rewrites the tap.

A formula already in the tap is never re-seeded from packaging/homebrew: the publish
action uses a seed only when the formula is absent. So a wrong license there survives
every release until the renderer fixes it, which is how all three published formulae
went on declaring Elastic-2.0 long after the project moved to MIT.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

_RENDER = pathlib.Path(__file__).resolve().parents[1] / "packaging/tools/render_formula.py"
_SEEDS = pathlib.Path(__file__).resolve().parents[1] / "packaging/homebrew"

_SHA = "1" * 64


def _render(formula: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_RENDER), str(formula), "--version", "9.9.9", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("seed", "mode", "sha_flag"),
    [
        ("lilbee-compat.rb", "--compat", "--sha-linux-compat"),
        ("lilbee-rocm.rb", "--rocm", "--sha-linux-rocm"),
    ],
)
def test_a_stale_license_is_rewritten(tmp_path, seed, mode, sha_flag):
    """The tap's copy is the one users see, and only this pass can correct it."""
    formula = tmp_path / seed
    formula.write_text(
        _SEEDS.joinpath(seed).read_text().replace('license "MIT"', 'license "Elastic-2.0"')
    )
    result = _render(formula, mode, sha_flag, _SHA)
    assert result.returncode == 0, result.stderr
    assert 'license "MIT"' in formula.read_text()
    assert "Elastic-2.0" not in formula.read_text()


def test_every_seed_already_declares_the_project_license(tmp_path):
    """A seed with the wrong license would ship it to any flavor not yet in the tap."""
    for seed in _SEEDS.glob("*.rb"):
        assert 'license "MIT"' in seed.read_text(), seed.name


_MACOS_INTEL_BLOCK = """
    on_intel do
      url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-macos-x86_64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
    end
"""

_ARM64, _MACOS_INTEL, _LINUX = "1" * 64, "2" * 64, "3" * 64
_EVERY_DIGEST = (
    "--sha-macos-arm64",
    _ARM64,
    "--sha-macos-x86_64",
    _MACOS_INTEL,
    "--sha-linux-x86_64",
    _LINUX,
)


def _seed(tmp_path, drop: str = "") -> pathlib.Path:
    """The default formula seed, optionally with *drop* cut out of it."""
    seed = _SEEDS.joinpath("lilbee.rb").read_text(encoding="utf-8")
    assert not drop or drop in seed
    formula = tmp_path / "lilbee.rb"
    formula.write_text(seed.replace(drop, "") if drop else seed, encoding="utf-8")
    return formula


def test_the_default_formula_takes_a_digest_for_every_platform_it_ships(tmp_path):
    """Intel macOS has an asset in every release; the formula has to carry its digest."""
    formula = _seed(tmp_path)
    result = _render(formula, *_EVERY_DIGEST)
    assert result.returncode == 0, result.stderr
    rendered = formula.read_text(encoding="utf-8")
    for sha in (_ARM64, _MACOS_INTEL, _LINUX):
        assert rendered.count(sha) == 1, rendered


def test_a_formula_missing_a_platform_block_fails_loudly(tmp_path):
    """Skipping the absent block instead is how Intel Macs lost `brew install` outright."""
    result = _render(_seed(tmp_path, drop=_MACOS_INTEL_BLOCK), *_EVERY_DIGEST)
    assert result.returncode != 0
    assert "lilbee-macos-x86_64" in result.stderr


def test_the_default_formula_requires_the_intel_macos_digest(tmp_path):
    """A release that resolved no Intel digest must stop, not publish a partial formula."""
    result = _render(_seed(tmp_path), "--sha-macos-arm64", _ARM64, "--sha-linux-x86_64", _LINUX)
    assert result.returncode != 0
    assert "--sha-macos-x86_64" in result.stderr


def test_a_formula_with_no_license_line_fails_loudly(tmp_path):
    """Silently rendering a formula Homebrew will reject is worse than stopping."""
    formula = tmp_path / "lilbee-rocm.rb"
    formula.write_text(
        _SEEDS.joinpath("lilbee-rocm.rb").read_text().replace('  license "MIT"\n', "")
    )
    result = _render(formula, "--rocm", "--sha-linux-rocm", _SHA)
    assert result.returncode != 0
    assert "license" in result.stderr
