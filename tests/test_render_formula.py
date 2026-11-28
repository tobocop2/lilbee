"""The Homebrew renderer writes each tap formula from its template in packaging/homebrew.

The publish never reads the tap's copy, so a block one release drops comes back in the
next. The renderer also owns the license: all three published formulae once went on
declaring Elastic-2.0 long after the project moved to MIT.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest
import yaml

_RENDER = pathlib.Path(__file__).resolve().parents[1] / "packaging/tools/render_formula.py"
_SEEDS = pathlib.Path(__file__).resolve().parents[1] / "packaging/homebrew"
_CELLS = pathlib.Path(__file__).resolve().parents[1] / ".github/release-cells.yml"
_OLD_MAC_CELLS = pathlib.Path(__file__).resolve().parents[1] / ".github/oldmac-cells.yml"

_SHA = "1" * 64


def _render(
    formula: pathlib.Path, *args: str, template: pathlib.Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Render *template* (by default *formula* itself) into *formula*."""
    source = str(template or formula)
    command = [sys.executable, str(_RENDER), str(formula), "--template", source]
    return subprocess.run(
        [*command, "--version", "9.9.9", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("seed", "mode", "sha_flag"),
    [
        ("lilbee-compat.rb", "--compat", "--sha-linux-compat"),
        ("lilbee-cuda.rb", "--cuda", "--sha-linux-cu125"),
        ("lilbee-rocm.rb", "--rocm", "--sha-linux-rocm"),
        ("lilbee-compat-cuda.rb", "--compat-cuda", "--sha-linux-compat-cu124"),
    ],
)
def test_a_stale_license_is_rewritten(tmp_path, seed, mode, sha_flag):
    """A template with a stale license still publishes the project license."""
    formula = tmp_path / seed
    formula.write_text(
        _SEEDS.joinpath(seed).read_text().replace('license "MIT"', 'license "Elastic-2.0"')
    )
    result = _render(formula, mode, sha_flag, _SHA)
    assert result.returncode == 0, result.stderr
    assert 'license "MIT"' in formula.read_text()
    assert "Elastic-2.0" not in formula.read_text()


def test_every_seed_already_declares_the_project_license(tmp_path):
    """A seed with the wrong license would ship it to every release of that flavor."""
    for seed in _SEEDS.glob("*.rb"):
        assert 'license "MIT"' in seed.read_text(), seed.name


_MACOS_INTEL_BLOCK = """
    on_intel do
      url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-macos-x86_64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"

      on_sonoma :or_older do
        url "https://github.com/tobocop2/lilbee/releases/download/v#{version}/lilbee-compat-macos-x86_64"
        sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      end
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


_OLD_MAC = "3" * 63 + "4"
_OLD_INTEL_MAC = "3" * 63 + "5"


def test_each_old_mac_digest_lands_under_its_own_compat_url(tmp_path):
    """Each old-Mac url sits inside its arch block, so each digest must reach its own url."""
    formula = _seed(tmp_path)
    result = _render(
        formula,
        *_EVERY_DIGEST,
        "--sha-macos-arm64-compat",
        _OLD_MAC,
        "--sha-macos-x86_64-compat",
        _OLD_INTEL_MAC,
    )
    assert result.returncode == 0, result.stderr
    rendered = formula.read_text(encoding="utf-8")
    assert f'lilbee-compat-macos-arm64"\n        sha256 "{_OLD_MAC}"' in rendered
    assert f'lilbee-compat-macos-x86_64"\n        sha256 "{_OLD_INTEL_MAC}"' in rendered
    assert f'lilbee-macos-arm64"\n      sha256 "{_ARM64}"' in rendered
    assert f'lilbee-macos-x86_64"\n      sha256 "{_MACOS_INTEL}"' in rendered
    assert rendered.count("on_sonoma :or_older do") == 2


def test_without_an_old_mac_digest_the_formula_drops_that_block(tmp_path):
    """The old-Mac build ships apart from the release; a placeholder digest must never publish."""
    formula = _seed(tmp_path)
    result = _render(formula, *_EVERY_DIGEST)
    assert result.returncode == 0, result.stderr
    rendered = formula.read_text(encoding="utf-8")
    assert "compat-macos" not in rendered.split("MACOS_FLOORS")[0]
    assert "on_sonoma" not in rendered
    assert "0" * 64 not in rendered
    assert f'lilbee-macos-arm64"\n      sha256 "{_ARM64}"\n    end\n' in rendered
    assert f'lilbee-macos-x86_64"\n      sha256 "{_MACOS_INTEL}"\n    end\n' in rendered


def test_an_old_mac_block_without_a_digest_is_dropped_on_its_arch_only(tmp_path):
    formula = _seed(tmp_path)
    result = _render(formula, *_EVERY_DIGEST, "--sha-macos-arm64-compat", _OLD_MAC)
    assert result.returncode == 0, result.stderr
    rendered = formula.read_text(encoding="utf-8")
    assert f'lilbee-compat-macos-arm64"\n        sha256 "{_OLD_MAC}"' in rendered
    assert rendered.count("on_sonoma :or_older do") == 1
    assert f'lilbee-macos-x86_64"\n      sha256 "{_MACOS_INTEL}"\n    end\n' in rendered


def test_a_release_without_old_mac_digests_does_not_break_the_next_publish(tmp_path):
    """Each publish renders the repo's template into the tap file, never the tap file."""
    tap = tmp_path / "tap" / "lilbee.rb"
    tap.parent.mkdir()
    template = _SEEDS / "lilbee.rb"
    with_compat = (
        *("--sha-macos-arm64-compat", _OLD_MAC),
        *("--sha-macos-x86_64-compat", _OLD_INTEL_MAC),
    )

    first = _render(tap, *_EVERY_DIGEST, template=template)
    assert first.returncode == 0, first.stderr
    assert "on_sonoma" not in tap.read_text(encoding="utf-8")
    second = _render(tap, *_EVERY_DIGEST, *with_compat, template=template)
    assert second.returncode == 0, second.stderr
    assert tap.read_text(encoding="utf-8").count("on_sonoma :or_older do") == 2
    third = _render(tap, *_EVERY_DIGEST, template=template)
    assert third.returncode == 0, third.stderr


def test_the_publish_renders_the_template_and_never_the_tap_copy():
    action = _SEEDS.parents[1] / ".github/actions/publish-homebrew-formula/action.yml"
    run = yaml.safe_load(action.read_text(encoding="utf-8"))["runs"]["steps"][0]["run"]

    assert 'template="packaging/homebrew/$(basename "${FORMULA_PATH}")"' in run
    assert '--template "${template}"' in run
    assert "[ ! -f" not in run


def test_every_published_formula_has_a_template():
    """The publish reads only the template, so a flavor without one cannot publish."""
    workflows = _SEEDS.parents[1] / ".github/workflows"
    texts = [workflow.read_text(encoding="utf-8") for workflow in workflows.glob("*.yml")]
    published = {
        path for text in texts for path in re.findall(r"formula-path: Formula/(\S+\.rb)", text)
    }
    assert len(published) == 5, published
    assert sorted(published) == sorted(seed.name for seed in _SEEDS.glob("*.rb"))


def test_a_formula_without_the_old_mac_block_fails_when_no_digest_is_given(tmp_path):
    seed = _SEEDS.joinpath("lilbee.rb").read_text(encoding="utf-8")
    block = seed[seed.index("\n\n      on_sonoma") : seed.index("    end\n\n    on_intel")]
    result = _render(_seed(tmp_path, drop=block), *_EVERY_DIGEST)
    assert result.returncode != 0
    assert "old-Mac block for lilbee-compat-macos-arm64" in result.stderr


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


def test_the_compat_cuda_flavor_pins_the_cu124_asset_and_leaves_compat_alone(tmp_path):
    """The two compat assets share a prefix, so each digest must land under its own url."""
    formula = tmp_path / "lilbee-compat-cuda.rb"
    formula.write_text(
        _SEEDS.joinpath("lilbee-compat-cuda.rb").read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = _render(formula, "--compat-cuda", "--sha-linux-compat-cu124", _SHA)
    assert result.returncode == 0, result.stderr
    rendered = formula.read_text(encoding="utf-8")
    assert 'version "9.9.9"' in rendered
    assert f'lilbee-compat-linux-x86_64-cu124"\n      sha256 "{_SHA}"' in rendered

    compat = tmp_path / "lilbee-compat.rb"
    compat.write_text(
        _SEEDS.joinpath("lilbee-compat.rb").read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = _render(compat, "--compat", "--sha-linux-compat", _SHA)
    assert result.returncode == 0, result.stderr
    assert "cu124" not in compat.read_text(encoding="utf-8")


def test_the_compat_cuda_flavor_requires_its_digest(tmp_path):
    formula = tmp_path / "lilbee-compat-cuda.rb"
    formula.write_text(
        _SEEDS.joinpath("lilbee-compat-cuda.rb").read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = _render(formula, "--compat-cuda")
    assert result.returncode != 0
    assert "--sha-linux-compat-cu124" in result.stderr


def test_every_formula_seed_conflicts_with_every_other_lilbee_binary_formula():
    """Two formulas that install the same binary must refuse each other."""
    names = {"lilbee-compat-cuda", "lilbee-compat", "lilbee-rocm"}
    for name in names:
        seed = _SEEDS.joinpath(f"{name}.rb").read_text(encoding="utf-8")
        for other in names - {name}:
            assert f'conflicts_with "tobocop2/lilbee/{other}"' in seed, (name, other)


def test_the_default_formula_refuses_each_mac_asset_below_the_floor_of_its_cell():
    """Homebrew refuses the install below the floor, instead of a binary that dies at load."""
    cells = [
        cell
        for path in (_CELLS, _OLD_MAC_CELLS)
        for cell in yaml.safe_load(path.read_text(encoding="utf-8"))
    ]
    floors = {
        cell["asset_name"]: cell["macos_floor"]
        for cell in cells
        if cell["asset_name"].startswith(("lilbee-macos-", "lilbee-compat-macos-"))
    }
    seed = (_SEEDS / "lilbee.rb").read_text(encoding="utf-8")
    table = seed[seed.index("MACOS_FLOORS = {") : seed.index("}.freeze")]
    assert dict(re.findall(r'"(lilbee-[\w-]+)"\s*=>\s*"([\d.]+)"', table)) == floors
    assert len(floors) == 4
    # A whole-formula macOS dependency would refuse the compat builds on the Macs they serve.
    assert "depends_on macos:" not in seed
