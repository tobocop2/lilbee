"""Tests for the engine pack index generator (lilbee.sh/packs/).

The packs are public wheels on ``engine-packs-<date>`` prerelease tags. lilbee
downloads each one from its release URL; the index lets ``pip download`` and
``uv pip install`` resolve the same files with ``--extra-index-url``. A draft
release answers 404 to the public, so its wheels must never reach the index.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_pack_index.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_pack_index", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bpi = _load()

_TEI = "lilbee_engine_tei-1.9.4-1.cpu-py3-none-manylinux_2_34_x86_64.whl"
_IK = "lilbee_engine_ik_llama_cpp-0+git13cffc006b82-1.cpu-py3-none-manylinux_2_34_x86_64.whl"
_SHA_A = "a" * 64
_SHA_B = "b" * 64
_BASE = "https://github.com/tobocop2/lilbee/releases/download"


def _asset(name: str, tag: str, sha: str | None = _SHA_A) -> dict[str, object]:
    return {
        "name": name,
        "browser_download_url": f"{_BASE}/{tag}/{name}",
        "digest": f"sha256:{sha}" if sha else None,
    }


def _release(tag: str, assets: list[dict[str, object]], draft: bool = False) -> dict[str, object]:
    return {"tag_name": tag, "draft": draft, "assets": assets}


def _gh_output(releases: list[dict[str, object]]) -> str:
    return "\n".join(json.dumps(r) for r in releases) + "\n"


def _patch_gh(
    monkeypatch: pytest.MonkeyPatch, releases: list[dict[str, object]]
) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake(cmd: list[str], **_kw: object) -> str:
        calls.append(cmd)
        return _gh_output(releases)

    monkeypatch.setattr(bpi.subprocess, "check_output", fake)
    return calls


def test_project_name_is_pep503_normalized() -> None:
    assert bpi.project_of(_TEI) == "lilbee-engine-tei"
    assert bpi.project_of(_IK) == "lilbee-engine-ik-llama-cpp"


def test_fetch_keeps_wheels_of_dated_pack_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_gh(
        monkeypatch,
        [
            _release(
                "engine-packs-20261116-tei",
                [_asset(_TEI, "engine-packs-20261116-tei"), _asset("SHA256SUMS", "x")],
            ),
        ],
    )
    wheels = bpi.fetch_wheels("tobocop2/lilbee")
    assert wheels == [(_TEI, f"{_BASE}/engine-packs-20261116-tei/{_TEI}", _SHA_A)]
    assert "repos/tobocop2/lilbee/releases" in calls[0]


def test_fetch_skips_draft_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_gh(
        monkeypatch,
        [_release("engine-packs-20261104", [_asset(_TEI, "engine-packs-20261104")], draft=True)],
    )
    assert bpi.fetch_wheels("tobocop2/lilbee") == []


def test_fetch_skips_releases_that_are_not_pack_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gh(
        monkeypatch,
        [
            _release("engine-packs-pin-3", [_asset(_TEI, "engine-packs-pin-3")]),
            _release("v0.6.90", [_asset(_TEI, "v0.6.90")]),
            _release("engine-binaries", [_asset(_TEI, "engine-binaries")]),
        ],
    )
    assert bpi.fetch_wheels("tobocop2/lilbee") == []


def test_fetch_ignores_blank_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bpi.subprocess, "check_output", lambda *_a, **_k: "\n \n")
    assert bpi.fetch_wheels("tobocop2/lilbee") == []


def test_fetch_requires_a_sha256_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_gh(
        monkeypatch,
        [_release("engine-packs-20261115", [_asset(_TEI, "engine-packs-20261115", sha=None)])],
    )
    with pytest.raises(SystemExit, match="no sha256 digest"):
        bpi.fetch_wheels("tobocop2/lilbee")


def test_fetch_merges_the_same_wheel_listed_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_gh(
        monkeypatch,
        [
            _release("engine-packs-20261115", [_asset(_TEI, "engine-packs-20261115")]),
            _release("engine-packs-20261116", [_asset(_TEI, "engine-packs-20261116")]),
        ],
    )
    assert len(bpi.fetch_wheels("tobocop2/lilbee")) == 1


def test_fetch_rejects_one_filename_with_two_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_gh(
        monkeypatch,
        [
            _release("engine-packs-20261115", [_asset(_TEI, "engine-packs-20261115")]),
            _release("engine-packs-20261116", [_asset(_TEI, "engine-packs-20261116", _SHA_B)]),
        ],
    )
    with pytest.raises(SystemExit, match="two different digests"):
        bpi.fetch_wheels("tobocop2/lilbee")


def test_write_index_layout_and_links(tmp_path: Path) -> None:
    url_tei = f"{_BASE}/engine-packs-20261116-tei/{_TEI}"
    url_ik = f"{_BASE}/engine-packs-20261115/{_IK}"
    bpi.write_index(tmp_path, [(_TEI, url_tei, _SHA_A), (_IK, url_ik, _SHA_B)])
    root = (tmp_path / "packs" / "index.html").read_text()
    assert 'href="lilbee-engine-tei/"' in root
    assert 'href="lilbee-engine-ik-llama-cpp/"' in root
    page = (tmp_path / "packs" / "lilbee-engine-tei" / "index.html").read_text()
    assert f'href="{url_tei}#sha256={_SHA_A}"' in page
    assert _IK not in page
    ik_page = (tmp_path / "packs" / "lilbee-engine-ik-llama-cpp" / "index.html").read_text()
    assert f'href="{url_ik}#sha256={_SHA_B}"' in ik_page


def test_main_writes_the_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_gh(
        monkeypatch,
        [_release("engine-packs-20261116-tei", [_asset(_TEI, "engine-packs-20261116-tei")])],
    )
    monkeypatch.setattr("sys.argv", ["build_pack_index.py", str(tmp_path)])
    assert bpi.main() == 0
    # The packs live in their own repository, which is the default source.
    assert "repos/tobocop2/lilbee-engine-packs/releases" in calls[0]
    assert (tmp_path / "packs" / "lilbee-engine-tei" / "index.html").is_file()


def test_main_fails_when_no_pack_is_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_gh(monkeypatch, [])
    monkeypatch.setattr("sys.argv", ["build_pack_index.py", str(tmp_path)])
    assert bpi.main() == 1
    assert "no engine pack wheels" in capsys.readouterr().err
    assert not (tmp_path / "packs").exists()


def test_pages_workflow_builds_the_pack_index() -> None:
    # The index only reaches lilbee.sh if the deploy job's input holds it: the step
    # must run the generator into the same dir the compat and backend indexes use.
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "pages.yml").read_text()
    )
    steps = workflow["jobs"]["build-wheel-indexes"]["steps"]
    runs = [s["run"] for s in steps if "build_pack_index.py wheel-indexes" in s.get("run", "")]
    assert len(runs) == 1
