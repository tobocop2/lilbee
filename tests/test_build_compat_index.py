"""The compat index lists the wheels of every fork, one project page per project."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_compat_index.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_compat_index", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bci = _load()

_SHA = "a" * 64
_CRAWLBERG = "crawlberg-1.13.0+compat-cp311-abi3-macosx_11_0_arm64.whl"
_LITER = "liter_llm-2.7.0+compat-cp311-abi3-macosx_11_0_x86_64.whl"
_LANCEDB = "lancedb-0.40.0+compat-cp39-abi3-macosx_11_0_x86_64.whl"


def _wheel(name: str) -> tuple[str, str, str]:
    return (name, f"https://example.test/{name}", _SHA)


def _fake_fetch(by_repo: dict[str, list[str]]):
    return lambda repo: [_wheel(n) for n in by_repo[repo]]


def _run(monkeypatch, tmp_path, by_repo, *argv) -> int:
    monkeypatch.setattr(bci, "fetch_wheels", _fake_fetch(by_repo))
    monkeypatch.setattr("sys.argv", ["build_compat_index.py", str(tmp_path), *argv])
    return bci.main()


@pytest.mark.parametrize(
    ("wheel", "project"),
    [(_CRAWLBERG, "crawlberg"), (_LITER, "liter-llm"), ("A.B_c-1.0-py3-none-any.whl", "a-b-c")],
)
def test_project_name_is_pep503_normalized(wheel, project):
    assert bci.project_of(wheel) == project


def test_every_fork_gets_its_own_project_page(monkeypatch, tmp_path):
    by_repo = {"o/lancedb": [_LANCEDB], "o/crawlberg": [_CRAWLBERG], "o/liter-llm": [_LITER]}
    rc = _run(
        monkeypatch,
        tmp_path,
        by_repo,
        "--repo",
        "o/lancedb",
        "--optional-repo",
        "o/crawlberg",
        "--optional-repo",
        "o/liter-llm",
    )
    assert rc == 0
    root = (tmp_path / "compat" / "index.html").read_text(encoding="utf-8")
    for project in ("crawlberg", "lancedb", "liter-llm"):
        assert f'<a href="{project}/">' in root
    crawlberg = (tmp_path / "compat" / "crawlberg" / "index.html").read_text(encoding="utf-8")
    assert f"{_CRAWLBERG}#sha256={_SHA}" in crawlberg
    assert _LITER not in crawlberg
    assert _LITER in (tmp_path / "compat" / "liter-llm" / "index.html").read_text(encoding="utf-8")


def test_an_asset_url_is_html_escaped_in_its_page(monkeypatch, tmp_path):
    url = "https://example.test/dl?name=a&sig=b"
    monkeypatch.setattr(bci, "fetch_wheels", lambda _repo: [(_LANCEDB, url, _SHA)])
    monkeypatch.setattr("sys.argv", ["build_compat_index.py", str(tmp_path), "--repo", "o/lancedb"])
    assert bci.main() == 0
    page = (tmp_path / "compat" / "lancedb" / "index.html").read_text(encoding="utf-8")
    assert f'href="https://example.test/dl?name=a&amp;sig=b#sha256={_SHA}"' in page


def test_an_optional_fork_with_no_wheels_is_skipped(monkeypatch, tmp_path, capsys):
    by_repo = {"o/lancedb": [_LANCEDB], "o/xberg": []}
    rc = _run(monkeypatch, tmp_path, by_repo, "--repo", "o/lancedb", "--optional-repo", "o/xberg")
    assert rc == 0
    assert "o/xberg yet; skipped" in capsys.readouterr().out
    assert not (tmp_path / "compat" / "xberg").exists()
    assert 'href="xberg/"' not in (tmp_path / "compat" / "index.html").read_text(encoding="utf-8")


def test_a_required_fork_with_no_wheels_fails(monkeypatch, tmp_path, capsys):
    rc = _run(monkeypatch, tmp_path, {"o/lancedb": []}, "--repo", "o/lancedb")
    assert rc == 1
    assert "no compat wheels found in o/lancedb" in capsys.readouterr().err
    assert not (tmp_path / "compat").exists()


def test_no_repo_is_a_usage_error(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, tmp_path, {})
    assert exc.value.code == 2


class _Run:
    def __init__(self, out: str) -> None:
        self.out = out

    def __call__(self, *_a, **_k) -> str:
        return self.out


def test_fetch_wheels_reads_the_api_digest(monkeypatch):
    line = '{"name": "w.whl", "url": "https://x/w.whl", "digest": "sha256:' + _SHA + '"}'
    monkeypatch.setattr(bci.subprocess, "check_output", _Run(line + "\n\n"))
    assert bci.fetch_wheels("o/r") == [("w.whl", "https://x/w.whl", _SHA)]


def test_the_release_filter_takes_compat_tags_and_nothing_else():
    tags = (
        "v1.13.0+compat",
        "lancedb-v0.40.0+compat",
        "v1.4.0+lilbee",
        "v1.13.0+compat-test",
        "v1.10.3+lilbee.06089bd",
        "xberg-v1.1.0.dev1+lilbee1",
        "v1.4.0",
    )
    asset = {"browser_download_url": "u", "digest": "d"}
    releases = [{"tag_name": t, "assets": [{"name": f"{t}.whl", **asset}]} for t in tags]
    out = subprocess.run(
        ["jq", "-c", bci._JQ],
        input=json.dumps(releases),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    picked = [json.loads(line)["name"] for line in out.splitlines()]
    assert picked == ["v1.13.0+compat.whl", "lancedb-v0.40.0+compat.whl"]


def test_fetch_wheels_refuses_an_asset_with_no_digest(monkeypatch):
    line = '{"name": "w.whl", "url": "https://x/w.whl", "digest": null}'
    monkeypatch.setattr(bci.subprocess, "check_output", _Run(line))
    with pytest.raises(bci.FetchError, match=r"w\.whl has no sha256 digest"):
        bci.fetch_wheels("o/r")


def _gh_fails(*_a, **_k):
    raise subprocess.CalledProcessError(1, ["gh", "api"])


def test_fetch_wheels_names_the_repo_whose_releases_cannot_be_read(monkeypatch):
    monkeypatch.setattr(bci.subprocess, "check_output", _gh_fails)
    with pytest.raises(bci.FetchError, match=r"cannot read the releases of o/r"):
        bci.fetch_wheels("o/r")


def _fetch_failing_for(repo: str, by_repo: dict[str, list[str]]):
    def fetch(name: str):
        if name == repo:
            raise bci.FetchError(f"cannot read the releases of {name}")
        return [_wheel(n) for n in by_repo[name]]

    return fetch


def test_an_optional_fork_that_cannot_be_read_is_skipped_with_a_warning(
    monkeypatch, tmp_path, capsys
):
    by_repo = {"o/lancedb": [_LANCEDB]}
    monkeypatch.setattr(bci, "fetch_wheels", _fetch_failing_for("o/crawlberg", by_repo))
    argv = ["--repo", "o/lancedb", "--optional-repo", "o/crawlberg"]
    monkeypatch.setattr("sys.argv", ["build_compat_index.py", str(tmp_path), *argv])

    assert bci.main() == 0
    assert "warning: cannot read the releases of o/crawlberg; skipped" in capsys.readouterr().err
    assert (tmp_path / "compat" / "lancedb" / "index.html").exists()
    assert not (tmp_path / "compat" / "crawlberg").exists()


def test_a_required_fork_that_cannot_be_read_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(bci, "fetch_wheels", _fetch_failing_for("o/lancedb", {}))
    monkeypatch.setattr("sys.argv", ["build_compat_index.py", str(tmp_path), "--repo", "o/lancedb"])

    assert bci.main() == 1
    assert "cannot read the releases of o/lancedb" in capsys.readouterr().err
    assert not (tmp_path / "compat").exists()
