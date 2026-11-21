"""The engine and compat indexes are built by dumb-pypi, with a sha256 on every link."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_wheel_indexes.py"
BASE = "https://releases.test/download"
RAW = "lilbee_engine_llama_cpp-0.6.90-py3-none-linux_x86_64.whl"


@pytest.fixture(scope="module")
def bwi() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_wheel_indexes", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact(root: Path, name: str, wheel: str = RAW, body: bytes = b"wheel") -> str:
    (root / name).mkdir(parents=True)
    (root / name / wheel).write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _links(page: Path) -> list[str]:
    return re.findall(r'href="([^"]+)"', page.read_text(encoding="utf-8"))


@pytest.mark.parametrize("backend", ["metal", "vulkan", "cu124", "cu125", "cu121", "cpu", "rocm"])
def test_every_backend_wheel_gets_the_build_tag(bwi: ModuleType, backend: str) -> None:
    name = bwi.release_name(RAW, backend)
    assert name == f"lilbee_engine_llama_cpp-0.6.90-1.{backend}-py3-none-linux_x86_64.whl"


def test_a_wheel_that_already_has_a_build_tag_is_refused(bwi: ModuleType) -> None:
    with pytest.raises(ValueError, match="no-buildtag layout"):
        bwi.release_name("p-1-1.cpu-py3-none-any.whl", "cpu")


def test_an_artifact_directory_names_its_backend(bwi: ModuleType) -> None:
    assert bwi.backend_of("wheel-multigpu-linux-cu125") == "cu125"
    assert bwi.backend_of("wheel-default-linux") is None


def test_each_backend_gets_an_index_that_links_its_own_wheel_with_the_sha256(
    bwi: ModuleType, tmp_path: Path
) -> None:
    artifacts, site = tmp_path / "a", tmp_path / "site"
    artifacts.mkdir()
    cpu = _artifact(artifacts, "wheel-multigpu-linux-cpu", body=b"cpu bytes")
    cuda = _artifact(artifacts, "wheel-multigpu-linux-cu125", body=b"cuda bytes")
    assert bwi.main(["engines", str(artifacts), str(site), "--release-base-url", BASE + "/"]) == 0
    for backend, digest in (("cpu", cpu), ("cu125", cuda)):
        asset = f"lilbee_engine_llama_cpp-0.6.90-1.{backend}-py3-none-linux_x86_64.whl"
        assert _links(site / backend / "lilbee-engine-llama-cpp" / "index.html") == [
            f"{BASE}/v0.6.90/{asset}#sha256={digest}"
        ]
        assert _links(site / backend / "index.html") == ["lilbee-engine-llama-cpp/index.html"]
    assert cpu != cuda


def test_no_link_keeps_the_placeholder_prefix(bwi: ModuleType, tmp_path: Path) -> None:
    artifacts, site = tmp_path / "a", tmp_path / "site"
    artifacts.mkdir()
    _artifact(artifacts, "wheel-multigpu-linux-cpu")
    bwi.main(["engines", str(artifacts), str(site)])
    pages = [p for p in site.rglob("*.html")]
    assert pages
    assert not any("packages.invalid" in p.read_text(encoding="utf-8") for p in pages)
    assert _links(site / "cpu" / "lilbee-engine-llama-cpp" / "index.html")[0].startswith(
        "https://github.com/tobocop2/lilbee/releases/download/v0.6.90/"
    )


def test_a_missing_input_directory_is_an_error(
    bwi: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert bwi.main(["engines", str(tmp_path / "none"), str(tmp_path / "site")]) == 1
    assert "input directory does not exist" in capsys.readouterr().err


def test_an_input_directory_without_wheels_writes_nothing(
    bwi: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a" / "wheel-default-linux").mkdir(parents=True)
    assert bwi.main(["engines", str(tmp_path / "a"), str(tmp_path / "site")]) == 0
    assert "nothing to index" in capsys.readouterr().err
    assert not (tmp_path / "site").exists()


def test_a_wheel_with_an_unparseable_name_is_skipped(
    bwi: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifacts = tmp_path / "a"
    artifacts.mkdir()
    (artifacts / "wheel-multigpu-linux-cpu").mkdir()
    (artifacts / "wheel-multigpu-linux-cpu" / "lilbee_engine_llama_cpp-.whl").write_bytes(b"x")
    assert bwi.engine_wheels(artifacts, BASE) == {}
    assert "skipping unparseable wheel filename" in capsys.readouterr().err


def test_compat_wheels_of_several_releases_share_one_project_page(
    bwi: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = "lancedb-0.33.0+compat-cp39-abi3-manylinux_2_28_x86_64.whl"
    new = "lancedb-0.40.0+compat-cp39-abi3-manylinux_2_28_x86_64.whl"
    wheels = [
        bwi.Wheel(old, "a" * 64, f"https://gh.test/r/lancedb-v0.33.0+compat/{old}"),
        bwi.Wheel(new, "b" * 64, f"https://gh.test/r/lancedb-v0.40.0+compat/{new}"),
    ]
    monkeypatch.setattr(bwi, "compat_wheels", lambda _repo: wheels)
    assert bwi.main(["compat", str(tmp_path)]) == 0
    assert sorted(_links(tmp_path / "compat" / "lancedb" / "index.html")) == [
        f"https://gh.test/r/lancedb-v0.33.0+compat/{old}#sha256={'a' * 64}",
        f"https://gh.test/r/lancedb-v0.40.0+compat/{new}#sha256={'b' * 64}",
    ]
    assert _links(tmp_path / "compat" / "index.html") == ["lancedb/index.html"]


def test_no_compat_wheels_is_an_error(
    bwi: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(bwi, "compat_wheels", lambda _repo: [])
    assert bwi.main(["compat", str(tmp_path)]) == 1
    assert "no compat wheels found" in capsys.readouterr().err
    assert not (tmp_path / "compat").exists()


class _Run:
    def __init__(self, bwi: ModuleType, out: str) -> None:
        self.out = out
        self.bwi = bwi

    def __call__(self, *_args: object, **_kw: object) -> str:
        return self.out


def test_the_release_api_listing_becomes_wheels_with_the_digest(
    bwi: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = '{"name": "a.whl", "url": "https://gh.test/a.whl", "digest": "sha256:%s"}\n\n' % (
        "c" * 64
    )
    monkeypatch.setattr(bwi.subprocess, "check_output", _Run(bwi, row))
    assert bwi.compat_wheels("o/r") == [bwi.Wheel("a.whl", "c" * 64, "https://gh.test/a.whl")]


def test_an_asset_without_a_digest_stops_the_build(
    bwi: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = '{"name": "a.whl", "url": "https://gh.test/a.whl", "digest": null}\n'
    monkeypatch.setattr(bwi.subprocess, "check_output", _Run(bwi, row))
    with pytest.raises(SystemExit, match=r"a\.whl has no sha256 digest"):
        bwi.compat_wheels("o/r")


def test_a_link_that_dumb_pypi_wrote_for_an_unknown_file_stops_the_build(
    bwi: ModuleType, tmp_path: Path
) -> None:
    simple = tmp_path / "p"
    simple.mkdir()
    (simple / "index.html").write_text(
        '<a href="https://packages.invalid/x.whl#">x</a>', encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="did not point"):
        bwi._point_links_at_assets(tmp_path, [])


def test_no_project_page_stops_the_build(bwi: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="did not point"):
        bwi._point_links_at_assets(tmp_path, [])


_ROOT = Path(__file__).resolve().parents[1]
_TOOL_CALL = re.compile(r"\b(?:tools|scripts)/[\w/]+\.py\b")


def _tool_problems(root: Path) -> list[str]:
    """Name every tool the Makefile or the Pages workflow calls that is missing, or not called."""
    problems: list[str] = []
    for rel in ("Makefile", ".github/workflows/pages.yml"):
        text = (root / rel).read_text(encoding="utf-8")
        problems += [
            f"{rel} calls {tool}, which does not exist"
            for tool in sorted(set(_TOOL_CALL.findall(text)))
            if not (root / tool).is_file()
        ]
    pages = (root / ".github/workflows/pages.yml").read_text(encoding="utf-8")
    if "tools/build_wheel_indexes.py" not in pages:
        problems.append("pages.yml does not call tools/build_wheel_indexes.py")
    return problems


def test_the_makefile_and_pages_workflow_call_only_tools_that_exist() -> None:
    assert _tool_problems(_ROOT) == []


@pytest.mark.parametrize("old", ["build_pep503_indexes", "build_compat_index"])
def test_a_call_to_a_deleted_index_script_is_found(old: str, tmp_path: Path) -> None:
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "Makefile").write_text(
        "all:\n\tpython tools/build_wheel_indexes.py\n", encoding="utf-8"
    )
    pages = tmp_path / ".github/workflows/pages.yml"
    pages.write_text(f"run: python tools/{old}.py x\n", encoding="utf-8")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/build_wheel_indexes.py").write_text("", encoding="utf-8")
    assert _tool_problems(tmp_path) == [
        f".github/workflows/pages.yml calls tools/{old}.py, which does not exist",
        "pages.yml does not call tools/build_wheel_indexes.py",
    ]
