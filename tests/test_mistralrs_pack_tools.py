"""The scripts that build the mistral.rs pack read archives and reports and refuse anything else."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from tests._pack_workflow import pack_workflow

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "wheel-build"
PACKAGING = Path(__file__).resolve().parents[1] / "packaging" / "engine-wheel-mistral-rs"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repack = _load("repack_mistralrs")
rust_licences = _load("rust_licences")
rust_notices = _load("rust_notices")


def _tar(path: Path, entries: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_repack_places_the_tool(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"./mistralrs": b"bin"})
    repack.repack(archive, tmp_path / "bin")
    assert (tmp_path / "bin" / "mistralrs").read_bytes() == b"bin"
    # Windows has no execute bit: there, every file that exists passes.
    assert os.access(tmp_path / "bin" / "mistralrs", os.X_OK)


def test_repack_refuses_a_shared_library(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"mistralrs": b"bin", "lib/libcudart.so.12": b"rt"})
    with pytest.raises(SystemExit, match=r"libcudart\.so\.12 is not the tool"):
        repack.repack(archive, tmp_path / "bin")
    assert not (tmp_path / "bin").exists()


def test_repack_reads_a_zip_with_the_exe(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("mistralrs.exe", b"exe")
    repack.repack(archive, tmp_path / "bin")
    assert (tmp_path / "bin" / "mistralrs.exe").read_bytes() == b"exe"


def test_repack_refuses_an_unknown_entry(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"mistralrs": b"bin", "extra.sh": b"x"})
    with pytest.raises(SystemExit, match=r"extra\.sh is not the tool"):
        repack.repack(archive, tmp_path / "bin")


def test_repack_refuses_an_archive_without_the_tool(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {})
    with pytest.raises(SystemExit, match="expected one tool, found 0"):
        repack.repack(archive, tmp_path / "bin")


def test_repack_refuses_an_archive_with_two_tools(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"mistralrs": b"a", "mistralrs.exe": b"b"})
    with pytest.raises(SystemExit, match="expected one tool, found 2"):
        repack.repack(archive, tmp_path / "bin")


def test_repack_refuses_a_link(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("mistralrs")
        info.type = tarfile.SYMTYPE
        info.linkname = "other"
        tar.addfile(info)
    with pytest.raises(SystemExit, match="mistralrs is a link"):
        repack.repack(archive, tmp_path / "bin")


def _entry(licence_id: str, text: str, crate: str = "serde") -> dict:
    return {
        "id": licence_id,
        "name": f"{licence_id} licence",
        "text": text,
        "used_by": [
            {"crate": {"name": crate, "version": "1.0.0", "repository": "https://x.test/r"}}
        ],
    }


def test_rust_licences_write_a_file_for_each_text(tmp_path: Path) -> None:
    report = {
        "licenses": [
            _entry("MIT", "text one"),
            _entry("MIT", "text two", "rand"),
            _entry("ISC", "isc"),
        ]
    }
    assert rust_licences.write_licences(report, tmp_path) == ["ISC", "MIT"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "ISC-001.txt",
        "MIT-001.txt",
        "MIT-002.txt",
    ]
    first = (tmp_path / "MIT-001.txt").read_text(encoding="utf-8")
    assert "serde 1.0.0 (https://x.test/r)" in first and first.rstrip().endswith("text one")


def test_rust_licences_refuse_an_empty_report(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="lists no licence text"):
        rust_licences.write_licences({"licenses": []}, tmp_path)


def test_rust_licences_refuse_an_empty_text(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="MIT text of serde is empty"):
        rust_licences.write_licences({"licenses": [_entry("MIT", "  ")]}, tmp_path)


_LOCK = """
[[package]]
name = "selectors"
version = "0.26.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "abc123"

[[package]]
name = "from-git"
version = "1.0.0"
source = "git+https://example.test/r#deadbeef"
"""


def _mpl_report(crate: str, version: str) -> dict:
    entry = _entry("MPL-2.0", "mpl text", crate)
    entry["used_by"][0]["crate"]["version"] = version
    return {"licenses": [entry, _entry("MIT", "mit")]}


def test_lock_checksums_keep_only_registry_packages() -> None:
    assert rust_notices.lock_checksums(_LOCK) == {("selectors", "0.26.0"): "abc123"}


def test_mpl_notice_names_the_exact_source_and_checksum() -> None:
    text = rust_notices.mpl_notice(
        _mpl_report("selectors", "0.26.0"), rust_notices.lock_checksums(_LOCK)
    )
    assert text is not None
    assert text.splitlines() == [
        "https://static.crates.io/crates/selectors/selectors-0.26.0.crate  abc123"
    ]
    assert "serde" not in text


def test_mpl_notice_is_absent_without_an_mpl_crate() -> None:
    assert rust_notices.mpl_notice({"licenses": [_entry("MIT", "mit")]}, {}) is None


def test_mpl_notice_refuses_a_crate_outside_the_registry() -> None:
    with pytest.raises(SystemExit, match=r"from-git 1\.0\.0 is not a crates\.io package"):
        rust_notices.mpl_notice(
            _mpl_report("from-git", "1.0.0"), rust_notices.lock_checksums(_LOCK)
        )


def test_vendored_files_take_notices_and_third_party_licences(tmp_path: Path) -> None:
    names = (
        "LICENSE",
        "NOTICE",
        "a/NOTICE.md",
        "a/third_party/x/LICENSE-BSD",
        "a/third_party/x/main.rs",
        "target/NOTICE",
    )
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("t", encoding="utf-8")
    found = [p.relative_to(tmp_path).as_posix() for p in rust_notices.vendored_files(tmp_path)]
    # The function sorts Path objects; Windows orders those without regard to case, so the
    # expectation sorts Path objects too.
    expected = ["NOTICE", "a/NOTICE.md", "a/third_party/x/LICENSE-BSD"]
    ordered = sorted(tmp_path / n for n in expected)
    assert found == [p.relative_to(tmp_path).as_posix() for p in ordered]


def test_write_notices_writes_each_file_into_the_tree(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "sub").mkdir(parents=True)
    (checkout / "sub" / "NOTICE").write_text("checkout notice", encoding="utf-8")
    crate = tmp_path / "registry" / "index-1" / "aws-lc-sys-1.0.0"
    crate.mkdir(parents=True)
    (crate / "NOTICE").write_text("crate notice", encoding="utf-8")
    tree = tmp_path / "tree"
    tree.mkdir()
    report = _mpl_report("selectors", "0.26.0")
    report["licenses"].append(_entry("OpenSSL", "ssl text", "aws-lc-sys"))
    names = rust_notices.write_notices(report, _LOCK, checkout, tmp_path / "registry", tree)
    assert names == [
        "SOURCE-MPL-2.0",
        "NOTICE-OpenSSL",
        "mistral.rs__sub__NOTICE",
        "aws-lc-sys-1.0.0__NOTICE",
    ]
    assert "Eric Young" in (tree / "NOTICE-OpenSSL").read_text(encoding="utf-8")
    vendored = tree / "THIRD-PARTY" / "vendored"
    assert (vendored / "aws-lc-sys-1.0.0__NOTICE").read_text(encoding="utf-8") == "crate notice"
    assert (vendored / "mistral.rs__sub__NOTICE").read_text(encoding="utf-8") == "checkout notice"


def _bash() -> str:
    """The bash on PATH, which the pack job runs on every OS.

    On Windows a bare ``bash`` starts the WSL launcher in System32 before PATH is searched.
    """
    found = shutil.which("bash")
    assert found is not None
    return found


def _mistralrs_job() -> dict:
    return yaml.safe_load(pack_workflow().read_text(encoding="utf-8"))["jobs"]["mistralrs"]


def test_the_pack_job_writes_the_notices_before_it_builds_the_wheel() -> None:
    steps = [step.get("run", "") for step in _mistralrs_job()["steps"]]
    writer = next(i for i, run in enumerate(steps) if "rust_notices.py" in run)
    builder = next(i for i, run in enumerate(steps) if "python -m build" in run)
    assert writer < builder


def test_the_pack_job_checks_the_notices_in_the_built_wheel() -> None:
    check = next(
        step["run"]
        for step in _mistralrs_job()["steps"]
        if step.get("name") == "Check the licences in the installed wheel"
    )
    assert "THIRD-PARTY/vendored/" in check
    assert "NOTICE-OpenSSL" in check and "SOURCE-MPL-2.0" in check


def _run_notice_loop(tmp_path: Path, listing: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the OpenSSL and MPL notice loop of the wheel check against a wheel listing."""
    check = next(
        step["run"]
        for step in _mistralrs_job()["steps"]
        if step.get("name") == "Check the licences in the installed wheel"
    )
    loop = "for pair in" + check.split("for pair in", 1)[1].split("\ndone", 1)[0] + "\ndone"
    (tmp_path / "crate-licences.txt").write_text(
        "MPL-2.0\nOpenSSL\n", encoding="utf-8", newline="\n"
    )
    listed = "\n".join(listing) + "\n"
    (tmp_path / "wheel-listing.txt").write_text(listed, encoding="utf-8", newline="\n")
    script = tmp_path / "check_notices.sh"
    script.write_text(f"set -euo pipefail\n{loop}\n", encoding="utf-8", newline="\n")
    return subprocess.run(
        [_bash(), str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


_HELD = [
    "x.dist-info/licenses/SOURCE-MPL-2.0 12",
    "x.dist-info/licenses/NOTICE-OpenSSL 12",
]


def test_the_pack_job_passes_a_wheel_that_holds_both_notices(tmp_path: Path) -> None:
    assert _run_notice_loop(tmp_path, _HELD).returncode == 0


@pytest.mark.parametrize("missing", ["SOURCE-MPL-2.0", "NOTICE-OpenSSL"])
def test_the_pack_job_fails_a_wheel_that_lacks_a_notice(tmp_path: Path, missing: str) -> None:
    listing = [line for line in _HELD if missing not in line]
    bad = _run_notice_loop(tmp_path, listing)
    assert bad.returncode == 1
    assert f"wheel lacks {missing}" in bad.stdout


@pytest.mark.parametrize("name", ["LICENSE-mistral.rs", "NOTICE-OpenSSL", "SOURCE-MPL-2.0"])
def test_the_pack_job_refuses_an_empty_licence_file(tmp_path: Path, name: str) -> None:
    step = next(
        step["run"]
        for step in _mistralrs_job()["steps"]
        if step.get("name") == "Name every licence in the wheel expression"
    )
    loop = "for f in" + step.split("for f in", 1)[1].split("\ndone", 1)[0] + "\ndone"
    script = tmp_path / "check_empty.sh"
    script.write_text(f"set -euo pipefail\n{loop}\n", encoding="utf-8", newline="\n")
    tree = tmp_path / "tree"
    tree.mkdir()
    for held in ("LICENSE-mistral.rs", "NOTICE-OpenSSL", "SOURCE-MPL-2.0"):
        (tree / held).write_text("text", encoding="utf-8")
    env = {**os.environ, "WHEEL_TREE": str(tree)}
    run = subprocess.run
    ok = run([_bash(), str(script)], env=env, capture_output=True, text=True, encoding="utf-8")
    assert ok.returncode == 0
    (tree / name).write_text("", encoding="utf-8")
    bad = run([_bash(), str(script)], env=env, capture_output=True, text=True, encoding="utf-8")
    assert bad.returncode == 1
    # The step joins the folder and the name with a slash on every OS.
    assert f"empty licence file {tree}/{name}" in bad.stdout


def _wheel_listing(tmp_path: Path) -> set[str]:
    """Build the wheel as the pack job does and list the files under dist-info/licenses."""
    tree = tmp_path / "wheel-tree"
    shutil.copytree(PACKAGING, tree)
    report = _mpl_report("selectors", "0.26.0")
    report["licenses"].append(_entry("OpenSSL", "ssl text", "aws-lc-sys"))
    checkout = tmp_path / "checkout"
    (checkout / "sub").mkdir(parents=True)
    (checkout / "sub" / "NOTICE").write_text("checkout notice", encoding="utf-8")
    names = rust_notices.write_notices(report, _LOCK, checkout, tmp_path / "registry", tree)
    assert {"SOURCE-MPL-2.0", "NOTICE-OpenSSL"} <= set(names)
    (tree / "LICENSE-mistral.rs").write_text("mistral.rs licence", encoding="utf-8")
    crates = tree / "THIRD-PARTY" / "mistral.rs"
    crates.mkdir(parents=True)
    (crates / "MIT.txt").write_text("mit", encoding="utf-8")
    step = next(
        step["run"] for step in _mistralrs_job()["steps"] if step.get("name") == "Build the wheel"
    )
    script = tmp_path / "set_metadata.py"
    script.write_text(step.split("<<'PY'\n")[1].split("\nPY")[0], encoding="utf-8")
    env = {**os.environ, "LICENSE_EXPRESSION": "MIT AND MPL-2.0 AND OpenSSL"}
    subprocess.run([sys.executable, "-I", str(script), "0.9.5"], cwd=tree, env=env, check=True)
    uv = shutil.which("uv")
    assert uv is not None
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path / "dist")],
        cwd=tree,
        check=True,
        capture_output=True,
    )
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        return {
            name.split(".dist-info/licenses/")[1]
            for name in archive.namelist()
            if ".dist-info/licenses/" in name
        }


def test_the_built_wheel_holds_every_licence_file_the_job_writes(tmp_path: Path) -> None:
    held = _wheel_listing(tmp_path)
    assert {
        "SOURCE-MPL-2.0",
        "NOTICE-OpenSSL",
        "LICENSE-mistral.rs",
        "THIRD-PARTY/mistral.rs/MIT.txt",
        "THIRD-PARTY/vendored/mistral.rs__sub__NOTICE",
    } <= held


def _wrap_archive(tmp_path: Path, run: str, archive: Path, checkout_time: int) -> str:
    """Wrap *archive* into a copy of the wheel project, the way the job does, and hash the wheel."""
    tree = tmp_path / run / "project"
    shutil.copytree(
        PACKAGING, tree, ignore=shutil.ignore_patterns("bin", "__pycache__", "*.egg-info")
    )
    packed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(TOOLS / "repack_mistralrs.py"),
            str(archive),
            str(tree / "lilbee_engine_mistral_rs" / "bin"),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    for path in (tree, *tree.rglob("*")):
        os.utime(path, (checkout_time, checkout_time))
    env = dict(line.split("=", 1) for line in packed.stdout.splitlines())
    uv = shutil.which("uv")
    assert uv is not None
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path / run / "dist")],
        cwd=tree,
        env={**os.environ, **env},
        check=True,
        capture_output=True,
    )
    (wheel,) = (tmp_path / run / "dist").glob("*.whl")
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


def test_wrapping_one_archive_twice_gives_one_wheel(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("mistralrs")
        info.size = 3
        info.mtime = 1_700_000_000
        tar.addfile(info, io.BytesIO(b"bin"))
    first = _wrap_archive(tmp_path, "first", archive, 1_750_000_000)
    second = _wrap_archive(tmp_path, "second", archive, 1_760_000_000)
    assert first == second


@pytest.mark.parametrize(
    ("archive_name", "member_time"),
    [("a.tar.gz", 1_700_000_000), ("a.zip", 1_700_000_000)],
)
def test_repack_reports_the_time_of_the_tool_in_the_archive(
    tmp_path: Path, archive_name: str, member_time: int
) -> None:
    archive = tmp_path / archive_name
    if archive_name.endswith(".zip"):
        stamp = time.gmtime(member_time)
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(zipfile.ZipInfo("mistralrs.exe", stamp[:6]), b"exe")
    else:
        with tarfile.open(archive, "w:gz") as tar:
            info = tarfile.TarInfo("./mistralrs")
            info.size = 3
            info.mtime = member_time
            tar.addfile(info, io.BytesIO(b"bin"))
    assert repack.repack(archive, tmp_path / "bin") == member_time


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the step is a POSIX shell script and the shim is a symlink",
)
def test_the_repack_step_writes_the_archive_time_to_the_job_environment(tmp_path: Path) -> None:
    """Run the step as the job does: the later steps read the time from the job environment."""
    run = next(
        step["run"]
        for step in _mistralrs_job()["steps"]
        if step.get("name") == "Repack the upstream archive into the wheel tree"
    )
    (tmp_path / "upstream").mkdir()
    with tarfile.open(tmp_path / "upstream" / "a.tar.gz", "w:gz") as tar:
        info = tarfile.TarInfo("mistralrs")
        info.size = 3
        info.mtime = 1_700_000_000
        tar.addfile(info, io.BytesIO(b"bin"))
    shutil.copytree(TOOLS, tmp_path / "tools" / "wheel-build")
    shims = tmp_path / "shims"
    shims.mkdir()
    (shims / "python").symlink_to(sys.executable)
    script = tmp_path / "step.sh"
    script.write_text(run, encoding="utf-8")
    job_env = tmp_path / "github-env"
    job_env.touch()
    done = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{shims}{os.pathsep}{os.environ['PATH']}",
            "ASSET": "a.tar.gz",
            "WHEEL_TREE": str(tmp_path / "tree"),
            "GITHUB_ENV": str(job_env),
        },
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert done.returncode == 0, done.stderr
    assert job_env.read_text(encoding="utf-8") == "SOURCE_DATE_EPOCH=1700000000\n"


def test_the_pack_job_has_no_cuda_step() -> None:
    job = _mistralrs_job()
    text = yaml.safe_dump(job)
    assert job["steps"]
    for word in ("CUDA", "cuda", "NCCL", "nccl", "ldd"):
        assert word not in text


def test_the_pack_matrix_holds_no_cuda_cell() -> None:
    cells = _mistralrs_job()["strategy"]["matrix"]["include"]
    assert {cell["cell"] for cell in cells} == {
        "mistralrs-linux-x86_64-cpu",
        "mistralrs-windows-x86_64-cpu",
        "mistralrs-macos-arm64-metal",
    }
