"""The scripts that build the mistral.rs pack read archives and reports and refuse anything else."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "wheel-build"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repack = _load("repack_mistralrs")
rust_licences = _load("rust_licences")
rust_notices = _load("rust_notices")
cuda_notice = _load("write_cuda_notice")
nccl_notice = _load("write_nccl_notice")


def _tar(path: Path, entries: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_repack_places_the_tool_and_the_libraries(tmp_path: Path) -> None:
    archive = _tar(
        tmp_path / "a.tar.gz",
        {"./mistralrs": b"bin", "./lib/libcudart.so.12": b"rt", "./lib/libnccl.so.2": b"nc"},
    )
    names = repack.repack(archive, tmp_path / "bin")
    assert names == ["libcudart.so.12", "libnccl.so.2"]
    assert (tmp_path / "bin" / "mistralrs").read_bytes() == b"bin"
    assert (tmp_path / "bin" / "mistralrs").stat().st_mode & 0o111
    assert (tmp_path / "bin" / "lib" / "libnccl.so.2").read_bytes() == b"nc"


def test_repack_leaves_the_driver_library_out(tmp_path: Path) -> None:
    archive = _tar(
        tmp_path / "a.tar.gz",
        {"mistralrs": b"bin", "lib/libcuda.so.1": b"drv", "lib/libcudart.so.12": b"rt"},
    )
    assert repack.repack(archive, tmp_path / "bin") == ["libcudart.so.12"]
    assert not (tmp_path / "bin" / "lib" / "libcuda.so.1").exists()


def test_repack_reads_a_zip_with_the_exe(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("mistralrs.exe", b"exe")
    assert repack.repack(archive, tmp_path / "bin") == []
    assert (tmp_path / "bin" / "mistralrs.exe").read_bytes() == b"exe"
    assert not (tmp_path / "bin" / "lib").exists()


def test_repack_refuses_an_unknown_entry(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"mistralrs": b"bin", "extra.sh": b"x"})
    with pytest.raises(SystemExit, match=r"extra\.sh is not the tool or a library"):
        repack.repack(archive, tmp_path / "bin")


def test_repack_refuses_an_archive_without_the_tool(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"lib/libcudart.so.12": b"rt"})
    with pytest.raises(SystemExit, match="expected one tool, found 0"):
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


def test_cuda_eula_text_stops_on_a_short_page() -> None:
    with pytest.raises(SystemExit, match="looks incomplete"):
        cuda_notice.eula_text("<p>Attachment A</p>", "https://x.test")


def test_cuda_eula_text_keeps_the_body_and_skips_scripts() -> None:
    page = "<script>var x</script><p>Attachment A</p><p>" + "word " * 5000 + "</p>"
    body = cuda_notice.eula_text(page, "https://x.test")
    assert body.startswith("Attachment A") and "var x" not in body


def test_cuda_notice_names_the_files_and_the_version() -> None:
    text = cuda_notice.notice("12.8.1", "cudart.", "BODY", "https://x.test/eula")
    assert "NVIDIA CUDA 12.8 runtime files: cudart." in text
    assert "version 12.8.1, from https://x.test/eula" in text and text.endswith("BODY\n")


def test_nccl_version_is_read_from_the_library(tmp_path: Path) -> None:
    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"\x00junk NCCL version 2.25.1+cuda12.8\x00")
    assert nccl_notice.nccl_version(library) == "2.25.1"


def test_nccl_version_refuses_a_library_without_one(tmp_path: Path) -> None:
    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"nothing")
    with pytest.raises(SystemExit, match="does not name its NCCL version"):
        nccl_notice.nccl_version(library)


def test_nccl_notice_names_the_tag() -> None:
    text = nccl_notice.notice("libnccl.so.2", "2.25.1", " LICENCE TEXT ")
    assert "bin/lib/libnccl.so.2 is NCCL 2.25.1" in text and "(tag v2.25.1-1)" in text
    assert text.endswith("LICENCE TEXT\n")


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
    assert "https://static.crates.io/crates/selectors/selectors-0.26.0.crate" in text
    assert "sha256: abc123" in text
    assert text.rstrip().endswith("mpl text")
    assert "serde" not in text


def test_mpl_notice_is_absent_without_an_mpl_crate() -> None:
    assert rust_notices.mpl_notice({"licenses": [_entry("MIT", "mit")]}, {}) is None


def test_mpl_notice_refuses_a_crate_outside_the_registry() -> None:
    with pytest.raises(SystemExit, match="from-git 1.0.0 is not a crates.io package"):
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
    assert found == ["NOTICE", "a/NOTICE.md", "a/third_party/x/LICENSE-BSD"]


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
        "NOTICE-MPL-2.0",
        "NOTICE-OpenSSL",
        "mistral.rs__sub__NOTICE",
        "aws-lc-sys-1.0.0__NOTICE",
    ]
    assert "Eric Young" in (tree / "NOTICE-OpenSSL").read_text(encoding="utf-8")
    vendored = tree / "THIRD-PARTY" / "vendored"
    assert (vendored / "aws-lc-sys-1.0.0__NOTICE").read_text(encoding="utf-8") == "crate notice"
    assert (vendored / "mistral.rs__sub__NOTICE").read_text(encoding="utf-8") == "checkout notice"
