"""Tests for scripts/vendor_llama_cpp.py."""

from __future__ import annotations

import base64
import hashlib
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts/ to sys.path so we can import the vendor module.
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import vendor_llama_cpp as vendor  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers to create synthetic wheels
# ---------------------------------------------------------------------------


def _make_wheel(
    path: Path,
    name: str,
    version: str,
    python: str = "py3",
    abi: str = "none",
    plat: str = "any",
    pkg_dir: str | None = None,
    pkg_files: dict[str, bytes] | None = None,
    metadata_extra: str = "",
) -> Path:
    """Create a minimal valid wheel zip at *path*."""
    tag = f"{python}-{abi}-{plat}"
    dist_info = f"{name}-{version}.dist-info"
    filename = f"{name}-{version}-{tag}.whl"
    path.mkdir(parents=True, exist_ok=True)
    whl_path = path / filename

    records: list[str] = []

    def _record_entry(arcname: str, data: bytes) -> str:
        digest = hashlib.sha256(data).digest()
        h = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return f"{arcname},sha256={h},{len(data)}"

    with zipfile.ZipFile(whl_path, "w", zipfile.ZIP_DEFLATED) as zf:
        meta = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n{metadata_extra}"
        meta_bytes = meta.encode()
        arcname = f"{dist_info}/METADATA"
        zf.writestr(arcname, meta_bytes)
        records.append(_record_entry(arcname, meta_bytes))

        wheel_content = f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: {tag}\n"
        wheel_bytes = wheel_content.encode()
        arcname = f"{dist_info}/WHEEL"
        zf.writestr(arcname, wheel_bytes)
        records.append(_record_entry(arcname, wheel_bytes))

        if pkg_dir and pkg_files:
            for relname, data in pkg_files.items():
                arcname = f"{pkg_dir}/{relname}"
                zf.writestr(arcname, data)
                records.append(_record_entry(arcname, data))

        record_arcname = f"{dist_info}/RECORD"
        records.append(f"{record_arcname},,")
        record_bytes = ("\n".join(records) + "\n").encode()
        zf.writestr(record_arcname, record_bytes)

    return whl_path


def _make_lilbee_wheel(tmp: Path, *, version: str = "0.6.0") -> Path:
    """Create a synthetic lilbee wheel with a dummy module file."""
    metadata_extra = (
        "Requires-Dist: lancedb\nRequires-Dist: llama-cpp-python\nRequires-Dist: tiktoken\n"
    )
    return _make_wheel(
        tmp,
        name="lilbee",
        version=version,
        pkg_dir="lilbee",
        pkg_files={
            "__init__.py": b"# lilbee\n",
            "config.py": b"CFG = True\n",
        },
        metadata_extra=metadata_extra,
    )


def _make_llama_wheel(tmp: Path, *, version: str = "0.3.18") -> Path:
    """Create a synthetic llama-cpp-python wheel with sample files."""
    return _make_wheel(
        tmp,
        name="llama_cpp_python",
        version=version,
        python="cp312",
        abi="cp312",
        plat="macosx_11_0_arm64",
        pkg_dir="llama_cpp",
        pkg_files={
            "__init__.py": b"# llama_cpp\n",
            "llama.py": b"class Llama: pass\n",
            "libllama.dylib": b"\x00" * 64,
        },
    )


# ---------------------------------------------------------------------------
# 1. test_detect_platform_tags
# ---------------------------------------------------------------------------


class TestDetectPlatformTags:
    """Verify _detect_tags for each OS."""

    def test_linux_x86_64(self) -> None:
        dl, whl = vendor._detect_tags("Linux", "x86_64")
        assert dl == "manylinux2014_x86_64"
        assert whl == "manylinux_2_17_x86_64.manylinux2014_x86_64"

    def test_linux_aarch64(self) -> None:
        dl, whl = vendor._detect_tags("Linux", "aarch64")
        assert dl == "manylinux2014_aarch64"
        assert whl == "manylinux_2_17_aarch64.manylinux2014_aarch64"

    def test_macos_arm64(self) -> None:
        dl, whl = vendor._detect_tags("Darwin", "arm64")
        assert dl == "macosx_11_0_arm64"
        assert whl == "macosx_11_0_arm64"

    def test_macos_x86(self) -> None:
        dl, whl = vendor._detect_tags("Darwin", "x86_64")
        assert dl == "macosx_10_15_x86_64"
        assert whl == "macosx_10_15_x86_64"

    def test_windows_amd64(self) -> None:
        dl, whl = vendor._detect_tags("Windows", "amd64")
        assert dl == "win_amd64"
        assert whl == "win_amd64"

    def test_windows_x86_64(self) -> None:
        dl, whl = vendor._detect_tags("Windows", "x86_64")
        assert dl == "win_amd64"
        assert whl == "win_amd64"

    def test_windows_x86(self) -> None:
        dl, whl = vendor._detect_tags("Windows", "i386")
        assert dl == "win32"
        assert whl == "win32"

    def test_unsupported_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Unsupported platform"):
            vendor._detect_tags("FreeBSD", "x86_64")

    def test_detect_platform_tag_delegates(self) -> None:
        with (
            patch.object(vendor.platform, "system", return_value="Darwin"),
            patch.object(vendor.platform, "machine", return_value="arm64"),
        ):
            assert vendor.detect_platform_tag() == "macosx_11_0_arm64"

    def test_detect_wheel_platform_delegates(self) -> None:
        with (
            patch.object(vendor.platform, "system", return_value="Linux"),
            patch.object(vendor.platform, "machine", return_value="x86_64"),
        ):
            assert vendor.detect_wheel_platform() == ("manylinux_2_17_x86_64.manylinux2014_x86_64")


# ---------------------------------------------------------------------------
# 2. test_download_wheel
# ---------------------------------------------------------------------------


class TestDownloadWheel:
    """Verify download_wheel calls pip with --index-url (not --extra-index-url)."""

    def test_uses_index_url(self, tmp_path: Path) -> None:
        dest = tmp_path / "dest"
        dest.mkdir()

        fake_whl = dest / "llama_cpp_python-0.3.18-cp312-cp312-macosx_11_0_arm64.whl"
        fake_whl.write_bytes(b"fake")

        with patch.object(vendor.subprocess, "check_call") as mock_call:
            result = vendor.download_wheel("0.3.18", "macosx_11_0_arm64", dest)

        cmd = mock_call.call_args[0][0]
        index_args = [a for a in cmd if "index-url" in a]
        assert len(index_args) == 1
        assert index_args[0].startswith("--index-url=")
        assert "--extra-index-url" not in " ".join(cmd)
        assert result == fake_whl

    def test_raises_when_no_wheel_found(self, tmp_path: Path) -> None:
        dest = tmp_path / "empty"
        dest.mkdir()

        with (
            patch.object(vendor.subprocess, "check_call"),
            pytest.raises(RuntimeError, match="No wheel downloaded"),
        ):
            vendor.download_wheel("0.3.18", "macosx_11_0_arm64", dest)


# ---------------------------------------------------------------------------
# 3. test_repack_wheel_injects_vendor
# ---------------------------------------------------------------------------


def test_repack_wheel_injects_vendor(tmp_path: Path) -> None:
    """After repack, the output wheel contains lilbee/_vendor/llama_cpp/ files."""
    lilbee_whl = _make_lilbee_wheel(tmp_path)
    llama_whl = _make_llama_wheel(tmp_path / "llama_dl")

    out = vendor.repack_wheel(lilbee_whl, llama_whl, "macosx_11_0_arm64")

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert "lilbee/_vendor/llama_cpp/__init__.py" in names
    assert "lilbee/_vendor/llama_cpp/llama.py" in names
    assert "lilbee/_vendor/llama_cpp/libllama.dylib" in names


# ---------------------------------------------------------------------------
# 4. test_repack_strips_dependency
# ---------------------------------------------------------------------------


def test_repack_strips_dependency(tmp_path: Path) -> None:
    """METADATA must not contain llama-cpp-python Requires-Dist."""
    lilbee_whl = _make_lilbee_wheel(tmp_path)
    llama_whl = _make_llama_wheel(tmp_path / "llama_dl")

    out = vendor.repack_wheel(lilbee_whl, llama_whl, "macosx_11_0_arm64")

    with zipfile.ZipFile(out) as zf:
        metadata = zf.read("lilbee-0.6.0.dist-info/METADATA").decode()

    assert "llama-cpp-python" not in metadata.lower()
    assert "Requires-Dist: lancedb" in metadata
    assert "Requires-Dist: tiktoken" in metadata


# ---------------------------------------------------------------------------
# 5. test_repack_updates_record
# ---------------------------------------------------------------------------


def test_repack_updates_record(tmp_path: Path) -> None:
    """RECORD must have sha256 hashes for all vendored files."""
    lilbee_whl = _make_lilbee_wheel(tmp_path)
    llama_whl = _make_llama_wheel(tmp_path / "llama_dl")

    out = vendor.repack_wheel(lilbee_whl, llama_whl, "macosx_11_0_arm64")

    with zipfile.ZipFile(out) as zf:
        record_text = zf.read("lilbee-0.6.0.dist-info/RECORD").decode()

    for vendored in (
        "lilbee/_vendor/llama_cpp/__init__.py",
        "lilbee/_vendor/llama_cpp/llama.py",
        "lilbee/_vendor/llama_cpp/libllama.dylib",
    ):
        matching = [line for line in record_text.splitlines() if line.startswith(vendored)]
        assert len(matching) == 1, f"Missing RECORD entry for {vendored}"
        assert "sha256=" in matching[0]

    record_self = [
        line for line in record_text.splitlines() if "RECORD" in line and line.endswith(",,")
    ]
    assert len(record_self) == 1


# ---------------------------------------------------------------------------
# 6. test_repack_correct_filename
# ---------------------------------------------------------------------------


def test_repack_correct_filename(tmp_path: Path) -> None:
    """Output wheel has the correct platform-specific filename."""
    lilbee_whl = _make_lilbee_wheel(tmp_path)
    llama_whl = _make_llama_wheel(tmp_path / "llama_dl")

    py = vendor.python_tag()
    out = vendor.repack_wheel(lilbee_whl, llama_whl, "macosx_11_0_arm64")

    expected = f"lilbee-0.6.0-{py}-{py}-macosx_11_0_arm64.whl"
    assert out.name == expected
    assert out.exists()


# ---------------------------------------------------------------------------
# 7. test_repack_preserves_existing_files
# ---------------------------------------------------------------------------


def test_repack_preserves_existing_files(tmp_path: Path) -> None:
    """Original lilbee package files are still present after repack."""
    lilbee_whl = _make_lilbee_wheel(tmp_path)
    llama_whl = _make_llama_wheel(tmp_path / "llama_dl")

    out = vendor.repack_wheel(lilbee_whl, llama_whl, "macosx_11_0_arm64")

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert "lilbee/__init__.py" in names
    assert "lilbee/config.py" in names


# ---------------------------------------------------------------------------
# 8. test_hash_computation
# ---------------------------------------------------------------------------


def test_hash_computation() -> None:
    """_sha256_urlsafe_b64 returns the correct PEP 376 RECORD hash format."""
    data = b"hello world"
    digest = hashlib.sha256(data).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    result = vendor._sha256_urlsafe_b64(data)
    assert result == expected
    assert "=" not in result
    assert "+" not in result


# ---------------------------------------------------------------------------
# 9. test_main_integration
# ---------------------------------------------------------------------------


def test_main_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: main() with mocked download produces a valid vendored wheel."""
    lilbee_whl = _make_lilbee_wheel(tmp_path)
    llama_whl = _make_llama_wheel(tmp_path / "llama_dl")

    def fake_download(version: str, plat: str, dest: Path) -> Path:
        return llama_whl

    monkeypatch.setattr(vendor, "download_wheel", fake_download)
    monkeypatch.setattr(
        "sys.argv",
        [
            "vendor_llama_cpp.py",
            str(lilbee_whl),
            "--platform",
            "macosx_11_0_arm64",
            "--wheel-platform",
            "macosx_11_0_arm64",
        ],
    )

    vendor.main()

    assert not lilbee_whl.exists()
    py = vendor.python_tag()
    out = tmp_path / f"lilbee-0.6.0-{py}-{py}-macosx_11_0_arm64.whl"
    assert out.exists()

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert "lilbee/_vendor/llama_cpp/__init__.py" in names


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_python_tag() -> None:
    """python_tag returns the expected cpXYZ format."""
    tag = vendor.python_tag()
    assert tag.startswith("cp3")
    assert tag == f"cp{sys.version_info.major}{sys.version_info.minor}"


def test_find_dist_info_raises(tmp_path: Path) -> None:
    """_find_dist_info raises when no .dist-info directory exists."""
    with pytest.raises(RuntimeError, match=r"No \.dist-info found"):
        vendor._find_dist_info(tmp_path)


def test_main_missing_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() exits with error when wheel path doesn't exist."""
    monkeypatch.setattr(
        "sys.argv",
        ["vendor_llama_cpp.py", "/nonexistent/lilbee-0.6.0-py3-none-any.whl"],
    )
    with pytest.raises(SystemExit):
        vendor.main()


def test_dir_size_mb(tmp_path: Path) -> None:
    """_dir_size_mb returns correct size."""
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00" * 1024 * 1024)
    size = vendor._dir_size_mb(tmp_path)
    assert abs(size - 1.0) < 0.01


def test_repack_cleans_pycache(tmp_path: Path) -> None:
    """__pycache__ directories from llama_cpp are not included in the output."""
    llama_dir = tmp_path / "llama_dl"
    llama_whl = _make_wheel(
        llama_dir,
        name="llama_cpp_python",
        version="0.3.18",
        python="cp312",
        abi="cp312",
        plat="macosx_11_0_arm64",
        pkg_dir="llama_cpp",
        pkg_files={
            "__init__.py": b"# llama_cpp\n",
            "__pycache__/foo.cpython-312.pyc": b"\x00pyc",
        },
    )
    lilbee_whl = _make_lilbee_wheel(tmp_path)

    out = vendor.repack_wheel(lilbee_whl, llama_whl, "macosx_11_0_arm64")

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert not any("__pycache__" in n for n in names)
