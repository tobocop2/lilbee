"""The ROCm bundler packs the runtime the engine links, discovered rather than listed.

The wheel this guards exists so an AMD user needs a kernel driver and nothing else.
These tests cover the walk that packs it: every dependency edge is declared by the
fixture, so a library the bundler fails to follow is a test failure rather than a
user's crash, and none of it needs a GPU or a six-hour build.

Whether the result actually loads on a machine without ROCm is settled elsewhere,
by build-multigpu.yml running the bundle through a real ld.so in a container.
"""

from __future__ import annotations

import pathlib
import shutil
import stat
import subprocess

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/bundle_rocm_runtime.sh"

# What the engine build itself drops in the wheel's bin/, before any ROCm is packed.
_BUILT = ("llama-server", "libggml-hip.so", "libggml-base.so")

# A plausible ROCm 7.2 dependency graph. libomp sits in lib/llvm/lib rather than lib,
# which is the split that made the first version of this bundler ship an unloadable
# wheel, and rocblas <-> amdhip is a genuine cycle the walk has to terminate through.
_ROCM_GRAPH = {
    "libamdhip64.so.7": ["librocblas.so.5", "libnuma.so.1"],
    "librocblas.so.5": ["libamdhip64.so.7", "libstdc++.so.6"],
    "libhipblas.so.3": ["librocblas.so.5"],
    "libomp.so": [],
}
_LLVM_LIBS = ("libomp.so",)

# libomp hangs off llama-server and the base library rather than the HIP backend,
# which is how the rocm cell really links: everything is compiled with AMD's clang so
# ggml's OpenMP is libomp, but the HIP backend itself does not use OpenMP. Walking
# only the backend therefore never reaches it.
_NEEDED = {
    "llama-server": ["libggml-base.so", "libomp.so", "libc.so.6"],
    "libggml-hip.so": ["libamdhip64.so.7", "libhipblas.so.3"],
    "libggml-base.so": ["libstdc++.so.6", "libomp.so"],
    **_ROCM_GRAPH,
}


def _write_stub_readelf(bin_dir: pathlib.Path, needed: dict[str, list[str]]) -> None:
    """A readelf that reports DT_NEEDED from *needed*, keyed by basename."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    cases = "\n".join(
        "    {name}) {body} ;;".format(
            name=name,
            body=(
                "printf '%s\\n' "
                + " ".join(f"' 0x0001 (NEEDED) Shared library: [{dep}]'" for dep in deps)
                if deps
                else ":"
            ),
        )
        for name, deps in needed.items()
    )
    stub = bin_dir / "readelf"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'target="${!#}"\n'
        'case "$(basename "${target}")" in\n'
        f"{cases}\n"
        "    *) ;;\n"
        "esac\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def rocm_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A ROCm install holding the graph above, split across lib and lib/llvm/lib."""
    lib = tmp_path / "rocm/lib"
    llvm = lib / "llvm/lib"
    (lib / "rocblas/library").mkdir(parents=True)
    llvm.mkdir(parents=True)
    (lib / "rocblas/library/TensileLibrary.dat").write_text("kernels")
    for name in _ROCM_GRAPH:
        target = llvm if name in _LLVM_LIBS else lib
        (target / name).write_text(f"elf: {name}")
    return tmp_path / "rocm"


@pytest.fixture
def bin_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """The wheel's bin/ as the engine build leaves it."""
    path = tmp_path / "bin"
    path.mkdir()
    for name in _BUILT:
        (path / name).write_text(f"elf: {name}")
    return path


def _run(
    bin_dir: pathlib.Path,
    rocm_tree: pathlib.Path,
    tmp_path: pathlib.Path,
    needed: dict[str, list[str]] | None = None,
) -> subprocess.CompletedProcess[str]:
    stub_dir = tmp_path / "stub"
    _write_stub_readelf(stub_dir, _NEEDED if needed is None else needed)
    env = {
        "ROCM_PATH": str(rocm_tree),
        "PATH": f"{stub_dir}:/usr/bin:/bin",
    }
    return subprocess.run(
        ["bash", str(_SCRIPT), str(bin_dir)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


@pytest.fixture
def bundled(bin_dir, rocm_tree, tmp_path):
    """A successful run, for the several assertions that share one."""
    result = _run(bin_dir, rocm_tree, tmp_path)
    assert result.returncode == 0, result.stderr
    return bin_dir


def test_script_is_executable():
    """build_llama_server.sh invokes it by path, so a lost bit fails the rocm cell."""
    assert _SCRIPT.stat().st_mode & stat.S_IEXEC


def test_bundles_the_direct_dependencies(bundled):
    """hip and hipblas are what libggml-hip.so links; without them nothing loads."""
    assert (bundled / "libamdhip64.so.7").is_file()
    assert (bundled / "libhipblas.so.3").is_file()


def test_follows_the_closure_transitively(bundled):
    """rocblas is reached only through hip; a one-level copy would miss it."""
    assert (bundled / "librocblas.so.5").is_file()


def test_bundles_the_openmp_runtime_from_the_llvm_directory(bundled):
    """libomp lives in lib/llvm/lib, not lib, and is reached from llama-server alone.

    Two ways to miss it, both of which shipped: search only ROCm's lib directory, or
    seed the walk with the HIP backend rather than everything the build produced.
    """
    assert (bundled / "libomp.so").is_file()


def test_leaves_host_libraries_alone(bundled):
    """libc and libstdc++ come from the machine; bundling them breaks the ABI."""
    assert not (bundled / "libstdc++.so.6").exists()
    assert not (bundled / "libc.so.6").exists()
    assert not (bundled / "libnuma.so.1").exists()


def test_bundles_the_rocblas_kernels(bundled):
    """rocBLAS loads these as data at runtime; without them the first matmul fails."""
    assert (bundled / "rocblas/library/TensileLibrary.dat").is_file()


def test_reports_what_it_packed_and_how_big(bin_dir, rocm_tree, tmp_path):
    """The build log is where the wheel's size is read from after a CI run."""
    result = _run(bin_dir, rocm_tree, tmp_path)
    assert "libamdhip64.so.7" in result.stdout
    assert "rocm bundle size:" in result.stdout


def test_fails_when_the_backend_was_never_built(bin_dir, rocm_tree, tmp_path):
    """A CPU-only build is the exact bug this whole branch exists to stop shipping."""
    (bin_dir / "libggml-hip.so").unlink()
    result = _run(bin_dir, rocm_tree, tmp_path)
    assert result.returncode == 1
    assert "no libggml-hip.so was built" in result.stderr


def test_fails_when_the_rocm_tree_is_absent(bin_dir, tmp_path):
    """Silently packing nothing would ship the CPU wheel again under a rocm tag."""
    result = _run(bin_dir, tmp_path / "nonexistent", tmp_path)
    assert result.returncode == 1
    assert "does not exist" in result.stderr


def test_fails_when_a_required_library_is_missing(bin_dir, rocm_tree, tmp_path):
    """hipblas unreachable through DT_NEEDED still means an engine that cannot load."""
    needed = {**_NEEDED, "libggml-hip.so": ["libamdhip64.so.7"]}
    result = _run(bin_dir, rocm_tree, tmp_path, needed=needed)
    assert result.returncode == 1
    assert "libhipblas" in result.stderr


def test_fails_when_the_kernels_are_missing(bin_dir, rocm_tree, tmp_path):
    """A library that loads and dies on first use is worse than one that never loads."""
    shutil.rmtree(rocm_tree / "lib/rocblas")
    result = _run(bin_dir, rocm_tree, tmp_path)
    assert result.returncode == 1
    assert "rocBLAS kernels missing" in result.stderr
