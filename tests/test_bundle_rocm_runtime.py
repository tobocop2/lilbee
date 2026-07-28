"""The ROCm bundler packs the runtime the engine links, discovered rather than listed.

The wheel this guards exists so an AMD user needs a kernel driver and nothing else.
These tests cover the walk that packs it: every dependency edge is declared by the
fixture, so a library the bundler fails to follow is a test failure rather than a
user's crash, and none of it needs a GPU or a six-hour build.

Whether the result actually loads on a machine without ROCm is settled elsewhere, by
``tools/qa/assert_rocm_bundle_loads.sh`` running it through a real ld.so in a container.
"""

from __future__ import annotations

import pathlib
import shutil
import stat
import subprocess
import sys

import pytest

# These drive POSIX shell through bash and build a colon-separated PATH. The scripts
# they cover only ever run on the Linux wheel and executable builders.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the scripts under test are POSIX shell, Linux-only in CI"
)

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools/wheel-build/bundle_rocm_runtime.sh"

# What the engine build itself drops in the wheel's bin/, before any ROCm is packed.
_BUILT = ("llama-server", "libggml-hip.so", "libggml-base.so")

# A plausible ROCm 7.2 dependency graph. libomp sits in lib/llvm/lib rather than lib,
# which is the split that made the first version of this bundler ship an unloadable
# wheel, and rocblas <-> amdhip is a genuine cycle the walk has to terminate through.
_ROCM_GRAPH = {
    "libamdhip64.so.7": ["librocblas.so.5", "libnuma.so.1"],
    "librocblas.so.5": ["libamdhip64.so.7", "libstdc++.so.6"],
    # hipBLAS drags in rocSOLVER for LAPACK entry points ggml never calls, and
    # rocSOLVER alone keeps rocroller alive. 888 MB and 80 MB in the real bundle.
    "libhipblas.so.3": ["librocblas.so.5", "librocsolver.so.0"],
    "librocsolver.so.0": ["librocroller.so.1"],
    "librocroller.so.1": [],
    "libomp.so": [],
    # Nothing declares comgr: libamdhip64 dlopens it by name.
    "libamd_comgr.so.3": [],
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
    removed = bin_dir / "removed"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'target="${!#}"\n'
        'base="$(basename "${target}")"\n'
        '{ case "${base}" in\n'
        f"{cases}\n"
        "    *) ;;\n"
        "esac\n"
        "} | while IFS= read -r line; do\n"
        '  dep="${line##*[}"; dep="${dep%]}"\n'
        f'  if [ -f "{removed}" ] && grep -qxF "${{base}} ${{dep}}" "{removed}"\n'
        "  then continue; fi\n"
        '  printf "%s\\n" "${line}"\n'
        "done\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def rocm_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A ROCm install holding the graph above, split across lib and lib/llvm/lib."""
    lib = tmp_path / "rocm/lib"
    llvm = lib / "llvm/lib"
    (lib / "rocblas/library").mkdir(parents=True)
    llvm.mkdir(parents=True)
    (lib / "rocblas/library/TensileManifest.txt").write_text("shared metadata")
    for arch in ("gfx906", "gfx90a", "gfx942", "gfx1100"):
        (lib / f"rocblas/library/TensileLibrary_{arch}.dat").write_text(f"kernels for {arch}")
    for name in _ROCM_GRAPH:
        target = llvm if name in _LLVM_LIBS else lib
        (target / name).write_text(f"elf: {name}")
    # ROCm ships three names for comgr; only the SONAME form may be bundled.
    (lib / "libamd_comgr.so").write_text("elf: libamd_comgr.so.3")
    (lib / "libamd_comgr.so.3.0.0").write_text("elf: libamd_comgr.so.3")
    return tmp_path / "rocm"


@pytest.fixture
def bin_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """The wheel's bin/ as the engine build leaves it."""
    path = tmp_path / "bin"
    path.mkdir()
    for name in _BUILT:
        (path / name).write_text(f"elf: {name}")
    return path


def _write_stub_patchelf(bin_dir: pathlib.Path, log: pathlib.Path) -> None:
    """A patchelf that records its arguments instead of rewriting an ELF."""
    stub = bin_dir / "patchelf"
    # --remove-needed must actually take effect, or the orphan collection below is
    # tested against a graph that never changed.
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{log}"\n'
        'if [ "$1" = "--remove-needed" ]; then\n'
        f'  echo "$(basename "$3") $2" >> "{bin_dir / "removed"}"\n'
        "fi\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


# Everything bundle_rocm_runtime.sh shells out to, minus readelf and patchelf which
# are stubbed. PATH is built from this alone so "patchelf is absent" is a property of
# the test rather than of the host: CI runners have a real one, developer laptops
# usually do not, and the test that asserts absence flipped between them.
_REAL_TOOLS = (
    # bash and env first: the stubs below carry a #!/usr/bin/env bash shebang, which
    # resolves through this PATH like everything else.
    "bash",
    "env",
    "basename",
    "cmp",
    "cp",
    "cut",
    "dirname",
    "du",
    "find",
    "grep",
    "head",
    "mkdir",
    "rm",
    "sed",
    "tr",
)


def _link_real_tools(stub_dir: pathlib.Path) -> None:
    stub_dir.mkdir(parents=True, exist_ok=True)
    for tool in _REAL_TOOLS:
        found = shutil.which(tool)
        assert found, f"{tool} not on PATH; the bundler needs it"
        target = stub_dir / tool
        if not target.exists():
            target.symlink_to(found)


def _run(
    bin_dir: pathlib.Path,
    rocm_tree: pathlib.Path,
    tmp_path: pathlib.Path,
    needed: dict[str, list[str]] | None = None,
    targets: str | None = None,
    with_patchelf: bool = True,
) -> subprocess.CompletedProcess[str]:
    stub_dir = tmp_path / "stub"
    _write_stub_readelf(stub_dir, _NEEDED if needed is None else needed)
    if with_patchelf:
        _write_stub_patchelf(stub_dir, tmp_path / "patchelf.log")
    _link_real_tools(stub_dir)
    env = {
        "ROCM_PATH": str(rocm_tree),
        "PATH": str(stub_dir),
    }
    if targets is not None:
        env["ROCM_TARGETS"] = targets
    # bash by absolute path: PATH above is the script's lookup table, not ours.
    bash = shutil.which("bash")
    assert bash, "bash not on PATH"
    return subprocess.run(
        [bash, str(_SCRIPT), str(bin_dir)],
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
    assert (bundled / "rocblas/library/TensileLibrary_gfx942.dat").is_file()


def test_drops_a_soname_alias_nothing_loads_by(bin_dir, rocm_tree, tmp_path):
    """Three real copies of a 437MB library is most of the wheel, for nothing."""
    (bin_dir / "libggml-hip.so.0.15.1").write_text("elf: libggml-hip.so")
    result = _run(bin_dir, rocm_tree, tmp_path)
    assert result.returncode == 0, result.stderr
    assert not (bin_dir / "libggml-hip.so.0.15.1").exists()
    assert (bin_dir / "libggml-hip.so").is_file()


def test_keeps_an_alias_something_links_against(bin_dir, rocm_tree, tmp_path):
    """A name in someone's DT_NEEDED is load-bearing however identical the bytes."""
    (bin_dir / "libggml-base.so.0").write_text("elf: libggml-base.so")
    needed = {**_NEEDED, "llama-server": ["libggml-base.so.0", "libomp.so"]}
    result = _run(bin_dir, rocm_tree, tmp_path, needed=needed)
    assert result.returncode == 0, result.stderr
    assert (bin_dir / "libggml-base.so.0").is_file()


def _alias_group(bin_dir: pathlib.Path, stem: str, body: str) -> None:
    """The three real files the build leaves where a SONAME symlink chain would be."""
    for name in (f"{stem}.so", f"{stem}.so.0", f"{stem}.so.0.15.1"):
        (bin_dir / name).write_text(body)


def test_keeps_only_the_soname_when_the_backend_is_linked(bin_dir, rocm_tree, tmp_path):
    """GGML_BACKEND_DL is off here, so libggml.so.0 links the backend by SONAME.

    The plain name and the fully-versioned file are then dead weight, and at 437MB each
    they are most of the wheel.
    """
    _alias_group(bin_dir, "libggml-hip", "elf: libggml-hip.so")
    needed = {**_NEEDED, "libggml-base.so": ["libggml-hip.so.0", "libomp.so"]}
    result = _run(bin_dir, rocm_tree, tmp_path, needed=needed)
    assert result.returncode == 0, result.stderr
    assert (bin_dir / "libggml-hip.so.0").is_file()
    assert not (bin_dir / "libggml-hip.so.0.15.1").exists()
    assert not (bin_dir / "libggml-hip.so").exists()


def test_keeps_the_plain_name_when_nothing_links_the_group(bin_dir, rocm_tree, tmp_path):
    """The other build shape: with backends dlopened, the plain name is the only one asked for.

    Deleting it would leave a bundle that loads and then finds no GPU backend, which no
    linker-level check could catch.
    """
    _alias_group(bin_dir, "libggml-extra", "elf: libggml-extra.so")
    result = _run(bin_dir, rocm_tree, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (bin_dir / "libggml-extra.so").is_file()
    assert not (bin_dir / "libggml-extra.so.0").exists()
    assert not (bin_dir / "libggml-extra.so.0.15.1").exists()


def test_never_drops_the_only_copy_of_a_library(bin_dir, rocm_tree, tmp_path):
    """A lone versioned file has no surviving twin, so it must be left alone."""
    (bin_dir / "libsolo.so.4").write_text("elf: the only copy")
    result = _run(bin_dir, rocm_tree, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (bin_dir / "libsolo.so.4").is_file()


def test_keeps_an_alias_whose_bytes_differ(bin_dir, rocm_tree, tmp_path):
    """Only an exact duplicate is redundant; a different build is a different library."""
    (bin_dir / "libggml-hip.so.0.15.1").write_text("elf: a genuinely different build")
    result = _run(bin_dir, rocm_tree, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (bin_dir / "libggml-hip.so.0.15.1").is_file()


def test_repoints_copied_libraries_at_the_bundle(bundled, tmp_path):
    """A copied library keeps a runpath into the ROCm install, which a user has not.

    Without the rewrite the bundle only resolves on a machine that already has ROCm,
    which is every machine that can build it and no machine that needs it.
    """
    calls = (tmp_path / "patchelf.log").read_text().splitlines()
    rewrites = [line for line in calls if "--set-rpath" in line]
    rewritten = {line.split()[-1].rsplit("/", 1)[-1] for line in rewrites}
    # Not rocsolver or rocroller: those are pruned before this runs.
    assert rewritten == {
        "libamdhip64.so.7",
        "librocblas.so.5",
        "libhipblas.so.3",
        "libomp.so",
        "libamd_comgr.so.3",
    }
    assert all("--set-rpath $ORIGIN" in line for line in rewrites)


def test_leaves_the_engines_own_libraries_alone(bundled, tmp_path):
    """The build already baked $ORIGIN into those; rewriting them is not this job."""
    calls = (tmp_path / "patchelf.log").read_text()
    assert "libggml-hip.so" not in calls
    assert "llama-server" not in calls


def test_fails_when_patchelf_is_absent(bin_dir, rocm_tree, tmp_path):
    """Silently skipping the rewrite ships a bundle that resolves only where built."""
    result = _run(bin_dir, rocm_tree, tmp_path, with_patchelf=False)
    assert result.returncode == 1
    assert "patchelf is required" in result.stderr


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


def test_drops_kernels_for_architectures_the_engine_cannot_run(bin_dir, rocm_tree, tmp_path):
    """rocBLAS ships kernels for every arch its own build covered, not the ones we compiled.

    They are the single largest thing in the bundle after the backend itself.
    """
    result = _run(bin_dir, rocm_tree, tmp_path, targets="gfx942;gfx1100")
    assert result.returncode == 0, result.stderr
    kernels = bin_dir / "rocblas/library"
    assert (kernels / "TensileLibrary_gfx942.dat").is_file()
    assert (kernels / "TensileLibrary_gfx1100.dat").is_file()
    assert not (kernels / "TensileLibrary_gfx906.dat").exists()
    assert not (kernels / "TensileLibrary_gfx90a.dat").exists()


def test_keeps_kernel_files_that_name_no_architecture(bin_dir, rocm_tree, tmp_path):
    """Shared metadata has no gfx token and dropping it would break every target."""
    result = _run(bin_dir, rocm_tree, tmp_path, targets="gfx942")
    assert result.returncode == 0, result.stderr
    assert (bin_dir / "rocblas/library/TensileManifest.txt").is_file()


def test_keeps_every_kernel_when_the_targets_are_unknown(bin_dir, rocm_tree, tmp_path):
    """Guessing wrong costs a card that silently falls back to CPU, so unset means keep all."""
    result = _run(bin_dir, rocm_tree, tmp_path)
    assert result.returncode == 0, result.stderr
    kernels = bin_dir / "rocblas/library"
    assert (kernels / "TensileLibrary_gfx906.dat").is_file()
    assert (kernels / "TensileLibrary_gfx942.dat").is_file()


def test_drops_the_lapack_dependency_ggml_never_calls(bundled, tmp_path):
    """rocSOLVER is 888MB of the bundle and ggml calls only the hipBLAS GEMM family."""
    assert not (bundled / "librocsolver.so.0").exists()
    calls = (tmp_path / "patchelf.log").read_text()
    assert "--remove-needed librocsolver.so.0" in calls


def test_collects_what_the_dropped_dependency_alone_kept(bundled):
    """rocroller is reachable only through rocSOLVER, so it goes with it."""
    assert not (bundled / "librocroller.so.1").exists()


def test_keeps_libraries_still_reachable_by_another_path(bundled):
    """rocBLAS hangs off hip as well as rocSOLVER; pruning one edge must not take it."""
    assert (bundled / "librocblas.so.5").is_file()
    assert (bundled / "libhipblas.so.3").is_file()
    assert (bundled / "libamdhip64.so.7").is_file()


def test_bundles_the_library_that_is_only_ever_dlopened(bundled):
    """libamdhip64 loads comgr by SONAME, so no DT_NEEDED walk can discover it."""
    assert (bundled / "libamd_comgr.so.3").is_file()


def test_never_collects_a_dlopened_library_as_an_orphan(bundled):
    """It is unreachable by construction; collecting it breaks every AMD GPU."""
    assert (bundled / "libamd_comgr.so.3").is_file()


def test_fails_when_the_dlopened_library_is_missing(bin_dir, rocm_tree, tmp_path):
    """Nothing links it, so only an explicit check can notice it never got copied."""
    (rocm_tree / "lib/libamd_comgr.so.3").unlink()
    result = _run(bin_dir, rocm_tree, tmp_path)
    assert result.returncode == 1
    assert "libamd_comgr" in result.stderr


def test_takes_only_the_soname_of_a_dlopened_library(bundled):
    """cp -L on each of libfoo.so, .so.3 and .so.3.0.0 is three full copies of one library."""
    assert (bundled / "libamd_comgr.so.3").is_file()
    assert not (bundled / "libamd_comgr.so").exists()
    assert not (bundled / "libamd_comgr.so.3.0.0").exists()


def test_keeps_the_dlopened_soname_when_nothing_links_it(bin_dir, rocm_tree, tmp_path):
    """With rocroller gone nothing DT_NEEDEDs comgr, and dedup must still not take it."""
    needed = {**_NEEDED, "librocroller.so.1": []}
    result = _run(bin_dir, rocm_tree, tmp_path, needed=needed)
    assert result.returncode == 0, result.stderr
    assert (bin_dir / "libamd_comgr.so.3").is_file()


def test_never_repoints_a_library_it_deleted(bundled, tmp_path):
    """patchelf fails hard on a missing file, so the bundled list must track deletions.

    A stale entry took the whole rocm cell down after a dedup pass removed the file
    the repoint step then tried to rewrite.
    """
    rewritten = [
        line.split()[-1]
        for line in (tmp_path / "patchelf.log").read_text().splitlines()
        if "--set-rpath" in line
    ]
    assert rewritten, "no runpaths rewritten at all"
    missing = [path for path in rewritten if not pathlib.Path(path).exists()]
    assert missing == [], f"repointed files that no longer exist: {missing}"
