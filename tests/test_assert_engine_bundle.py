"""Tests for the engine-bundle gate that keeps a backend-less wheel off a release."""

from __future__ import annotations

import importlib.util
import shlex
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "qa" / "assert_engine_bundle.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("assert_engine_bundle", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acb = _load()

_LINUX_RUNTIME = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12")
_WINDOWS_RUNTIME = ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")
_WINDOWS_CRT = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll")
# What a rocm wheel must carry beside libggml-hip.so.
_ROCM_RUNTIME = (
    "libamdhip64.so.7",
    "librocblas.so.4",
    "libhipblas.so.3",
    "rocblas/library/TensileLibrary.dat",
)
# What a runtime-dispatch cell ships instead of a single libggml-cpu.
_LINUX_VARIANTS = ("libggml-cpu-x64.so", "libggml-cpu-haswell.so", "libggml-cpu-sapphirerapids.so")
_WINDOWS_VARIANTS = ("ggml-cpu-x64.dll", "ggml-cpu-haswell.dll")


def _wheel(tmp_path: Path, name: str, bin_files: tuple[str, ...]) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for entry in bin_files:
            zf.writestr(f"lilbee_engine_llama_cpp/bin/{entry}", b"x")
    return path


def test_linux_cuda_wheel_without_runtime_is_rejected(tmp_path: Path) -> None:
    # The shipped-broken shape: ggml-cuda present, the runtime it links absent.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so"),
    )
    problems = acb.check(wheel)
    assert len(problems) == 3
    assert any("libcudart.so" in p for p in problems)
    assert any("libcublasLt.so" in p for p in problems)


def test_linux_cuda_wheel_with_runtime_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so", *_LINUX_RUNTIME),
    )
    assert acb.check(wheel) == []


def test_windows_cuda_wheel_without_runtime_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cu125-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cuda.dll", *_WINDOWS_CRT),
    )
    assert len(acb.check(wheel)) == 3


def test_windows_cuda_wheel_with_runtime_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cu125-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cuda.dll", *_WINDOWS_RUNTIME, *_WINDOWS_CRT),
    )
    assert acb.check(wheel) == []


def test_cuda_detected_from_payload_not_filename(tmp_path: Path) -> None:
    # A freshly built wheel has no -1.cu125- build tag; that is added at attach time.
    # Detection must still fire, or the gate silently passes every build artifact.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cuda.dll", *_WINDOWS_CRT),
    )
    assert len(acb.check(wheel)) == 3


def test_non_cuda_wheel_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cpu-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", *_LINUX_VARIANTS),
    )
    assert acb.check(wheel) == []


def test_non_cuda_wheel_carrying_cuda_runtime_is_rejected(tmp_path: Path) -> None:
    # Guards the other direction: the copy step must stay scoped to CUDA backends.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.vulkan-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-vulkan.dll", "cudart64_12.dll", *_WINDOWS_CRT),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "non-CUDA wheel ships CUDA runtime" in problems[0]


def test_windows_wheel_without_the_msvc_runtime_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.vulkan-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-vulkan.dll", "msvcp140.dll"),
    )
    problems = acb.check(wheel)
    assert [p for p in problems if "vcruntime140.dll" in p]
    assert [p for p in problems if "vcruntime140_1.dll" in p]
    assert not [p for p in problems if "msvcp140.dll" in p]


def test_empty_bin_dir_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lilbee_engine_llama_cpp-1.0-py3-none-win_amd64.whl"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("lilbee_engine_llama_cpp/__init__.py", b"")
    assert acb.check(path) == [f"{path.name}: no lilbee_engine_llama_cpp/bin/ payload at all"]


def test_macos_cuda_payload_asserts_nothing(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-py3-none-macosx_11_0_arm64.whl",
        ("llama-server", "libggml-cuda.so"),
    )
    assert acb.check(wheel) == []


def test_rocm_wheel_without_the_hip_backend_is_rejected(tmp_path: Path) -> None:
    # The shape that shipped: BACKEND=rocm, and the bundle holds only the CPU
    # backend because the cmake flag naming it was ignored.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-base.so", "libggml-cpu.so"),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "libggml-hip.so" in problems[0]


def test_rocm_wheel_with_the_hip_backend_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so", "libggml-hip.so", *_ROCM_RUNTIME),
    )
    assert acb.check(wheel) == []


def test_a_versioned_soname_alone_satisfies_the_backend_check(tmp_path: Path) -> None:
    # The engine links by SONAME, so the versioned file is the real library and
    # the bare name is a symlink beside it. A bundle that carries only the former
    # is complete, and failing it would be a false alarm.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so", "libggml-hip.so.0.15.1", *_ROCM_RUNTIME),
    )
    assert acb.check(wheel) == []


def test_backend_argument_covers_a_wheel_built_before_its_tag_is_added(tmp_path: Path) -> None:
    # A freshly built artifact has no -1.rocm- tag, and that is exactly where the
    # gate has to fire: at build time, on the runner that produced it.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    assert acb.check(wheel) == []
    problems = acb.check(wheel, backend="rocm")
    assert len(problems) == 1
    assert "libggml-hip.so" in problems[0]


def test_sycl_wheel_without_its_backend_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.sycl-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    assert len(acb.check(wheel)) == 1


def test_windows_vulkan_wheel_without_its_backend_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.vulkan-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cpu.dll", *_WINDOWS_CRT),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "ggml-vulkan.dll" in problems[0]


def test_macos_metal_wheel_checks_the_dylib_name(tmp_path: Path) -> None:
    missing = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.metal-py3-none-macosx_11_0_arm64.whl",
        ("llama-server", "libggml-cpu.dylib"),
    )
    problems = acb.check(missing)
    assert len(problems) == 1
    assert "libggml-metal.dylib" in problems[0]

    present = _wheel(
        tmp_path,
        "ok-1.0-1.metal-py3-none-macosx_11_0_arm64.whl",
        ("llama-server", "libggml-cpu.dylib", "libggml-metal.0.15.1.dylib"),
    )
    assert acb.check(present) == []


def test_every_cuda_flavor_maps_to_the_cuda_backend_library(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cu124-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so", *_LINUX_RUNTIME),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "libggml-cuda.so" in problems[0]


def test_an_unrecognised_backend_asserts_nothing(tmp_path: Path) -> None:
    # A new flavor should not fail the gate before anyone teaches it the name of
    # the library that flavor builds.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cann-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    assert acb.check(wheel) == []


def test_rocm_wheel_without_the_runtime_is_rejected(tmp_path: Path) -> None:
    # The backend is present but the libraries it links are not, so it loads on a
    # machine with ROCm installed and silently falls back to CPU everywhere else.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so", "libggml-hip.so"),
    )
    problems = acb.check(wheel)
    assert len(problems) == len(_ROCM_RUNTIME)
    assert any("libamdhip64" in p for p in problems)
    assert any("rocblas/library" in p for p in problems)


def test_rocm_wheel_with_the_runtime_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so", "libggml-hip.so", *_ROCM_RUNTIME),
    )
    assert acb.check(wheel) == []


def test_rocm_wheel_without_the_tensile_kernels_is_rejected(tmp_path: Path) -> None:
    # rocBLAS loads these as data at runtime rather than linking them, so the
    # library resolves and then fails on the first matrix multiply.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-hip.so", *_ROCM_RUNTIME[:3]),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "rocblas/library" in problems[0]


def test_a_rocm_soname_bump_does_not_fail_the_gate(tmp_path: Path) -> None:
    # The gate must key on the library, not the ROCm release that produced it, or
    # every upgrade is a red build for a wheel that is in fact correct.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        (
            "llama-server",
            "libggml-hip.so",
            "libamdhip64.so.9",
            "librocblas.so.10",
            "libhipblas.so.42",
            "rocblas/library/TensileLibrary.dat",
        ),
    )
    assert acb.check(wheel) == []


def test_a_non_rocm_wheel_is_not_asked_for_the_rocm_runtime(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.vulkan-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-vulkan.so", "libvulkan.so.1", *_LINUX_VARIANTS),
    )
    assert acb.check(wheel) == []


def test_cpu_linux_wheel_with_a_single_cpu_library_is_rejected(tmp_path: Path) -> None:
    # The pre-dispatch shape: one libggml-cpu.so carrying only the AVX baseline.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cpu-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "libggml-cpu-<variant>.so" in problems[0]


def test_cpu_windows_wheel_needs_variant_modules(tmp_path: Path) -> None:
    single = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cpu-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cpu.dll", *_WINDOWS_CRT),
    )
    problems = acb.check(single)
    assert len(problems) == 1
    assert "ggml-cpu-<variant>.dll" in problems[0]

    dispatched = _wheel(
        tmp_path,
        "ok-1.0-1.cpu-py3-none-win_amd64.whl",
        ("llama-server.exe", *_WINDOWS_VARIANTS, *_WINDOWS_CRT),
    )
    assert acb.check(dispatched) == []


def test_a_lone_variant_module_is_rejected(tmp_path: Path) -> None:
    # One variant means the build host's instruction set, not runtime dispatch.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cpu-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu-haswell.so"),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "libggml-cpu-<variant>.so" in problems[0]


def test_vulkan_linux_wheel_without_the_loader_is_rejected(tmp_path: Path) -> None:
    # Under GGML_BACKEND_DL a missing loader is a silent CPU fallback, not a
    # startup failure, so only the payload can prove the loader ships.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.vulkan-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-vulkan.so", *_LINUX_VARIANTS),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "libvulkan.so.1" in problems[0]


def test_vulkan_windows_wheel_is_not_asked_for_variants_or_loader(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.vulkan-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-vulkan.dll", "ggml-cpu.dll", *_WINDOWS_CRT),
    )
    assert acb.check(wheel) == []


def test_main_passes_the_backend_through(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    assert acb.main(["--backend", "rocm", str(wheel)]) == 1
    assert "libggml-hip.so" in capsys.readouterr().err


def test_main_returns_nonzero_and_names_the_wheel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so"),
    )
    assert acb.main([str(wheel)]) == 1
    assert "libcudart.so" in capsys.readouterr().err


def test_main_returns_zero_on_a_good_wheel(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine_llama_cpp-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so", *_LINUX_RUNTIME),
    )
    assert acb.main([str(wheel)]) == 0


def test_main_rejects_a_missing_file(tmp_path: Path) -> None:
    assert acb.main([str(tmp_path / "nope.whl")]) == 1


def test_main_with_no_arguments_is_a_usage_error() -> None:
    assert acb.main([]) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))


_ROCM_PAYLOAD = (
    "libggml-hip.so",
    "libamdhip64.so.7",
    "librocblas.so.5",
    "libhipblas.so.3",
    "rocblas/library/kernels.dat",
)
_ROCM_WHEEL = "lilbee_engine_llama_cpp-0.1-1.rocm-py3-none-manylinux_2_17_x86_64.whl"


def test_a_wheel_too_big_to_upload_is_a_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub caps one artifact at 2 GiB, and a bundled ROCm engine is the one near it."""
    wheel = _wheel(tmp_path, _ROCM_WHEEL, _ROCM_PAYLOAD)
    monkeypatch.setattr(acb, "WHEEL_SIZE_LIMIT", 1)
    assert any("exceeds the" in p for p in acb.check(wheel))


def test_a_wheel_within_the_limit_passes(tmp_path: Path) -> None:
    """The common case must not pay for the check with a false failure."""
    wheel = _wheel(tmp_path, _ROCM_WHEEL, _ROCM_PAYLOAD)
    assert acb.check(wheel) == []


def _server_wheel(
    tmp_path: Path, *, server: str | None = "llama-server", mode: int = 0o755, base: bool = True
) -> Path:
    path = tmp_path / "lilbee_engine_llama_cpp-1-1.cpu-py3-none-manylinux_2_34_x86_64.whl"
    with zipfile.ZipFile(path, "w") as zf:
        if server:
            info = zipfile.ZipInfo(f"lilbee_engine_llama_cpp/bin/{server}")
            info.external_attr = (0o100000 | mode) << 16
            zf.writestr(info, b"x")
        zf.writestr("lilbee_engine_llama_cpp/bin/libggml-cpu-x64.so", b"x")
        zf.writestr("lilbee_engine_llama_cpp/bin/libggml-cpu-haswell.so", b"x")
        if base:
            zf.writestr("lilbee_engine_llama_cpp/bin/libggml-base.so.0.9.5", b"x")
    return path


def test_a_wheel_with_the_pinned_server_passes_the_server_check(tmp_path: Path) -> None:
    assert acb.check(_server_wheel(tmp_path), "cpu", tool="llama-server") == []


@pytest.mark.parametrize(
    ("server", "mode", "message"),
    [
        (None, 0o755, "has no llama-server in its bin directory"),
        ("gguf-parser", 0o755, "has no llama-server in its bin directory"),
        ("llama-server", 0o644, "without the execute bit"),
    ],
)
def test_a_wheel_whose_server_cannot_run_fails_the_server_check(
    tmp_path: Path, server: str | None, mode: int, message: str
) -> None:
    wheel = _server_wheel(tmp_path, server=server, mode=mode)
    problems = acb.check(wheel, "cpu", tool="llama-server")
    assert [p for p in problems if message in p]


def test_the_server_check_asks_for_no_library_of_one_engine(tmp_path: Path) -> None:
    wheel = _server_wheel(tmp_path, base=False)
    assert acb.check(wheel, "cpu", tool="llama-server") == []


def test_a_wheel_with_no_server_passes_when_no_tool_is_named(tmp_path: Path) -> None:
    assert acb.check(_server_wheel(tmp_path, server=None), "cpu") == []


def test_the_tool_flag_reaches_the_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _server_wheel(tmp_path, mode=0o644)
    assert acb.main(["--backend", "cpu", "--tool", "llama-server", str(wheel)]) == 1
    assert "without the execute bit" in capsys.readouterr().err


def test_a_wheel_without_the_named_library_passes_without_the_flag(tmp_path: Path) -> None:
    wheel = _server_wheel(tmp_path, base=False)
    assert acb.main(["--tool", "llama-server", str(wheel)]) == 0


def test_a_wheel_without_the_named_library_fails_with_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _server_wheel(tmp_path, base=False)
    args = ["--tool", "llama-server", "--library", "ggml-base", str(wheel)]
    assert acb.main(args) == 1
    assert "no library file for 'ggml-base'" in capsys.readouterr().err


def test_a_wheel_with_the_named_library_passes_with_the_flag(tmp_path: Path) -> None:
    wheel = _server_wheel(tmp_path)
    assert acb.main(["--tool", "llama-server", "--library", "ggml-base", str(wheel)]) == 0


def test_every_named_library_is_required(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _server_wheel(tmp_path)
    args = ["--tool", "llama-server", "--library", "ggml-base", "--library", "ggml-rpc", str(wheel)]
    assert acb.main(args) == 1
    err = capsys.readouterr().err
    assert "'ggml-rpc'" in err
    assert "'ggml-base'" not in err


def test_a_library_without_a_tool_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _server_wheel(tmp_path)
    assert acb.main(["--library", "ggml-base", str(wheel)]) == 2
    assert "usage:" in capsys.readouterr().err


_WORKFLOWS = _SCRIPT.parents[2] / ".github" / "workflows"


def _bundle_check_calls(workflow: Path) -> list[list[str]]:
    """Return the argument list of each assert_engine_bundle.py call in a workflow's run scripts."""
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    calls: list[list[str]] = []
    for job in document["jobs"].values():
        for step in job["steps"]:
            script = step.get("run", "").replace("\\\n", " ")
            lexer = shlex.shlex(script, posix=True, punctuation_chars=";&|()\n")
            lexer.whitespace = " \t\r"
            lexer.whitespace_split = True
            lexer.commenters = "#"
            words = list(lexer)
            for index, word in enumerate(words):
                if word.endswith("assert_engine_bundle.py"):
                    end = index + 1
                    while end < len(words) and not set(words[end]) <= set(";&|()\n"):
                        end += 1
                    calls.append(words[index + 1 : end])
    return calls


def _flag_values(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == flag]


def _assert_asks_for_ggml_base(workflow: Path) -> None:
    calls = _bundle_check_calls(workflow)
    assert calls
    for args in calls:
        assert _flag_values(args, "--tool") == ["llama-server"]
        assert _flag_values(args, "--library") == ["ggml-base"]


@pytest.mark.parametrize("workflow", ["verify-release.yml", "build-multigpu.yml"])
def test_the_release_workflows_ask_for_ggml_base(workflow: str) -> None:
    _assert_asks_for_ggml_base(_WORKFLOWS / workflow)


def test_a_comment_holding_the_flags_does_not_count_as_the_call(tmp_path: Path) -> None:
    workflow = tmp_path / "wf.yml"
    workflow.write_text(
        "jobs:\n"
        "  check:\n"
        "    steps:\n"
        "      - run: |\n"
        "          python tools/qa/assert_engine_bundle.py engine/*.whl"
        " # --tool llama-server --library ggml-base\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        _assert_asks_for_ggml_base(workflow)
    assert _bundle_check_calls(workflow) == [["engine/*.whl"]]


def test_a_flag_in_another_command_does_not_count_as_the_call(tmp_path: Path) -> None:
    workflow = tmp_path / "wf.yml"
    workflow.write_text(
        "jobs:\n"
        "  check:\n"
        "    steps:\n"
        "      - run: |\n"
        "          python tools/qa/assert_engine_bundle.py \\\n"
        "            engine/*.whl && echo --tool llama-server --library ggml-base\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError):
        _assert_asks_for_ggml_base(workflow)


def _bin_dir(tmp_path: Path, bin_files: tuple[str, ...]) -> Path:
    root = tmp_path / "dist" / "lilbee.dist" / "lilbee_engine" / "bin"
    for entry in bin_files:
        (root / entry).parent.mkdir(parents=True, exist_ok=True)
        (root / entry).write_bytes(b"x")
    return root


def test_a_directory_with_every_windows_file_passes(tmp_path: Path) -> None:
    root = _bin_dir(
        tmp_path, ("llama-server.exe", "ggml-cuda.dll", *_WINDOWS_RUNTIME, *_WINDOWS_CRT)
    )
    assert acb.check(root, "cu124", "Windows") == []


def test_a_directory_without_vcomp140_is_rejected_as_an_executable(tmp_path: Path) -> None:
    files = ("llama-server.exe", "ggml-cuda.dll", *_WINDOWS_RUNTIME, *_WINDOWS_CRT[:-1])
    problems = acb.check(_bin_dir(tmp_path, files), "cu124", "Windows")
    assert len(problems) == 1
    assert "windows executable is missing vcomp140.dll" in problems[0]


def test_vcomp140_beside_the_bin_directory_does_not_count(tmp_path: Path) -> None:
    root = _bin_dir(
        tmp_path, ("llama-server.exe", "ggml-cuda.dll", *_WINDOWS_RUNTIME, *_WINDOWS_CRT[:-1])
    )
    (root.parent.parent / "vcomp140.dll").write_bytes(b"x")
    assert any("vcomp140.dll" in p for p in acb.check(root, "cu124", "Windows"))


def test_a_directory_rocm_runtime_is_read_from_nested_kernels(tmp_path: Path) -> None:
    files = ("libggml-hip.so.0.15.1", *_ROCM_PAYLOAD[1:])
    root = _bin_dir(tmp_path, files)
    assert acb.check(root, "rocm", "Linux") == []
    (root / "rocblas" / "library" / "kernels.dat").unlink()
    problems = acb.check(root, "rocm", "Linux")
    assert len(problems) == 1
    assert "Tensile kernels" in problems[0]


def test_a_directory_missing_its_backend_library_is_rejected(tmp_path: Path) -> None:
    root = _bin_dir(tmp_path, ("llama-server", "libggml-base.so"))
    problems = acb.check(root, "vulkan", "Linux")
    assert any("vulkan executable is missing libggml-vulkan.so" in p for p in problems)
    assert any("libvulkan.so.1" in p for p in problems)


def test_a_directory_has_no_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _bin_dir(tmp_path, ("llama-server", "libggml-metal.dylib"))
    monkeypatch.setattr(acb, "WHEEL_SIZE_LIMIT", 0)
    assert acb.check(root, "metal", "macOS") == []


def test_an_empty_directory_has_no_payload(tmp_path: Path) -> None:
    root = tmp_path / "bin"
    root.mkdir()
    assert "no lilbee_engine_llama_cpp/bin/ payload" in acb.check(root, "cpu", "Linux")[0]


def test_main_checks_a_directory_with_a_platform(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _bin_dir(tmp_path, ("llama-server", "libggml-cuda.so"))
    args = ["--backend", "cu125", "--platform", "Linux", str(root)]
    assert acb.main(args) == 1
    assert "libcudart.so" in capsys.readouterr().err
    (root / "libcudart.so.12").write_bytes(b"x")
    (root / "libcublas.so.12").write_bytes(b"x")
    (root / "libcublasLt.so.12").write_bytes(b"x")
    assert acb.main(args) == 0


def test_main_refuses_a_directory_without_a_platform(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _bin_dir(tmp_path, ("llama-server",))
    assert acb.main(["--backend", "cpu", str(root)]) == 1
    assert "needs --platform" in capsys.readouterr().err


def test_main_refuses_the_tool_check_on_a_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _bin_dir(tmp_path, ("llama-server", "libggml-cpu.so"))
    args = ["--backend", "cpu", "--platform", "Linux", "--tool", "llama-server", str(root)]
    assert acb.main(args) == 1
    assert "--tool reads a wheel, not a directory" in capsys.readouterr().err


def test_main_refuses_a_directory_without_a_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _bin_dir(tmp_path, ("llama-server", "libggml-vulkan.so"))
    assert acb.main(["--platform", "Linux", str(root)]) == 1
    assert "needs --backend" in capsys.readouterr().err
