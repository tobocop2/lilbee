"""Tests for the engine-bundle gate that keeps a backend-less wheel off a release."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

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
_WINDOWS_CRT = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
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
            zf.writestr(f"lilbee_engine/bin/{entry}", b"x")
    return path


def test_linux_cuda_wheel_without_runtime_is_rejected(tmp_path: Path) -> None:
    # The shipped-broken shape: ggml-cuda present, the runtime it links absent.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so"),
    )
    problems = acb.check(wheel)
    assert len(problems) == 3
    assert any("libcudart.so" in p for p in problems)
    assert any("libcublasLt.so" in p for p in problems)


def test_linux_cuda_wheel_with_runtime_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so", *_LINUX_RUNTIME),
    )
    assert acb.check(wheel) == []


def test_windows_cuda_wheel_without_runtime_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cuda.dll", *_WINDOWS_CRT),
    )
    assert len(acb.check(wheel)) == 3


def test_windows_cuda_wheel_with_runtime_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cuda.dll", *_WINDOWS_RUNTIME, *_WINDOWS_CRT),
    )
    assert acb.check(wheel) == []


def test_cuda_detected_from_payload_not_filename(tmp_path: Path) -> None:
    # A freshly built wheel has no -1.cu125- build tag; that is added at attach time.
    # Detection must still fire, or the gate silently passes every build artifact.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cuda.dll", *_WINDOWS_CRT),
    )
    assert len(acb.check(wheel)) == 3


def test_non_cuda_wheel_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cpu-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", *_LINUX_VARIANTS),
    )
    assert acb.check(wheel) == []


def test_non_cuda_wheel_carrying_cuda_runtime_is_rejected(tmp_path: Path) -> None:
    # Guards the other direction: the copy step must stay scoped to CUDA backends.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.vulkan-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-vulkan.dll", "cudart64_12.dll", *_WINDOWS_CRT),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "non-CUDA wheel ships CUDA runtime" in problems[0]


def test_windows_wheel_without_the_msvc_runtime_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.vulkan-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-vulkan.dll", "msvcp140.dll"),
    )
    problems = acb.check(wheel)
    assert [p for p in problems if "vcruntime140.dll" in p]
    assert [p for p in problems if "vcruntime140_1.dll" in p]
    assert not [p for p in problems if "msvcp140.dll" in p]


def test_empty_bin_dir_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lilbee_engine-1.0-py3-none-win_amd64.whl"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("lilbee_engine/__init__.py", b"")
    assert acb.check(path) == [f"{path.name}: no lilbee_engine/bin/ payload at all"]


def test_macos_cuda_payload_asserts_nothing(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-py3-none-macosx_11_0_arm64.whl",
        ("llama-server", "libggml-cuda.so"),
    )
    assert acb.check(wheel) == []


def test_rocm_wheel_without_the_hip_backend_is_rejected(tmp_path: Path) -> None:
    # The shape that shipped: BACKEND=rocm, and the bundle holds only the CPU
    # backend because the cmake flag naming it was ignored.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-base.so", "libggml-cpu.so"),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "libggml-hip.so" in problems[0]


def test_rocm_wheel_with_the_hip_backend_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so", "libggml-hip.so", *_ROCM_RUNTIME),
    )
    assert acb.check(wheel) == []


def test_a_versioned_soname_alone_satisfies_the_backend_check(tmp_path: Path) -> None:
    # The engine links by SONAME, so the versioned file is the real library and
    # the bare name is a symlink beside it. A bundle that carries only the former
    # is complete, and failing it would be a false alarm.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so", "libggml-hip.so.0.15.1", *_ROCM_RUNTIME),
    )
    assert acb.check(wheel) == []


def test_backend_argument_covers_a_wheel_built_before_its_tag_is_added(tmp_path: Path) -> None:
    # A freshly built artifact has no -1.rocm- tag, and that is exactly where the
    # gate has to fire: at build time, on the runner that produced it.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    assert acb.check(wheel) == []
    problems = acb.check(wheel, backend="rocm")
    assert len(problems) == 1
    assert "libggml-hip.so" in problems[0]


def test_sycl_wheel_without_its_backend_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.sycl-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    assert len(acb.check(wheel)) == 1


def test_windows_vulkan_wheel_without_its_backend_is_rejected(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.vulkan-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-cpu.dll", *_WINDOWS_CRT),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "ggml-vulkan.dll" in problems[0]


def test_macos_metal_wheel_checks_the_dylib_name(tmp_path: Path) -> None:
    missing = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.metal-py3-none-macosx_11_0_arm64.whl",
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
        "lilbee_engine-1.0-1.cu124-py3-none-manylinux_2_17_x86_64.whl",
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
        "lilbee_engine-1.0-1.cann-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    assert acb.check(wheel) == []


def test_rocm_wheel_without_the_runtime_is_rejected(tmp_path: Path) -> None:
    # The backend is present but the libraries it links are not, so it loads on a
    # machine with ROCm installed and silently falls back to CPU everywhere else.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so", "libggml-hip.so"),
    )
    problems = acb.check(wheel)
    assert len(problems) == len(_ROCM_RUNTIME)
    assert any("libamdhip64" in p for p in problems)
    assert any("rocblas/library" in p for p in problems)


def test_rocm_wheel_with_the_runtime_passes(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so", "libggml-hip.so", *_ROCM_RUNTIME),
    )
    assert acb.check(wheel) == []


def test_rocm_wheel_without_the_tensile_kernels_is_rejected(tmp_path: Path) -> None:
    # rocBLAS loads these as data at runtime rather than linking them, so the
    # library resolves and then fails on the first matrix multiply.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
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
        "lilbee_engine-1.0-1.rocm-py3-none-manylinux_2_17_x86_64.whl",
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
        "lilbee_engine-1.0-1.vulkan-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-vulkan.so", "libvulkan.so.1", *_LINUX_VARIANTS),
    )
    assert acb.check(wheel) == []


def test_cpu_linux_wheel_with_a_single_cpu_library_is_rejected(tmp_path: Path) -> None:
    # The pre-dispatch shape: one libggml-cpu.so carrying only the AVX baseline.
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cpu-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "libggml-cpu-<variant>.so" in problems[0]


def test_cpu_windows_wheel_needs_variant_modules(tmp_path: Path) -> None:
    single = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cpu-py3-none-win_amd64.whl",
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
        "lilbee_engine-1.0-1.cpu-py3-none-manylinux_2_17_x86_64.whl",
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
        "lilbee_engine-1.0-1.vulkan-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-vulkan.so", *_LINUX_VARIANTS),
    )
    problems = acb.check(wheel)
    assert len(problems) == 1
    assert "libvulkan.so.1" in problems[0]


def test_vulkan_windows_wheel_is_not_asked_for_variants_or_loader(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.vulkan-py3-none-win_amd64.whl",
        ("llama-server.exe", "ggml-vulkan.dll", "ggml-cpu.dll", *_WINDOWS_CRT),
    )
    assert acb.check(wheel) == []


def test_main_passes_the_backend_through(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cpu.so"),
    )
    assert acb.main(["--backend", "rocm", str(wheel)]) == 1
    assert "libggml-hip.so" in capsys.readouterr().err


def test_main_returns_nonzero_and_names_the_wheel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
        ("llama-server", "libggml-cuda.so"),
    )
    assert acb.main([str(wheel)]) == 1
    assert "libcudart.so" in capsys.readouterr().err


def test_main_returns_zero_on_a_good_wheel(tmp_path: Path) -> None:
    wheel = _wheel(
        tmp_path,
        "lilbee_engine-1.0-1.cu125-py3-none-manylinux_2_17_x86_64.whl",
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
_ROCM_WHEEL = "lilbee_engine-0.1-1.rocm-py3-none-manylinux_2_17_x86_64.whl"


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
