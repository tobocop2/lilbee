"""The CUDA guard must survive a runtime major bump."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.providers.fleet import cuda_runtime


class TestTheGuardFollowsTheRuntimeMajor:
    """Pinning the soname to .so.12 made a cu13 build invisible to the guard.

    A newer engine links libcudart.so.13. The substring test never matched it, so
    the whole check returned early and a cu13 build that could not initialize a
    device fell silently to CPU: the exact failure this guard exists to catch.
    """

    @pytest.mark.parametrize(
        ("ldd_line", "expected"),
        [
            ("\tlibcudart.so.12 => /usr/lib/libcudart.so.12", 12),
            ("\tlibcudart.so.13 => /usr/lib/libcudart.so.13", 13),
            ("\tlibcudart.so.12.4.127 => /usr/lib/libcudart.so.12.4.127", 12),
            ("\tlibcublas.so.13 => /usr/lib/libcublas.so.13", 13),
            ("\tlibfoo.so.1 => /usr/lib/libfoo.so.1", None),
        ],
        ids=["cu12", "cu13", "cu12-full-version", "cublas-13", "unrelated"],
    )
    def test_the_linked_major_is_read_not_matched(
        self, ldd_line: str, expected: int | None
    ) -> None:
        assert cuda_runtime._linked_cuda_major(ldd_line) == expected

    def test_a_cu13_build_is_recognised_as_a_cuda_build(self, monkeypatch) -> None:
        monkeypatch.setattr(
            cuda_runtime,
            "_ldd_output",
            lambda _b, _e: "\tlibcudart.so.13 => /usr/lib/libcudart.so.13",
        )
        assert cuda_runtime._links_cuda_runtime(Path("/bin/llama-server"), {}) is True


class TestTheGuardNamesMIG:
    """A MIG parent answers has_nvidia_gpu while CUDA enumerates zero devices.

    The cause list blamed a driver mismatch and a visibility mask, neither of
    which is the cause, so the one host shape that reliably produces this exact
    symptom was the one shape the message did not mention.
    """

    def test_mig_is_among_the_listed_causes(self, monkeypatch) -> None:
        from lilbee.providers.base import ProviderError

        monkeypatch.setattr(cuda_runtime.sys, "platform", "linux")
        monkeypatch.setattr(cuda_runtime, "_links_cuda_runtime", lambda _b, _e: True)
        monkeypatch.setattr(cuda_runtime.model_cache, "has_nvidia_gpu", lambda: True)
        with pytest.raises(ProviderError) as excinfo:
            cuda_runtime.assert_cuda_devices_usable(Path("/bin/x"), [], "no CUDA devices found")
        assert "MIG" in str(excinfo.value)
