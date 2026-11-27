"""What lilbee reads to decide why an engine died.

Two ways to get this wrong: look at the wrong end of the log, or match a line
that is not a death at all. Both make the recovery loop act on fiction.
"""

from __future__ import annotations

import pytest

from lilbee.providers.base import ProviderErrorKind
from lilbee.providers.fleet.client import _UPSTREAM_LOG_TAIL_CHARS, classify_upstream_death


class TestTheTailIsActuallyTheTail:
    def test_a_log_larger_than_one_chunk_still_yields_its_last_line(self, monkeypatch) -> None:
        # llama-swap replays a model's whole ring in one write and its ring is
        # 100KB, while httpx hands over at most 64KB per chunk. Stopping at the
        # first chunk returns the head of a warm model's log, and the fatal line
        # is always the last one.
        import lilbee.providers.fleet.client as client_mod

        fatal = "ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory"
        body = ("routine chatter\n" * 8000) + fatal

        class _Stream:
            def iter_text(self):
                for i in range(0, len(body), 65536):
                    yield body[i : i + 65536]

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        monkeypatch.setattr(client_mod.httpx, "stream", lambda *_a, **_k: _Stream())
        tail = client_mod._fetch_log_tail("http://127.0.0.1:1/logs/x")
        assert fatal in tail
        assert len(tail) <= _UPSTREAM_LOG_TAIL_CHARS


class TestAWarningIsNotADeath:
    def test_the_vulkan_pinned_memory_warning_is_not_capacity(self) -> None:
        # Non-fatal, and the engine falls back to unpinned and keeps working.
        tail = (
            "WARNING: failed to allocate 256.00 MB of pinned memory\n"
            "ggml_vulkan: unsupported op, aborting\n"
        )
        assert classify_upstream_death(tail) is not ProviderErrorKind.CAPACITY

    def test_a_real_allocation_failure_beside_a_warning_still_counts(self) -> None:
        tail = (
            "WARNING: failed to allocate 256.00 MB of pinned memory\n"
            "llama_init_from_model: failed to allocate compute buffers\n"
        )
        assert classify_upstream_death(tail) is ProviderErrorKind.CAPACITY


class TestBackendsThatWordItTheirOwnWay:
    def test_a_sycl_device_allocation_failure_counts(self) -> None:
        tail = (
            "Native API failed. Native API returns: "
            "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY\nException caught at file:ggml-sycl.cpp\n"
        )
        assert classify_upstream_death(tail) is ProviderErrorKind.CAPACITY

    def test_a_sycl_resource_exhaustion_counts(self) -> None:
        tail = "PI_ERROR_OUT_OF_RESOURCES\nException caught at file:ggml-sycl.cpp, line:432\n"
        assert classify_upstream_death(tail) is ProviderErrorKind.CAPACITY


class TestAMissingEngineProgram:
    @pytest.mark.parametrize(
        "tail",
        [
            'exec: "llama-server": executable file not found in $PATH',
            "fork/exec /opt/engine/llama-server: no such file or directory",
            "sh: 1: /opt/engine/llama-server: not found",
            "bash: line 1: /opt/engine/llama-server: No such file or directory",
            "bash: llama-server: command not found",
            'exec: "C:\\\\engine\\\\llama-server.exe": file does not exist',
        ],
    )
    def test_each_launcher_wording_is_a_missing_engine(self, tail: str) -> None:
        assert classify_upstream_death(tail) is ProviderErrorKind.ENGINE_MISSING

    def test_a_missing_model_file_is_not_a_missing_engine(self) -> None:
        # The engine ran and could not open its model: the program is there.
        tail = (
            "gguf_init_from_file: failed to open GGUF file '/m/x.gguf' (No such file or directory)"
        )
        assert classify_upstream_death(tail) is not ProviderErrorKind.ENGINE_MISSING

    def test_a_cause_the_engine_printed_wins_over_an_exec_error(self) -> None:
        # A tail can span an attempt that found no program and a later one that ran
        # out of memory. The memory shortfall is what a retry has to fix.
        tail = (
            "fork/exec /opt/engine/llama-server: no such file or directory\n"
            "ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory"
        )
        assert classify_upstream_death(tail) is ProviderErrorKind.CAPACITY
