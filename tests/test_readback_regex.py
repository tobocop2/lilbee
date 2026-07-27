"""The buffer regex, derived from every emit site in llama.cpp rather than samples."""

from __future__ import annotations

import pytest

from lilbee.providers.fleet.readback import MIB, parse_device_buffers

# Every "<n> buffer size = <n> MiB" format string in llama.cpp, rendered. Found by
# grepping src/, ggml/src/, tools/ and common/ rather than by reading two logs.
_EMITS = [
    ("model", "load_tensors: %12s model buffer size = %8.2f MiB"),
    ("KV", "llama_kv_cache: %10s KV buffer size = %8.2f MiB"),
    ("compute", "sched_reserve: %10s compute buffer size = %8.2f MiB"),
    ("output", "llama_context: %10s  output buffer size = %8.2f MiB"),
    ("LoRA", "llama_adapter: %10s LoRA buffer size = %8.2f MiB"),
    ("RS", "llama_memory_recurrent: %10s RS buffer size = %8.2f MiB"),
    ("DSV4 state", "llama_kv_cache: %10s DSV4 shift state buffer size = %8.2f MiB"),
]

# Lines that contain "buffer size" and must NOT be read as an allocation.
_NOT_ALLOCATIONS = [
    "~llama_context:  MTL0 compute buffer size is 97.1250 MiB, matches expectation of 97.1250 MiB",
    "~llama_context:  MTL0 compute buffer size of 97.1250 MiB, does not match expectation of 1 MiB",
    "ggml_backend: copy buffer size: 128 MB",
    "ggml_opencl: A_q_d buffer size reduced from 100 to 50 due to device limitations.",
    "ggml_opencl: device max image buffer size (pixels): 16384",
]


class TestEveryUpstreamEmitIsRead:
    """A kind this does not know is VRAM it does not count.

    RS is recurrent state, which every Mamba and RWKV model allocates, and LoRA
    is every adapter. Enumerating kinds by hand meant those were silently zero.
    """

    @pytest.mark.parametrize(("kind", "template"), _EMITS, ids=[k for k, _ in _EMITS])
    def test_it_is_counted(self, kind: str, template: str) -> None:
        line = template % ("CUDA0", 128.0)
        assert parse_device_buffers(line) == {"CUDA0": int(128.0 * MIB)}


class TestNothingElseIsMistakenForOne:
    """Several upstream lines carry the words and are not allocations."""

    @pytest.mark.parametrize("line", _NOT_ALLOCATIONS)
    def test_it_is_ignored(self, line: str) -> None:
        assert parse_device_buffers(line) == {}


class TestTheTimestampPrefixDoesNotHideThem:
    """--log-file prefixes every line with a timestamp and level."""

    def test_a_prefixed_line_still_parses(self) -> None:
        line = "0.00.220.431 I load_tensors:         MTL0 model buffer size =    82.41 MiB"
        assert parse_device_buffers(line) == {"MTL0": int(82.41 * MIB)}


class TestTheDriftAlarmSurvivesUpstreamRewording:
    """The alarm is armed by a phrase the engine keeps rewriting. Pinning the old
    wording meant a newer engine loaded, parsed to nothing, and said nothing."""

    def test_the_wording_through_b9665_still_counts(self) -> None:
        from lilbee.providers.fleet.readback import load_finished

        assert load_finished("0.00 I load_model:   initializing slots, n_slots = 4\n")

    def test_the_wording_from_b9829_counts_too(self) -> None:
        from lilbee.providers.fleet.readback import load_finished

        assert load_finished("0.00 I load_model:   initializing, n_slots = 4, n_ctx_slot = 4096\n")

    def test_an_unrelated_line_does_not(self) -> None:
        from lilbee.providers.fleet.readback import load_finished

        assert not load_finished("srv    load_model: loading model\n")


class TestBuffersThatBelongToNoOneCard:
    def test_a_row_split_buffer_makes_no_phantom_device(self) -> None:
        from lilbee.providers.fleet.readback import parse_device_buffers

        text = (
            "load_tensors:        SYCL0 model buffer size =   100.00 MiB\n"
            "load_tensors:   SYCL_Split model buffer size =   200.00 MiB\n"
        )
        assert set(parse_device_buffers(text)) == {"SYCL0"}

    def test_an_amx_repack_buffer_is_host_memory(self) -> None:
        from lilbee.providers.fleet.readback import device_footprint

        text = (
            "load_tensors:        CUDA0 model buffer size =   100.00 MiB\n"
            "load_tensors:          AMX model buffer size =   500.00 MiB\n"
        )
        assert device_footprint(text) == 100 * 1024 * 1024


class TestARealVulkanLoad:
    """Captured from a GTX 1070 Ti on Linux, engine build 9665 (e3a74b299).

    Vulkan is where every AMD and Intel GPU lands, and until this capture the
    module's treatment of it was read from ggml's source rather than observed:
    in particular the claim that every backend names its pinned-host allocator
    "<backend>_Host", which had only ever been seen on CUDA.
    """

    @staticmethod
    def _log() -> str:
        from pathlib import Path

        return (Path(__file__).parent / "fixtures" / "engine-load-vulkan.log").read_text()

    def test_the_device_and_its_host_allocator_both_parse(self) -> None:
        from lilbee.providers.fleet.readback import parse_device_buffers

        assert set(parse_device_buffers(self._log())) == {"CPU", "Vulkan0", "Vulkan_Host"}

    def test_only_the_card_counts_toward_the_gpu_footprint(self) -> None:
        from lilbee.providers.fleet.readback import device_footprint

        # 98.87 model + 45.00 KV + 13.26 compute; CPU_Mapped and Vulkan_Host out.
        assert round(device_footprint(self._log()) / 1024**2, 2) == 157.13

    def test_the_load_finished_marker_matches_this_build(self) -> None:
        from lilbee.providers.fleet.readback import engine_build, load_finished

        assert load_finished(self._log())
        assert engine_build(self._log()) == "9665 (e3a74b299)"
