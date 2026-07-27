"""Tests for reading the engine's real footprint back out of its startup report."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lilbee.providers.fleet.readback import (
    MIB,
    device_footprint,
    load_finished,
    parse_device_buffers,
    report_divergence,
)
from lilbee.providers.roles import WorkerRole


def _mib(*values: float) -> int:
    """Bytes for a set of MiB figures, truncated per line as the parser does."""
    return sum(int(value * MIB) for value in values)


# Real llama.cpp startup output, trimmed to the lines that report allocations.
_CUDA_LOAD = """
llama_model_loader: loaded meta data with 30 key-value pairs
load_tensors: offloading 36 repeating layers to GPU
load_tensors: offloaded 37/37 layers to GPU
load_tensors:        CUDA0 model buffer size =  4589.31 MiB
load_tensors:   CPU_Mapped model buffer size =   315.30 MiB
llama_context: n_ctx = 8192
llama_kv_cache_unified:      CUDA0 KV buffer size =  1152.00 MiB
llama_context:      CUDA0 compute buffer size =   304.00 MiB
llama_context:        CPU compute buffer size =    24.01 MiB
"""

_SPLIT_LOAD = """
load_tensors:        CUDA0 model buffer size =  2000.00 MiB
load_tensors:        CUDA1 model buffer size =  2048.00 MiB
llama_kv_cache_unified:      CUDA0 KV buffer size =   512.00 MiB
llama_kv_cache_unified:      CUDA1 KV buffer size =   512.00 MiB
"""


class TestParseDeviceBuffers:
    def test_sums_model_kv_and_compute_per_device(self) -> None:
        buffers = parse_device_buffers(_CUDA_LOAD)
        assert buffers["CUDA0"] == _mib(4589.31, 1152.00, 304.00)
        # CPU_Mapped folds into CPU: the mmapped weights are host memory too.
        assert buffers["CPU"] == _mib(315.30, 24.01)

    def test_keeps_each_card_of_a_split_apart(self) -> None:
        buffers = parse_device_buffers(_SPLIT_LOAD)
        assert buffers == {
            "CUDA0": _mib(2000.00, 512.00),
            "CUDA1": _mib(2048.00, 512.00),
        }

    def test_a_log_with_no_buffer_report_parses_empty(self) -> None:
        # An older engine, a load that died before allocating, or a rotated log.
        assert parse_device_buffers("llama_model_loader: loaded meta data\n") == {}


class TestDeviceFootprint:
    def test_excludes_host_buffers(self) -> None:
        # CPU_Mapped is the mmapped weights and CPU is host scratch. Charging
        # either against a card reports a phantom overrun on every partial offload.
        assert device_footprint(_CUDA_LOAD) == _mib(4589.31, 1152.00, 304.00)

    def test_sums_every_card_of_a_split(self) -> None:
        assert device_footprint(_SPLIT_LOAD) == _mib(2000.00, 512.00) + _mib(2048.00, 512.00)


class TestReportDivergence:
    def test_warns_when_the_engine_used_materially_more(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            warned = report_divergence(
                WorkerRole.CHAT, "org/m.gguf", 4 * 1024**3, 6 * 1024**3, tolerance=0.15
            )
        assert warned is True
        assert "allocated 6.0 GiB" in caplog.text
        assert "planned for 4.0 GiB" in caplog.text
        assert "+50%" in caplog.text

    def test_warns_when_the_estimate_was_far_too_large(self, caplog) -> None:
        # Quieter, but it is why a role gets fewer slots or a split it did not need.
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            warned = report_divergence(
                WorkerRole.RERANK, "org/r.gguf", 8 * 1024**3, 2 * 1024**3, tolerance=0.15
            )
        assert warned is True
        assert "-75%" in caplog.text

    def test_stays_quiet_inside_the_tolerance(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            warned = report_divergence(
                WorkerRole.CHAT, "org/m.gguf", 4 * 1024**3, int(4.3 * 1024**3), tolerance=0.15
            )
        assert warned is False
        assert caplog.text == ""

    def test_an_unparsed_or_unestimated_instance_says_nothing(self, caplog) -> None:
        # No buffer report, or a model enrolled at its file size with no estimate:
        # there is no comparison to make, and a warning would be noise.
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            assert report_divergence(WorkerRole.CHAT, "m", 0, 6 * 1024**3, tolerance=0.15) is False
            assert report_divergence(WorkerRole.CHAT, "m", 4 * 1024**3, 0, tolerance=0.15) is False
        assert caplog.text == ""


class TestAgainstRealEngineOutput:
    """The fixture is a real llama-server load, captured with --log-file -lv 4.

    Everything above is a hand-written sample and would keep passing if the
    engine's format moved. This one fails when it does.
    """

    @staticmethod
    def _fixture() -> str:
        return (Path(__file__).parent / "fixtures" / "engine-load-metal.log").read_text()

    def test_finds_every_buffer_the_engine_reported(self) -> None:
        buffers = parse_device_buffers(self._fixture())
        # Weights and scratch on the GPU, weights and output on the host.
        assert set(buffers) == {"MTL0", "CPU"}

    def test_folds_the_mapped_buffer_into_its_own_device(self) -> None:
        # The engine reports MTL0_Mapped beside MTL0; both are that card's memory.
        buffers = parse_device_buffers(self._fixture())
        assert buffers["MTL0"] == _mib(82.41, 45.00, 97.12)

    def test_charges_only_the_gpu(self) -> None:
        # CPU_Mapped weights and the CPU output/compute buffers are host memory.
        assert device_footprint(self._fixture()) == _mib(82.41, 45.00, 97.12)


class TestTheCheckRunsOnARealLog:
    """The whole path: engine log on disk, estimate in hand, one warning."""

    def test_warns_using_the_engine_s_own_report(self, tmp_path, caplog) -> None:
        from lilbee.providers.fleet.readback import check_launch, engine_log_path

        log = engine_log_path(tmp_path, "chat-0")
        log.write_text((Path(__file__).parent / "fixtures" / "engine-load-metal.log").read_text())
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            # The engine really allocated ~0.22 GiB; planning charged 4 GiB.
            warned = check_launch(tmp_path, "chat-0", WorkerRole.CHAT, "org/m.gguf", 4 * 1024**3)
        assert warned is True
        assert "planned for 4.0 GiB" in caplog.text

    def test_a_missing_log_says_nothing(self, tmp_path, caplog) -> None:
        from lilbee.providers.fleet.readback import check_launch

        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            assert check_launch(tmp_path, "chat-0", WorkerRole.CHAT, "m", 4 * 1024**3) is False
        assert caplog.text == ""

    def test_the_engine_is_told_where_to_write_and_how_loudly(self, tmp_path) -> None:
        from lilbee.providers.fleet.readback import (
            ENV_LOG_FILE,
            ENV_LOG_VERBOSITY,
            engine_log_env,
        )

        env = engine_log_env(tmp_path, "rerank-1")
        assert env[ENV_LOG_FILE].endswith("engine-rerank-1.log")
        # Below this the engine prints no buffer report at all, so the check
        # would silently never fire.
        assert env[ENV_LOG_VERBOSITY] == "4"


class TestFormatDriftIsLoud:
    """The engine's log is not a contract, so its silence must not be.

    llama-server exposes no memory over its API: /props carries none of it and
    /metrics is token counters. The log is the only source, which makes a format
    change a real event this has to survive noisily rather than quietly.
    """

    _MOVED = (
        "srv    load_model: loading model 'm.gguf'\n"
        "load_tensors: CUDA0 weights arena = 8192.00 MiB\n"
        "srv    load_model: initializing slots, n_slots = 4\n"
    )

    def test_a_finished_load_with_no_readable_report_says_so(self, tmp_path, caplog) -> None:
        from lilbee.providers.fleet.readback import check_launch, engine_log_path
        from lilbee.providers.roles import WorkerRole

        engine_log_path(tmp_path, "chat-0").write_text(self._MOVED)
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            check_launch(tmp_path, "chat-0", WorkerRole.CHAT, "m", 4 * 1024**3)
        assert "reported no memory usage where lilbee reads it" in caplog.text
        assert "unverified" in caplog.text

    def test_a_load_still_in_progress_stays_quiet(self, tmp_path, caplog) -> None:
        # The report is written during load; asking early is not a format change.
        from lilbee.providers.fleet.readback import check_launch, engine_log_path
        from lilbee.providers.roles import WorkerRole

        engine_log_path(tmp_path, "chat-0").write_text("srv load_model: loading model 'm.gguf'\n")
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            check_launch(tmp_path, "chat-0", WorkerRole.CHAT, "m", 4 * 1024**3)
        assert caplog.text == ""

    def test_the_real_capture_shows_a_finished_load(self) -> None:
        # Pins the completion marker against real output, so the branch above
        # cannot silently stop firing either.
        from lilbee.providers.fleet.readback import load_finished

        assert load_finished(
            (Path(__file__).parent / "fixtures" / "engine-load-metal.log").read_text()
        )

    def test_the_drift_warning_names_both_builds(self, tmp_path, caplog) -> None:
        # The report has to say what was seen and what it was verified against,
        # or the reader cannot tell which one moved.
        from lilbee.providers.fleet.readback import (
            VERIFIED_ENGINE_BUILD,
            check_launch,
            engine_log_path,
        )
        from lilbee.providers.roles import WorkerRole

        engine_log_path(tmp_path, "chat-0").write_text(
            "common_params_print_info: build 9999 (deadbee) with clang for Linux\n" + self._MOVED
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            check_launch(tmp_path, "chat-0", WorkerRole.CHAT, "m", 4 * 1024**3)
        assert "9999 (deadbee)" in caplog.text
        assert VERIFIED_ENGINE_BUILD in caplog.text

    def test_the_verified_build_matches_the_fixture(self) -> None:
        # The constant is the pin. If the fixture is re-captured from a newer
        # engine without updating it, the pin has stopped meaning anything.
        from lilbee.providers.fleet.readback import VERIFIED_ENGINE_BUILD, engine_build

        captured = engine_build(
            (Path(__file__).parent / "fixtures" / "engine-load-metal.log").read_text()
        )
        assert captured == VERIFIED_ENGINE_BUILD

    def test_a_log_with_no_build_line_still_warns(self, tmp_path, caplog) -> None:
        from lilbee.providers.fleet.readback import check_launch, engine_log_path
        from lilbee.providers.roles import WorkerRole

        engine_log_path(tmp_path, "chat-0").write_text(self._MOVED)
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            check_launch(tmp_path, "chat-0", WorkerRole.CHAT, "m", 4 * 1024**3)
        assert "build unknown" in caplog.text


class TestTheUpstreamFormatStrings:
    """Built from llama.cpp's own printf specifiers, not from a captured sample.

    The fixture proves the parser handled one real load. This proves it handles
    the format that produced it, including the exact %12s and %10s padding, so a
    reader can compare these three strings against upstream source directly.
    """

    @pytest.mark.parametrize(
        ("source", "template", "device", "mib"),
        [
            (
                "src/llama-model.cpp",
                "load_tensors: %12s model buffer size = %8.2f MiB",
                "CUDA0",
                4589.31,
            ),
            (
                "src/llama-kv-cache.cpp",
                "llama_kv_cache: %10s KV buffer size = %8.2f MiB",
                "CUDA0",
                1152.0,
            ),
            (
                "src/llama-context.cpp",
                "sched_reserve: %10s compute buffer size = %8.2f MiB",
                "MTL0",
                304.0,
            ),
        ],
        ids=["model", "kv", "compute"],
    )
    def test_a_line_rendered_from_upstream_parses(
        self, source: str, template: str, device: str, mib: float
    ) -> None:
        assert source  # names where the format lives, for the next reader
        assert parse_device_buffers(template % (device, mib)) == {device: int(mib * MIB)}


class TestAgainstRealCudaOutput:
    """A two-A40 tensor-split load, captured on real hardware.

    The Metal fixture could not show this: CUDA names its pinned-host allocator
    CUDA_Host, which was being charged to the GPU as though it were a third card.
    """

    @staticmethod
    def _fixture() -> str:
        return (Path(__file__).parent / "fixtures" / "engine-load-cuda-split.log").read_text()

    def test_pinned_host_memory_is_not_charged_to_a_card(self) -> None:
        # CUDA_Host holds the output and compute staging buffers in host RAM.
        assert device_footprint(self._fixture()) == _mib(37.86, 24.00, 24.91) + _mib(
            61.08, 21.00, 24.91
        )

    def test_both_cards_are_reported_separately(self) -> None:
        buffers = parse_device_buffers(self._fixture())
        assert buffers["CUDA0"] == _mib(37.86, 24.00, 24.91)
        assert buffers["CUDA1"] == _mib(61.08, 21.00, 24.91)

    def test_an_even_split_request_did_not_land_evenly(self) -> None:
        # Launched with --tensor-split 1,1 and the cards still differ by 23%.
        # This is the case the per-device check exists for: the total is right.
        buffers = parse_device_buffers(self._fixture())
        assert buffers["CUDA1"] > buffers["CUDA0"] * 1.2

    def test_the_capture_is_from_a_finished_load(self) -> None:
        assert load_finished(self._fixture())


class TestBothEnvSpellingsAreSet:
    """llama.cpp renamed these, so lilbee cannot pick one and be right.

    common/arg.cpp registers LLAMA_ARG_LOG_FILE and LLAMA_ARG_LOG_VERBOSITY on
    current master. Builds around 9310 read the same settings without the ARG,
    verified by running both pairs against one binary: the unprefixed pair
    produced the report and the prefixed pair produced no file at all.
    """

    def test_every_name_carries_the_same_value(self, tmp_path) -> None:
        from lilbee.providers.fleet.readback import (
            ENV_ARG_LOG_FILE,
            ENV_ARG_LOG_VERBOSITY,
            ENV_LOG_FILE,
            ENV_LOG_VERBOSITY,
            engine_log_env,
        )

        env = engine_log_env(tmp_path, "chat-0")
        assert env[ENV_LOG_FILE] == env[ENV_ARG_LOG_FILE]
        assert env[ENV_LOG_VERBOSITY] == env[ENV_ARG_LOG_VERBOSITY] == "4"
        assert env[ENV_LOG_FILE].endswith("engine-chat-0.log")
