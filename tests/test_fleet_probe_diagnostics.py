"""The device probe's failures must reach the user, not vanish."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import devices as devices_mod


class TestASpawnFailureIsReported:
    """An unrunnable probe looked exactly like a host with no GPU."""

    def test_the_reason_the_probe_could_not_run_is_logged(self, monkeypatch, caplog) -> None:
        def _boom(_binary, _timeout):
            raise OSError("Exec format error")

        monkeypatch.setattr(devices_mod, "_run_list_devices", _boom)
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.devices"):
            probe = devices_mod.probe_devices(Path("/bin/llama-server"))
        assert probe.devices == []
        assert probe.spoke_protocol is False
        assert "Exec format error" in caplog.text


class TestATimeoutKeepsWhatTheProbeSaid:
    """The probe's own partial output beats fixed advice about someone else's GPU."""

    def test_partial_output_is_carried_into_the_error(self, monkeypatch) -> None:
        def _timeout(*_a, **_k):
            raise subprocess.TimeoutExpired(
                cmd="llama-server --list-devices",
                timeout=30,
                output=b"ggml_vulkan: device 0 hung\n",
            )

        monkeypatch.setattr(devices_mod, "run_bounded", _timeout)
        with pytest.raises(ProviderError) as excinfo:
            devices_mod._run_list_devices(Path("/bin/llama-server"), 30.0)
        assert "ggml_vulkan: device 0 hung" in str(excinfo.value)

    def test_the_advice_names_every_vendor_not_just_one(self, monkeypatch) -> None:
        # An AMD or Intel host hanging in its own driver was told to run nvidia-smi,
        # a tool it does not have, about a GPU it does not own.
        def _timeout(*_a, **_k):
            raise subprocess.TimeoutExpired(cmd="x", timeout=30)

        monkeypatch.setattr(devices_mod, "run_bounded", _timeout)
        with pytest.raises(ProviderError) as excinfo:
            devices_mod._run_list_devices(Path("/bin/llama-server"), 30.0)
        message = str(excinfo.value)
        for tool in ("nvidia-smi", "rocm-smi", "xpu-smi"):
            assert tool in message


class TestACrashAfterTheHeaderIsNotAnUnsupportedFlag:
    """Two different failures wore the same message."""

    def test_a_probe_that_answered_then_died_says_it_crashed(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(
            devices_mod,
            "_run_list_devices",
            lambda _b, _t: (
                "Available devices:\n  CUDA0: gpu (1 MiB, 1 MiB free)\nSegfault\n",
                139,
            ),
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.devices"):
            devices_mod.probe_devices(Path("/bin/llama-server"))
        assert "does not appear to support" not in caplog.text
        assert "crashed" in caplog.text

    def test_a_probe_that_never_answered_still_says_unsupported(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(
            devices_mod, "_run_list_devices", lambda _b, _t: ("usage: llama-server [options]\n", 1)
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.devices"):
            devices_mod.probe_devices(Path("/bin/llama-server"))
        assert "does not appear to support" in caplog.text


class TestEveryVendorGetsTheWarning:
    """The silent-CPU warning was gated to hosts with an NVIDIA card.

    An Intel Arc or an AMD card the engine failed to enumerate produced exactly
    the same symptom, a fleet planned for CPU, and said nothing at all.
    """

    @staticmethod
    def _probe_reporting_nothing(monkeypatch, vendor_ids: set[int]) -> None:
        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.fleet.devices import DeviceProbe

        monkeypatch.setattr(
            planning_mod, "probe_devices", lambda _b: DeviceProbe([], "", spoke_protocol=True)
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.cuda_runtime.assert_gpu_devices_usable", lambda *_a: None
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.installed_gpu_vendor_ids",
            lambda: frozenset(vendor_ids),
        )

    @pytest.mark.parametrize(
        ("vendor_id", "vendor"),
        [(0x10DE, "NVIDIA"), (0x1002, "AMD"), (0x8086, "Intel")],
        ids=["nvidia", "amd", "intel"],
    )
    def test_an_unenumerated_card_of_any_vendor_is_reported(
        self, monkeypatch, caplog, vendor_id: int, vendor: str
    ) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        self._probe_reporting_nothing(monkeypatch, {vendor_id})
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            planning_mod.resolve_devices(Path("/bin/llama-server"))
        assert vendor in caplog.text
        assert "reported none" in caplog.text

    def test_a_host_with_no_gpu_at_all_stays_quiet(self, monkeypatch, caplog) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        self._probe_reporting_nothing(monkeypatch, set())
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            planning_mod.resolve_devices(Path("/bin/llama-server"))
        assert "reported none" not in caplog.text


class TestABootTimeEmptyAnswerIsRetried:
    """A driver not ready when the daemon starts must not decide the whole run.

    The plan snapshot is taken once, on a clean box, and is not re-taken until a
    full teardown. An empty first answer therefore persists for the daemon's
    life, so a GPU host that booted a second too early served on CPU until
    someone restarted it.
    """

    @staticmethod
    def _answers(monkeypatch, results: list[list[object]]) -> list[int]:
        from lilbee.providers.fleet import planning as planning_mod

        calls: list[int] = []

        def _resolve(_binary):
            calls.append(1)
            return list(results[min(len(calls) - 1, len(results) - 1)]), False

        monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
        monkeypatch.setattr("lilbee.providers.fleet.gpu_env.apply_fleet_gpu_env", lambda: None)
        monkeypatch.setattr(
            "lilbee.providers.fleet.cuda_runtime.apply_cuda_runtime_env", lambda *_a: None
        )
        monkeypatch.setattr(planning_mod, "_resolve_devices_and_refusal", _resolve)
        monkeypatch.setattr(planning_mod, "_PROBE_RETRY_DELAY_S", 0.0)
        return calls

    def test_a_gpu_host_probes_again_before_accepting_nothing(self, monkeypatch) -> None:
        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.fleet.devices import FleetDevice

        card = FleetDevice("CUDA", 0, "gpu", 1, 1)
        calls = self._answers(monkeypatch, [[], [], [card]])
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.installed_gpu_vendor_ids",
            lambda: frozenset({0x10DE}),
        )
        devices, _refused = planning_mod._probe_engine_devices()
        assert devices == [card]
        assert len(calls) == 3

    def test_a_host_with_no_gpu_accepts_the_first_answer(self, monkeypatch) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        calls = self._answers(monkeypatch, [[]])
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.installed_gpu_vendor_ids", frozenset
        )
        devices, _refused = planning_mod._probe_engine_devices()
        assert devices == []
        assert len(calls) == 1

    def test_a_first_answer_with_devices_is_not_retried(self, monkeypatch) -> None:
        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.fleet.devices import FleetDevice

        calls = self._answers(monkeypatch, [[FleetDevice("CUDA", 0, "gpu", 1, 1)]])
        planning_mod._probe_engine_devices()
        assert len(calls) == 1

    def test_a_card_that_never_appears_gives_up_and_says_nothing_is_there(
        self, monkeypatch
    ) -> None:
        # A card the engine genuinely cannot use (no driver, wrong build) must not
        # retry forever; the fleet still has to start, on CPU, with the warning.
        from lilbee.providers.fleet import planning as planning_mod

        calls = self._answers(monkeypatch, [[]])
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.installed_gpu_vendor_ids",
            lambda: frozenset({0x1002}),
        )
        devices, _refused = planning_mod._probe_engine_devices()
        assert devices == []
        assert len(calls) == 1 + planning_mod._PROBE_RETRIES


class TestPartialOutputArrivesInEitherShape:
    """CPython hands back bytes from a timeout even when the pipe is in text mode."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (b"ggml: device hung\n", "ggml: device hung"),
            ("ggml: device hung\n", "ggml: device hung"),
            (b"\xff\xfe not utf-8", "�� not utf-8"),
            (None, "(nothing)"),
        ],
        ids=["bytes", "str", "undecodable", "absent"],
    )
    def test_the_probe_tail_reads_it_either_way(self, raw: object, expected: str) -> None:
        from lilbee.providers.fleet.devices import _decoded_output, _probe_tail

        assert _probe_tail(_decoded_output(raw)) == expected
