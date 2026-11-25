"""The device probe's failures must reach the user, not vanish."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import devices as devices_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines import llama as llama_mod
from lilbee.providers.fleet.engines.llama import LlamaCppAdapter
from lilbee.providers.roles import WorkerRole


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
        from lilbee.providers.fleet.devices import DeviceProbe

        monkeypatch.setattr(
            llama_mod, "probe_devices", lambda _b: DeviceProbe([], "", spoke_protocol=True)
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.cuda_runtime.assert_cuda_devices_usable", lambda *_a: None
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.rocm_runtime.assert_rocm_devices_usable", lambda *_a: None
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.installed_gpu_vendor_ids",
            lambda: frozenset(vendor_ids),
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.undriven_gpu_vendor_ids", frozenset
        )

    @pytest.mark.parametrize(
        ("vendor_id", "vendor", "hint"),
        [
            (0x10DE, "NVIDIA", "CUDA_VISIBLE_DEVICES"),
            (0x1002, "AMD", "ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES"),
            (0x8086, "Intel", "ONEAPI_DEVICE_SELECTOR"),
        ],
        ids=["nvidia", "amd", "intel"],
    )
    def test_an_unenumerated_card_of_any_vendor_is_reported(
        self, monkeypatch, caplog, vendor_id: int, vendor: str, hint: str
    ) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        self._probe_reporting_nothing(monkeypatch, {vendor_id})
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            planning_mod.resolve_devices(Path("/bin/llama-server"), engine=LLAMA)
        assert vendor in caplog.text
        assert "reported none" in caplog.text
        assert f"driver, {hint}, and" in caplog.text

    def test_a_card_with_no_kernel_driver_is_named_with_the_cause(
        self, monkeypatch, caplog
    ) -> None:
        # An NVIDIA-only laptop with the nvidia module not loaded: the card does not
        # count as a usable GPU, but the user still needs to hear why there is none.
        from lilbee.providers.fleet import planning as planning_mod

        self._probe_reporting_nothing(monkeypatch, set())
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.undriven_gpu_vendor_ids",
            lambda: frozenset({0x10DE}),
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            planning_mod.resolve_devices(Path("/bin/llama-server"), engine=LLAMA)
        assert "NVIDIA GPU" in caplog.text
        assert "no kernel driver" in caplog.text

    def test_a_card_with_an_unreadable_driver_link_is_not_called_driverless(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        from lilbee.providers.fleet import gpu_hardware
        from lilbee.providers.fleet import planning as planning_mod

        card = tmp_path / "0000-01-00.0"
        card.mkdir()
        (card / "class").write_text("0x030000\n", encoding="utf-8")
        (card / "vendor").write_text("0x1002\n", encoding="utf-8")
        denied = card / "driver"
        real_stat = Path.stat
        real_undriven = gpu_hardware.undriven_gpu_vendor_ids

        def stat(path: Path, **kwargs):
            if path == denied:
                raise PermissionError(13, "Permission denied", str(path))
            return real_stat(path, **kwargs)

        self._probe_reporting_nothing(monkeypatch, set())
        monkeypatch.setattr(Path, "stat", stat)
        monkeypatch.setattr(gpu_hardware.sys, "platform", "linux")
        monkeypatch.setattr(gpu_hardware, "_SYSFS_PCI_DEVICE_DIR", tmp_path)
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.undriven_gpu_vendor_ids", real_undriven
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            planning_mod.resolve_devices(Path("/bin/llama-server"), engine=LLAMA)
        assert "no kernel driver" not in caplog.text

    def test_a_host_with_no_gpu_at_all_stays_quiet(self, monkeypatch, caplog) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        self._probe_reporting_nothing(monkeypatch, set())
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            planning_mod.resolve_devices(Path("/bin/llama-server"), engine=LLAMA)
        assert "reported none" not in caplog.text


class _FakeEngine(LlamaCppAdapter):
    """The llama adapter with device answers from a script; the last answer repeats."""

    def __init__(self, answers: list[object]) -> None:
        self.answers = answers
        self.probes = 0

    def binary(self) -> Path:
        return Path("/bin/srv")

    def discover_devices(self, _binary: Path):
        answer = self.answers[min(self.probes, len(self.answers) - 1)]
        self.probes += 1
        return answer


class TestAnEmptyProbeIsRetriedByOneLadder:
    """A driver not ready when the daemon starts must not decide the whole run.

    The plan snapshot is taken once, on a clean box, and is not re-taken until a
    full teardown. An empty first answer therefore persists for the daemon's
    life, so a GPU host that booted a second too early served on CPU until
    someone restarted it. One retry ladder, in the device read, owns that wait.
    """

    @pytest.fixture(autouse=True)
    def _host(self, monkeypatch) -> Iterator[list[float]]:
        """A one-card host that never sleeps; the yielded list records each delay."""
        from lilbee.providers.fleet import planning as planning_mod

        delays: list[float] = []
        monkeypatch.setattr(planning_mod.time, "sleep", delays.append)
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.installed_gpu_vendor_ids",
            lambda: frozenset({0x10DE}),
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.undriven_gpu_vendor_ids", frozenset
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.cuda_runtime.assert_cuda_devices_usable", lambda *_a: None
        )
        monkeypatch.setattr(
            "lilbee.providers.fleet.rocm_runtime.assert_rocm_devices_usable", lambda *_a: None
        )
        monkeypatch.setattr(planning_mod, "engine_binary_identity", lambda _b: "v1")
        planning_mod.clear_read_device_cache()
        planning_mod.clear_plan_probe()
        yield delays
        planning_mod.clear_read_device_cache()
        planning_mod.clear_plan_probe()

    @staticmethod
    def _empty():
        from lilbee.providers.fleet.devices import DeviceProbe

        return DeviceProbe([], "", spoke_protocol=True)

    @staticmethod
    def _card():
        from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice

        return DeviceProbe([FleetDevice("CUDA", 0, "gpu", 1, 1)], "", spoke_protocol=True)

    @staticmethod
    def _build(engine):
        from lilbee.providers.fleet import planning as planning_mod

        return planning_mod._probe_engine_devices(engine)

    @staticmethod
    def _view(engine):
        from lilbee.providers.fleet import planning as planning_mod

        return planning_mod._read_device_cache.get(engine.binary(), engine)

    @pytest.mark.parametrize("read", ["_build", "_view"], ids=["build", "read-view"])
    def test_an_always_empty_probe_takes_one_ladder(self, _host, read) -> None:
        engine = _FakeEngine([self._empty()])
        assert getattr(self, read)(engine).devices == []
        assert engine.probes == 5
        assert sum(_host) == pytest.approx(2.0)

    def test_the_build_and_the_read_view_take_the_same_ladder(self, _host) -> None:
        build_engine = _FakeEngine([self._empty()])
        self._build(build_engine)
        build_slept = sum(_host)
        _host.clear()
        view_engine = _FakeEngine([self._empty()])
        self._view(view_engine)
        assert build_engine.probes == view_engine.probes
        assert build_slept == pytest.approx(sum(_host))

    def test_a_stale_plan_read_probes_one_ladder_and_the_next_probes_nothing(
        self, monkeypatch
    ) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1)
        identity = ["v1"]
        monkeypatch.setattr(planning_mod, "_engine_identity", lambda _e: identity[0])
        engine = _FakeEngine([self._card()])
        planning_mod.capture_plan_probe({WorkerRole.CHAT: engine})
        captured = engine.probes
        identity[0] = "v2"
        engine.answers = [self._empty()]
        planning_mod._current_plan_probe(engine)
        assert engine.probes - captured == 5
        planning_mod._current_plan_probe(engine)
        assert engine.probes - captured == 5

    def test_a_cold_build_with_an_always_empty_probe_takes_one_ladder(
        self, monkeypatch, _host
    ) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1)
        engine = _FakeEngine([self._empty()])
        planning_mod.assert_engine_probeable(engine)
        planning_mod.capture_plan_probe({WorkerRole.CHAT: engine})
        assert engine.probes == 5
        assert sum(_host) == pytest.approx(2.0)

    def test_a_binary_change_between_the_check_and_the_capture_probes_again(
        self, monkeypatch
    ) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1)
        identity = ["v1"]
        monkeypatch.setattr(planning_mod, "engine_binary_identity", lambda _b: identity[0])
        engine = _FakeEngine([self._empty()])
        planning_mod.assert_engine_probeable(engine)
        identity[0] = "v2"
        planning_mod.capture_plan_probe({WorkerRole.CHAT: engine})
        assert engine.probes == 10

    @pytest.mark.parametrize(
        ("incumbent", "reaped", "probes"),
        [(True, False, 10), (False, True, 10), (False, False, 5)],
        ids=["stopped", "reaped", "none"],
    )
    def test_the_capture_probes_again_only_after_an_engine_was_stopped_or_reaped(
        self, monkeypatch, tmp_path, incumbent: bool, reaped: bool, probes: int
    ) -> None:
        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.fleet import provider as prov_mod

        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1)
        monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: reaped)
        monkeypatch.setattr(prov_mod, "engine_record_exists", lambda _d: incumbent)
        monkeypatch.setattr(prov_mod, "stop_engine", lambda _d: None)
        engine = _FakeEngine([self._empty()])
        planning_mod.assert_engine_probeable(engine)
        prov_mod._clear_engine_dir(tmp_path)
        planning_mod.capture_plan_probe({WorkerRole.CHAT: engine})
        assert engine.probes == probes

    def test_a_same_path_reinstall_between_the_check_and_the_capture_probes_again(
        self, monkeypatch, tmp_path
    ) -> None:
        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.fleet.binary import engine_binary_identity

        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1)
        monkeypatch.setattr(planning_mod, "engine_binary_identity", engine_binary_identity)
        binary = tmp_path / "srv"
        binary.write_bytes(b"v1")

        class _Installed(_FakeEngine):
            def binary(self) -> Path:
                return binary

        engine = _Installed([self._empty()])
        planning_mod.assert_engine_probeable(engine)
        binary.write_bytes(b"v2")
        planning_mod.capture_plan_probe({WorkerRole.CHAT: engine})
        assert engine.probes == 10

    def test_an_empty_check_older_than_the_window_is_probed_again(self, monkeypatch) -> None:
        import time
        from types import SimpleNamespace

        from lilbee.providers.fleet import planning as planning_mod

        now = [1000.0]
        monkeypatch.setattr(
            planning_mod, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=time.sleep)
        )
        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1)
        engine = _FakeEngine([self._empty()])
        planning_mod.assert_engine_probeable(engine)
        now[0] += planning_mod._DEVICE_PROBE_TTL_S + 1
        planning_mod.capture_plan_probe({WorkerRole.CHAT: engine})
        assert engine.probes == 10

    def test_a_check_that_found_a_card_is_followed_by_a_fresh_capture(self, monkeypatch) -> None:
        from lilbee.providers.fleet import planning as planning_mod

        monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: 1)
        engine = _FakeEngine([self._card()])
        planning_mod.assert_engine_probeable(engine)
        planning_mod.capture_plan_probe({WorkerRole.CHAT: engine})
        assert engine.probes == 2

    def test_the_missing_gpu_warning_is_logged_once_per_probe(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            self._build(_FakeEngine([self._empty()]))
        assert caplog.text.count("reported none") == 1

    def test_a_binary_that_does_not_speak_the_protocol_is_probed_once(self, monkeypatch) -> None:
        from lilbee.providers.fleet.devices import DeviceProbe

        monkeypatch.setattr("lilbee.providers.fleet.gpu_select.enumerate_gpu_vram", lambda: [])
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_select.integrated_vulkan_indices",
            MagicMock(return_value=set()),
        )
        engine = _FakeEngine([DeviceProbe([], "usage: llama-server", spoke_protocol=False)])
        assert self._build(engine).devices == []
        assert engine.probes == 1

    @pytest.mark.parametrize("read", ["_build", "_view"], ids=["build", "read-view"])
    def test_a_probe_that_recovers_returns_the_card(self, read) -> None:
        engine = _FakeEngine([self._empty(), self._empty(), self._card()])
        assert [d.index for d in getattr(self, read)(engine).devices] == [0]
        assert engine.probes == 3

    def test_a_host_with_no_gpu_accepts_the_first_answer(self, monkeypatch, _host) -> None:
        monkeypatch.setattr(
            "lilbee.providers.fleet.gpu_hardware.installed_gpu_vendor_ids", frozenset
        )
        engine = _FakeEngine([self._empty()])
        assert self._build(engine).devices == []
        assert engine.probes == 1
        assert _host == []

    def test_a_first_answer_with_devices_is_not_retried(self, _host) -> None:
        engine = _FakeEngine([self._card()])
        self._build(engine)
        assert engine.probes == 1
        assert _host == []


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


class TestAMissingRuntimeDllIsNamed:
    """Status 0xC0000135 means a DLL was not found, not that the flag is unknown."""

    def test_the_silent_probe_names_the_runtime(self, monkeypatch) -> None:
        monkeypatch.setattr(devices_mod, "_run_list_devices", lambda _b, _t: ("", 0xC0000135))
        with pytest.raises(ProviderError, match="Visual C"):
            devices_mod.probe_devices(Path("/bin/llama-server"))

    def test_a_probe_that_printed_its_header_keeps_the_crash_warning(
        self, monkeypatch, caplog
    ) -> None:
        output = "Available devices:\n"
        monkeypatch.setattr(devices_mod, "_run_list_devices", lambda _b, _t: (output, 0xC0000135))
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.devices"):
            devices_mod.probe_devices(Path("/bin/llama-server"))
        assert "then crashed" in caplog.text

    def test_another_status_keeps_the_flag_warning(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(devices_mod, "_run_list_devices", lambda _b, _t: ("", 3))
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.devices"):
            devices_mod.probe_devices(Path("/bin/llama-server"))
        assert "does not appear to support the flag" in caplog.text

    def test_the_planner_read_of_devices_lets_it_through(self, monkeypatch) -> None:
        from lilbee.providers.fleet import planning

        monkeypatch.setattr(devices_mod, "_run_list_devices", lambda _b, _t: ("", 0xC0000135))
        with pytest.raises(ProviderError, match="Visual C"):
            planning.resolve_devices(Path("/bin/llama-server"), LLAMA)


def test_the_probe_and_the_death_message_share_one_install_link(monkeypatch) -> None:
    from lilbee.providers.fleet.client import _upstream_death_message
    from lilbee.runtime.vc_runtime import VC_REDIST_URL

    monkeypatch.setattr(devices_mod, "_run_list_devices", lambda _b, _t: ("", 0xC0000135))
    with pytest.raises(ProviderError) as probe:
        devices_mod.probe_devices(Path("/bin/llama-server"))
    death = _upstream_death_message(
        ProviderErrorKind.MISSING_LIBRARY, 500, "m", None, "exit status 0xc0000135"
    )
    assert VC_REDIST_URL in str(probe.value)
    assert VC_REDIST_URL in death
