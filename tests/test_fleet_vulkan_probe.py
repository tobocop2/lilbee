"""Tests for isolating the native Vulkan enumeration in a child process."""

from __future__ import annotations

import logging

from lilbee.providers.fleet import gpu_select


class TestTheLoaderRunsOutOfProcess:
    """A faulting ICD must not be able to take the daemon with it.

    vkCreateInstance loads every vendor ICD on the host. A conflicting or broken
    one segfaults the caller, and no except clause in Python can catch that. The
    enumeration therefore runs in a child whose death is an answer, not a crash.
    """

    def test_a_child_that_dies_reports_no_opinion(self, monkeypatch, caplog) -> None:
        # A segfaulting ICD shows up as a negative return code, not an exception.
        monkeypatch.setattr(gpu_select, "_run_probe_child", lambda: ("", -11, "Segmentation fault"))
        with caplog.at_level(logging.DEBUG, logger="lilbee.providers.fleet.gpu_select"):
            assert gpu_select._enumerate_vulkan_devices() is None
        assert "Segmentation fault" in caplog.text

    def test_a_child_that_answers_nothing_is_a_host_with_no_adapters(self, monkeypatch) -> None:
        # Distinct from a crash: the loader ran and found none.
        monkeypatch.setattr(gpu_select, "_run_probe_child", lambda: ("[]", 0, ""))
        assert gpu_select._enumerate_vulkan_devices() == []

    def test_a_child_that_answers_is_parsed_back(self, monkeypatch) -> None:
        payload = (
            '[{"index": 0, "device_type": 2, "device_name": "RTX 4090", "vendor_id": 4318,'
            ' "vram_bytes": 100, "device_uuid": "abcd", "storage_buffer_16bit": true,'
            ' "free_bytes": 90}]'
        )
        monkeypatch.setattr(gpu_select, "_run_probe_child", lambda: (payload, 0, ""))
        devices = gpu_select._enumerate_vulkan_devices()
        assert devices is not None
        assert [(d.index, d.device_name, d.vram_bytes) for d in devices] == [(0, "RTX 4090", 100)]
        # The UUID is bytes on this side of the boundary and travels as hex.
        assert devices[0].device_uuid == bytes.fromhex("abcd")

    def test_unparseable_output_is_no_opinion(self, monkeypatch) -> None:
        monkeypatch.setattr(gpu_select, "_run_probe_child", lambda: ("not json", 0, ""))
        assert gpu_select._enumerate_vulkan_devices() is None


class TestTheChildEntryPoint:
    """The child is a plain module with a main(), like every other reinvocation."""

    def test_main_prints_the_enumeration_as_json(self, monkeypatch, capsys) -> None:
        import json

        from lilbee.providers.fleet import vulkan_probe

        monkeypatch.setattr(
            vulkan_probe,
            "enumerate_in_process",
            lambda: [
                gpu_select.VulkanDevice(
                    index=1,
                    device_type=1,
                    device_name="gpu",
                    vendor_id=4318,
                    vram_bytes=8,
                    device_uuid=b"\xab",
                    storage_buffer_16bit=True,
                    free_bytes=4,
                )
            ],
        )
        vulkan_probe.main()
        assert json.loads(capsys.readouterr().out) == [
            {
                "index": 1,
                "device_type": 1,
                "device_name": "gpu",
                "vendor_id": 4318,
                "vram_bytes": 8,
                "device_uuid": "ab",
                "storage_buffer_16bit": True,
                "free_bytes": 4,
            }
        ]

    def test_main_exits_non_zero_when_the_loader_has_no_opinion(self, monkeypatch) -> None:
        import pytest

        from lilbee.providers.fleet import vulkan_probe

        monkeypatch.setattr(vulkan_probe, "enumerate_in_process", lambda: None)
        with pytest.raises(SystemExit) as excinfo:
            vulkan_probe.main()
        assert excinfo.value.code != 0


class TestACrashingChildIsContained:
    """The whole point, exercised against a real process rather than a stub."""

    def test_a_segfaulting_child_leaves_the_parent_running(self) -> None:
        import signal
        import sys

        from lilbee.providers.fleet.proc import run_bounded

        # What a faulting ICD does inside vkCreateInstance. In-process this ends
        # the daemon; through run_bounded it is a negative return code.
        _stdout, returncode = run_bounded(
            [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)"],
            timeout_s=10.0,
            kill_wait_s=5.0,
            label="crash-probe",
        )
        assert returncode == -signal.SIGSEGV


class TestTheChildSpawnItself:
    """The spawn is bounded and its failures are answers, not exceptions."""

    def test_a_child_that_cannot_be_run_is_no_opinion(self, monkeypatch, caplog) -> None:
        from lilbee.providers.base import ProviderError

        def _cannot_run() -> tuple[str, int, str]:
            raise ProviderError("no interpreter")

        monkeypatch.setattr(gpu_select, "_run_probe_child", _cannot_run)
        with caplog.at_level(logging.DEBUG, logger="lilbee.providers.fleet.gpu_select"):
            assert gpu_select._enumerate_vulkan_devices() is None
        assert "could not be run" in caplog.text

    def test_the_spawn_is_bounded_and_names_itself(self, monkeypatch) -> None:
        # A wedged ICD can hang inside vkCreateInstance rather than fault, so the
        # read that asked must not hang with it.
        seen: dict[str, object] = {}

        def _fake_run_bounded(argv, **kwargs):
            seen["argv"] = argv
            seen.update(kwargs)
            return "[]", 0

        monkeypatch.setattr("lilbee.providers.fleet.proc.run_bounded", _fake_run_bounded)
        gpu_select._run_probe_child()
        assert seen["argv"][1:] == ["-m", "lilbee.providers.fleet.vulkan_probe"]
        assert seen["timeout_s"] > 0
        assert seen["label"] == "vulkan-probe"
