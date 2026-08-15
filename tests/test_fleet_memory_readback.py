"""The /memory readback path: probe, launch flag, and estimate check.

An engine carrying llama.cpp PR 26130 serves its per-device allocation on
``GET /memory`` when launched with ``--memory``. These tests cover the new
path end to end at the unit level: the ``--help`` capability probe, the flag
on the argv, the swap config leaving the trace-log env off, the response
parsing, and the swap manager routing the check to the endpoint. The log
path keeps its own tests untouched; an engine without the flag must behave
exactly as before.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from lilbee.providers.fleet import readback
from lilbee.providers.fleet import swap_manager as sm
from lilbee.providers.fleet.adapters import ROLE_SPECS, build_server_argv
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.readback import (
    MEMORY_FLAG,
    check_memory_report,
    parse_memory_rows,
    supports_memory_readback,
)
from lilbee.providers.fleet.swap_config import build_swap_config
from lilbee.providers.roles import WorkerRole

GIB = 1024**3

# GET /memory as PR 26130's server answers it for tinygemma3 on Metal: a GPU
# row with every buffer class including the vision projector, and a host row
# without total/free.
_METAL_VISION_PAYLOAD = {
    "n_layer": 8,
    "data": [
        {
            "name": "MTL0",
            "model": 40707072,
            "context": 510656512,
            "compute": 165136416,
            "mmproj": 2181120,
            "total": 22906503168,
            "free": 22187376640,
        },
        {"name": "Host", "model": 35660288, "context": 4195328, "compute": 152064032},
    ],
}
_MTL0_TOTAL = 40707072 + 510656512 + 165136416 + 2181120


class TestParseMemoryRows:
    def test_sums_every_buffer_class_per_device(self) -> None:
        rows = parse_memory_rows(_METAL_VISION_PAYLOAD)
        assert rows["MTL0"] == _MTL0_TOTAL
        assert rows["Host"] == 35660288 + 4195328 + 152064032

    def test_a_row_without_mmproj_sums_the_rest(self) -> None:
        payload = {"data": [{"name": "CUDA0", "model": 100, "context": 20, "compute": 3}]}
        assert parse_memory_rows(payload) == {"CUDA0": 123}

    def test_malformed_payloads_parse_empty(self) -> None:
        assert parse_memory_rows({}) == {}
        assert parse_memory_rows({"data": "nope"}) == {}
        assert parse_memory_rows([]) == {}
        assert parse_memory_rows({"data": [{"model": 5}, "junk", {"name": ""}]}) == {}

    def test_non_numeric_sizes_are_ignored(self) -> None:
        payload = {"data": [{"name": "CUDA0", "model": "big", "compute": 7}]}
        assert parse_memory_rows(payload) == {"CUDA0": 7}

    def test_host_and_cpu_rows_are_host_devices(self) -> None:
        # The endpoint names host memory "Host" (and "CPU_REPACK" on a
        # CPU-only build), unlike the log's "CUDA_Host" shape; both spellings
        # must be excluded from GPU totals.
        assert readback._is_host_device("Host")
        assert readback._is_host_device("CPU_REPACK")
        assert not readback._is_host_device("MTL0")


class TestCheckMemoryReport:
    def test_quiet_when_the_report_matches_the_estimate(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            warned = check_memory_report(
                WorkerRole.CHAT,
                "gemma",
                _MTL0_TOTAL,
                {"MTL0": _MTL0_TOTAL},
                _METAL_VISION_PAYLOAD,
            )
        assert not warned
        assert not caplog.records

    def test_mmproj_bytes_count_without_any_fudge(self, caplog) -> None:
        # The estimate charged the projector to MTL0; the report carries it in
        # the mmproj field. No est_unreported_bytes adjustment exists on this
        # path, and the check stays quiet because the row already balances.
        estimate = {"MTL0": _MTL0_TOTAL - 2181120 + 2181120}
        with caplog.at_level(logging.WARNING):
            warned = check_memory_report(
                WorkerRole.CHAT, "gemma", sum(estimate.values()), estimate, _METAL_VISION_PAYLOAD
            )
        assert not warned

    def test_warns_naming_the_card_that_diverged(self, caplog) -> None:
        payload = {
            "data": [
                {"name": "CUDA0", "model": 8 * GIB},
                {"name": "CUDA1", "model": 2 * GIB},
            ]
        }
        with caplog.at_level(logging.WARNING):
            warned = check_memory_report(
                WorkerRole.CHAT,
                "big-model",
                10 * GIB,
                {"CUDA0": 5 * GIB, "CUDA1": 5 * GIB},
                payload,
            )
        assert warned
        assert "CUDA0" in caplog.text

    def test_scalar_comparison_when_no_per_device_estimate(self, caplog) -> None:
        payload = {"data": [{"name": "CUDA0", "model": 8 * GIB}]}
        with caplog.at_level(logging.WARNING):
            warned = check_memory_report(WorkerRole.CHAT, "m", 4 * GIB, {}, payload)
        assert warned

    def test_an_empty_report_is_said_not_swallowed(self, caplog) -> None:
        # The engine took the flag but reported nothing lilbee can use; silence
        # here would read as "estimate fine" forever, like the log-format drift.
        with caplog.at_level(logging.WARNING):
            warned = check_memory_report(
                WorkerRole.CHAT, "m", 4 * GIB, {"CUDA0": 4 * GIB}, {"data": []}
            )
        assert warned
        assert "unverified" in caplog.text

    def test_host_rows_never_charge_a_card(self, caplog) -> None:
        payload = {
            "data": [
                {"name": "CUDA0", "model": 4 * GIB},
                {"name": "Host", "model": 40 * GIB},
            ]
        }
        with caplog.at_level(logging.WARNING):
            warned = check_memory_report(WorkerRole.CHAT, "m", 4 * GIB, {"CUDA0": 4 * GIB}, payload)
        assert not warned


class TestSupportsMemoryReadback:
    def setup_method(self) -> None:
        supports_memory_readback.cache_clear()

    def _binary_with_help(self, monkeypatch: pytest.MonkeyPatch, help_text: str) -> Path:
        class _Done:
            stdout = help_text
            stderr = ""

        monkeypatch.setattr(readback.subprocess, "run", lambda *a, **k: _Done())
        return Path("/fake/llama-server")

    def test_true_when_help_lists_the_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        binary = self._binary_with_help(monkeypatch, "--metrics\n--memory\n--slots")
        assert supports_memory_readback(binary)

    def test_false_when_help_lacks_the_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        binary = self._binary_with_help(monkeypatch, "--metrics\n--slots")
        assert not supports_memory_readback(binary)

    def test_a_longer_flag_is_not_mistaken_for_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        binary = self._binary_with_help(monkeypatch, "--memory-f32 use f32 for memory")
        assert not supports_memory_readback(binary)

    def test_a_binary_that_cannot_run_reads_as_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*a, **k):
            raise OSError("no such file")

        monkeypatch.setattr(readback.subprocess, "run", _boom)
        assert not supports_memory_readback(Path("/missing/llama-server"))

    def test_the_answer_is_cached_per_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        class _Done:
            stdout = "--memory"
            stderr = ""

        def _run(*a, **k):
            calls.append(a)
            return _Done()

        monkeypatch.setattr(readback.subprocess, "run", _run)
        binary = Path("/fake/llama-server")
        assert supports_memory_readback(binary)
        assert supports_memory_readback(binary)
        assert len(calls) == 1


class TestArgvAndSwapConfig:
    def _argv(self, *, memory_endpoint: bool) -> list[str]:
        return build_server_argv(
            binary=Path("/bin/llama-server"),
            spec=ROLE_SPECS[WorkerRole.CHAT],
            model_path=Path("/m/c.gguf"),
            devices=(0,),
            n_gpu_layers=99,
            slots=1,
            ctx_per_slot=4096,
            memory_endpoint=memory_endpoint,
        )

    def test_flag_rides_the_argv_when_the_engine_has_the_endpoint(self) -> None:
        assert MEMORY_FLAG in self._argv(memory_endpoint=True)

    def test_flag_is_absent_by_default(self) -> None:
        # A stock engine rejects unknown flags at launch; the flag must never
        # reach a binary that did not advertise it.
        assert MEMORY_FLAG not in self._argv(memory_endpoint=False)

    def _config(self, argv: list[str], tmp_path: Path) -> dict:
        launch = InstanceLaunch(
            role=WorkerRole.CHAT, argv=argv, env_overrides={}, model="chat-model"
        )
        rendered = build_swap_config([launch], {launch.model_id: 5900}, engine_log_dir=tmp_path)
        return json.loads(rendered)["models"][launch.model_id]

    def test_memory_mode_launch_gets_no_trace_log_env(self, tmp_path: Path) -> None:
        entry = self._config(["/bin/llama-server", MEMORY_FLAG], tmp_path)
        assert not any("LLAMA" in var for var in entry.get("env", []))

    def test_log_mode_launch_keeps_the_trace_log_env(self, tmp_path: Path) -> None:
        entry = self._config(["/bin/llama-server"], tmp_path)
        assert any(var.startswith(readback.ENV_LOG_FILE) for var in entry["env"])


class _MemoryResponse:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _manager_with_launch(
    tmp_path: Path, argv: list[str], port: int | None = 5901
) -> sm.SwapManager:
    manager = sm.SwapManager(tmp_path, SwapGroup.CHAT)
    launch = InstanceLaunch(
        role=WorkerRole.CHAT,
        argv=argv,
        env_overrides={},
        model="chat-model",
        est_vram_bytes=4 * GIB,
        est_vram_by_device={"CUDA0": 4 * GIB},
    )
    manager._launch_by_model = {launch.model_id: launch}
    if port is not None:
        manager._member_port_by_model = {launch.model_id: port}
    return manager


class TestSwapManagerRouting:
    def test_memory_launch_checks_the_endpoint_not_the_log(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        payload = {"data": [{"name": "CUDA0", "model": 4 * GIB}]}
        urls: list[str] = []

        def _responder(url):
            urls.append(url)
            return _MemoryResponse(payload)

        monkeypatch.setattr(sm, "_probe_client", lambda: _FakeClient(_responder))
        log_checks: list[str] = []
        monkeypatch.setattr(sm, "check_launch", lambda *a, **k: log_checks.append("log"))
        monkeypatch.setattr(sm, "report_missing_log", lambda *a, **k: log_checks.append("missing"))
        manager = _manager_with_launch(tmp_path, ["/bin/llama-server", MEMORY_FLAG])
        with caplog.at_level(logging.WARNING):
            manager._check_estimates({"chat-0"})
        assert urls == ["http://127.0.0.1:5901/memory"]
        assert log_checks == []
        assert not caplog.records  # report matches the estimate exactly

    def test_log_launch_keeps_the_existing_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_checks: list[str] = []
        monkeypatch.setattr(sm, "check_launch", lambda *a, **k: log_checks.append("log") or False)
        monkeypatch.setattr(sm, "report_missing_log", lambda *a, **k: False)
        monkeypatch.setattr(
            sm, "_probe_client", lambda: pytest.fail("log mode must not touch HTTP")
        )
        manager = _manager_with_launch(tmp_path, ["/bin/llama-server"])
        manager._check_estimates({"chat-0"})
        assert log_checks == ["log"]

    def test_an_unanswered_endpoint_is_said_not_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        import httpx

        def _refuse(url):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(sm, "_probe_client", lambda: _FakeClient(_refuse))
        manager = _manager_with_launch(tmp_path, ["/bin/llama-server", MEMORY_FLAG])
        with caplog.at_level(logging.WARNING):
            manager._check_estimates({"chat-0"})
        assert "unverified" in caplog.text

    def test_bound_manager_without_ports_skips_quietly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        monkeypatch.setattr(
            sm, "_probe_client", lambda: pytest.fail("no port known, nothing to fetch")
        )
        manager = _manager_with_launch(tmp_path, ["/bin/llama-server", MEMORY_FLAG], port=None)
        with caplog.at_level(logging.WARNING):
            manager._check_estimates({"chat-0"})
        assert not caplog.records


class _FakeClient:
    def __init__(self, responder) -> None:
        self._responder = responder

    def get(self, url, timeout=None):
        return self._responder(url)
