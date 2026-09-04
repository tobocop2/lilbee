"""Tests for the llama-swap process manager."""

from __future__ import annotations

import itertools
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from lilbee.providers.base import ProviderError
from lilbee.providers.fleet import swap_manager as sm
from lilbee.providers.fleet.groups import SwapGroup
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_manager import SwapManager
from lilbee.providers.roles import WorkerRole

# All lifecycle tests run one manager for the chat group unless stated otherwise.
_GROUP = SwapGroup.CHAT


class _FakeProc:
    """A stand-in subprocess that records teardown and reports a poll result."""

    def __init__(self, *, poll_result: int | None = None) -> None:
        self.pid = 4321
        self._poll_result = poll_result
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._poll_result

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _fake_response(*, status: int = 200, payload: object = None) -> object:
    # A live llama-swap answers /running with a {"running": [...]} body; default to
    # the empty-but-valid shape so "engine answers" fakes pass the identity check
    # (state_is_healthy/_proxy_answers now validate the payload, not just the status).
    if payload is None:
        payload = {"running": []}

    class _Resp:
        status_code = status

        def json(self) -> object:
            return payload

    return _Resp()


def _patch_spawn(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc) -> None:
    monkeypatch.setattr(sm, "resolve_llama_swap", lambda: Path("/fake/llama-swap"))
    monkeypatch.setattr(sm, "spawn_bound_child", lambda *a, **k: proc)
    # Isolate lifecycle tests from the real process-tree teardown.
    monkeypatch.setattr(sm, "_stop_own_fleet", lambda cfg, ports: None)


class _FakeProbeClient:
    """Stands in for the shared probe client; probes call .get on it."""

    def __init__(self, responder) -> None:
        self._responder = responder

    def get(self, url, timeout=None):
        return self._responder(url)


def _patch_http(monkeypatch: pytest.MonkeyPatch, responder) -> None:
    # The probes share one lru_cached httpx.Client (avoids rebuilding an SSL
    # context per poll), so patch the factory rather than httpx.get.
    monkeypatch.setattr(sm, "_probe_client", lambda: _FakeProbeClient(responder))


def _raise_connect_error(url):
    raise httpx.ConnectError("refused", request=None)


def _launch(role: WorkerRole) -> InstanceLaunch:
    return InstanceLaunch(
        role=role,
        argv=["/bin/llama-server"],
        env_overrides={},
        model=f"{role.value}-model",
    )


class TestStart:
    def test_writes_config_and_becomes_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT), _launch(WorkerRole.EMBED)])
        # Config was written and the endpoint is live.
        config = json.loads(mgr._config_path.read_text())
        assert mgr.endpoint().startswith("http://127.0.0.1:")
        # Each member got its own freshly allocated port, distinct from the proxy's.
        member_ports = {
            model_id: int(entry["cmd"].rsplit(" ", 1)[-1])
            for model_id, entry in config["models"].items()
        }
        proxy_port = int(mgr.endpoint().rsplit(":", 1)[-1])
        assert len(set(member_ports.values())) == 2
        assert proxy_port not in member_ports.values()

    def test_logs_the_engine_each_role_was_launched_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        """A fresh launch names its binary, build, backend and cards."""
        from lilbee.providers.fleet import planning as planning_mod
        from lilbee.providers.fleet.devices import FleetDevice

        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        monkeypatch.setattr(planning_mod, "engine_build_id", lambda: "wheel:0.6.91")
        monkeypatch.setattr(
            planning_mod,
            "probed_devices",
            lambda: (FleetDevice("Vulkan", 0, "NVIDIA GeForce RTX 3090", 1, 1),),
        )

        with caplog.at_level("INFO", logger="lilbee.providers.fleet.planning"):
            SwapManager(tmp_path, _GROUP).start(
                [_launch(WorkerRole.CHAT), _launch(WorkerRole.EMBED)]
            )

        launched = [r.message for r in caplog.records if r.message.startswith("Launched")]
        assert launched == [
            "Launched chat-0 serving chat-model on /bin/llama-server "
            "(build wheel:0.6.91, backend Vulkan, devices: Vulkan0: NVIDIA GeForce RTX 3090)",
            "Launched embed-0 serving embed-model on /bin/llama-server "
            "(build wheel:0.6.91, backend Vulkan, devices: Vulkan0: NVIDIA GeForce RTX 3090)",
        ]

    def test_redirects_llama_swap_stdio_to_a_log_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """llama-swap's stdout/stderr must go to a log file, never an inherited
        terminal: an inherited fd bleeds its HTTP access log onto a TUI/CLI
        parent's screen and corrupts the render."""
        captured: dict[str, object] = {}

        def _capturing_popen(*_args: object, **kwargs: object) -> _FakeProc:
            captured.update(kwargs)
            return _FakeProc(poll_result=None)

        monkeypatch.setattr(sm, "resolve_llama_swap", lambda: Path("/fake/llama-swap"))
        monkeypatch.setattr(sm, "spawn_bound_child", _capturing_popen)
        monkeypatch.setattr(sm, "_stop_own_fleet", lambda cfg, ports: None)
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))

        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT)])

        log_path = tmp_path / "logs" / "llama-swap-chat.log"
        assert log_path.exists()
        # stdout is the opened log file (its .name is the path), not None
        # (inherited terminal) nor a PIPE; stderr merges into the same file.
        assert getattr(captured["stdout"], "name", None) == str(log_path)
        assert captured["stdout"] is not subprocess.PIPE
        assert captured["stderr"] is subprocess.STDOUT
        # shutdown releases the captured handle.
        mgr.shutdown()
        assert mgr._log_file is None

    def test_bind_lifetime_is_forwarded_to_the_spawn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A keep-warm fleet spawns with the death binding off so it outlives lilbee."""
        captured: dict[str, object] = {}

        def _capturing_popen(*_args: object, **kwargs: object) -> _FakeProc:
            captured.update(kwargs)
            return _FakeProc(poll_result=None)

        monkeypatch.setattr(sm, "resolve_llama_swap", lambda: Path("/fake/llama-swap"))
        monkeypatch.setattr(sm, "spawn_bound_child", _capturing_popen)
        monkeypatch.setattr(sm, "_stop_own_fleet", lambda cfg, ports: None)
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))

        SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)], bind_lifetime=False)

        assert captured["bind_lifetime"] is False

    def test_raises_when_process_exits_before_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=1))  # already exited
        _patch_http(monkeypatch, lambda _url: _fake_response(status=503))
        with pytest.raises(ProviderError, match="exited before it was ready"):
            SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])

    def test_raises_when_never_healthy_in_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))

        def _refuse(_url: str) -> object:
            raise httpx.ConnectError("refused")

        _patch_http(monkeypatch, _refuse)
        monkeypatch.setattr(sm.time, "sleep", lambda _s: None)
        # sm.time is the global time module, so stray background threads (e.g.
        # Textual timers from earlier TUI tests) also call this fake; an endless
        # rising clock can be neither exhausted nor stalled by extra callers.
        clock = itertools.count(0.0, 10.4)
        monkeypatch.setattr(sm.time, "monotonic", lambda: next(clock))
        with pytest.raises(ProviderError, match="did not start in time"):
            SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])


class TestBootFailureLog:
    """A boot failure's error carries the tail of the captured llama-swap output."""

    def _spawn_writing(
        self, monkeypatch: pytest.MonkeyPatch, proc: _FakeProc, output: bytes
    ) -> None:
        """Patch the spawn so the fake child writes *output* to its stdout handle,
        as llama-swap would before dying."""

        def _spawn(*_args: object, **kwargs: Any) -> _FakeProc:
            kwargs["stdout"].write(output)
            kwargs["stdout"].flush()
            return proc

        monkeypatch.setattr(sm, "resolve_llama_swap", lambda: Path("/fake/llama-swap"))
        monkeypatch.setattr(sm, "spawn_bound_child", _spawn)
        monkeypatch.setattr(sm, "_stop_own_fleet", lambda cfg, ports: None)

    def test_exit_before_ready_carries_the_boot_log_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._spawn_writing(
            monkeypatch, _FakeProc(poll_result=1), b"bind: address already in use\n"
        )
        _patch_http(monkeypatch, lambda _url: _fake_response(status=503))
        with pytest.raises(ProviderError) as excinfo:
            SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        message = str(excinfo.value)
        assert "exited before it was ready" in message
        assert "bind: address already in use" in message
        assert str(tmp_path / "logs" / "llama-swap-chat.log") in message

    def test_tail_covers_only_the_current_boot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_path = tmp_path / "logs" / "llama-swap-chat.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_bytes(b"old boot noise\n")
        self._spawn_writing(monkeypatch, _FakeProc(poll_result=1), b"fresh boot output\n")
        _patch_http(monkeypatch, lambda _url: _fake_response(status=503))
        with pytest.raises(ProviderError) as excinfo:
            SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        assert "fresh boot output" in str(excinfo.value)
        assert "old boot noise" not in str(excinfo.value)

    def test_tail_is_capped_to_the_most_recent_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = b"EARLIEST-LINE\n" + b"x" * 3000 + b"\nFINAL-LINE\n"
        self._spawn_writing(monkeypatch, _FakeProc(poll_result=1), output)
        _patch_http(monkeypatch, lambda _url: _fake_response(status=503))
        with pytest.raises(ProviderError) as excinfo:
            SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        assert "FINAL-LINE" in str(excinfo.value)
        assert "EARLIEST-LINE" not in str(excinfo.value)

    def test_empty_log_keeps_the_plain_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=1))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=503))
        with pytest.raises(ProviderError) as excinfo:
            SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        assert str(excinfo.value) == "The local model engine exited before it was ready."

    def test_timeout_failure_also_carries_the_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._spawn_writing(monkeypatch, _FakeProc(poll_result=None), b"still loading the model\n")

        def _refuse(_url: str) -> object:
            raise httpx.ConnectError("refused")

        _patch_http(monkeypatch, _refuse)
        monkeypatch.setattr(sm.time, "sleep", lambda _s: None)
        clock = itertools.count(0.0, 10.4)
        monkeypatch.setattr(sm.time, "monotonic", lambda: next(clock))
        with pytest.raises(ProviderError) as excinfo:
            SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        message = str(excinfo.value)
        assert "did not start in time" in message
        assert "still loading the model" in message

    def test_unreadable_log_reads_as_empty(self, tmp_path: Path) -> None:
        # No start() ran, so the log file does not exist.
        assert SwapManager(tmp_path, _GROUP)._boot_log_tail() == ""


class TestEndpoint:
    def test_raises_before_start(self, tmp_path: Path) -> None:
        with pytest.raises(ProviderError, match="not running"):
            SwapManager(tmp_path, _GROUP).endpoint()


class TestRoleReady:
    def _started(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, responder) -> SwapManager:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda url: _fake_response(status=200))
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT)])
        _patch_http(monkeypatch, responder)
        return mgr

    def test_true_when_role_loaded_and_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        running = {"running": [{"model": "chat-0", "state": "ready"}]}
        mgr = self._started(tmp_path, monkeypatch, lambda _url: _fake_response(payload=running))
        assert mgr.role_ready(WorkerRole.CHAT) is True
        assert mgr.role_ready(WorkerRole.EMBED) is False

    def test_true_when_any_replica_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A role is ready as soon as one of its data-parallel replicas is up.
        running = {"running": [{"model": "embed-1", "state": "ready"}]}
        mgr = self._started(tmp_path, monkeypatch, lambda _url: _fake_response(payload=running))
        assert mgr.role_ready(WorkerRole.EMBED) is True

    def test_false_when_role_still_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        running = {"running": [{"model": "chat-0", "state": "starting"}]}
        mgr = self._started(tmp_path, monkeypatch, lambda _url: _fake_response(payload=running))
        assert mgr.role_ready(WorkerRole.CHAT) is False

    def test_false_when_running_probe_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _refuse(_url: str) -> object:
            raise httpx.ConnectError("refused")

        mgr = self._started(tmp_path, monkeypatch, _refuse)
        assert mgr.role_ready(WorkerRole.CHAT) is False

    def test_false_when_shutdown_clears_port_mid_probe(self, tmp_path: Path) -> None:
        # A concurrent shutdown clears _port, so endpoint() raises
        # ProviderError; the read-only probe must report False, not throw.
        mgr = SwapManager(tmp_path, _GROUP)  # never started -> _port is None
        assert mgr.role_ready(WorkerRole.CHAT) is False


class TestLifecycle:
    def test_shutdown_is_noop_when_not_started(self, tmp_path: Path) -> None:
        SwapManager(tmp_path, _GROUP).shutdown()  # must not raise

    def test_shutdown_reaps_owned_fleet_even_without_a_tracked_proc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """bb-dpp: a duplicate llama-swap this lilbee leaked is never recorded in
        ``_proc``, so shutdown must reap by config-path identity regardless.
        Teardown runs even when no swap is tracked."""
        reaped: list[Path] = []
        monkeypatch.setattr(sm, "_stop_own_fleet", lambda cfg, ports: reaped.append(cfg))
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.shutdown()  # never started -> _proc is None
        assert reaped == [mgr._config_path]  # the config-path reaper still ran

    def test_shutdown_terminates_and_clears(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        terminated: list[object] = []
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        monkeypatch.setattr(sm, "_stop_own_fleet", lambda cfg, ports: terminated.append(cfg))
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT)])
        mgr.shutdown()
        assert terminated  # the owned fleet was torn down
        with pytest.raises(ProviderError):
            mgr.endpoint()  # port cleared after shutdown

    def test_shutdown_releases_the_stopped_engines_death_pipe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stopping the engine must free its death pipe, or reloads accumulate one
        parked watcher and one fd per model switch on the pipe-bound platforms."""
        proc = _FakeProc(poll_result=None)
        _patch_spawn(monkeypatch, proc)
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        monkeypatch.setattr(sm, "_stop_own_fleet", lambda cfg, ports: None)
        released: list[int] = []
        monkeypatch.setattr(sm, "release_death_pipe", released.append)
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT)])
        mgr.shutdown()
        assert released == [proc.pid]


class _FakeChild:
    """A stand-in psutil.Process that records the signals it receives."""

    def __init__(self, *, running: bool = True) -> None:
        self.pid = 999
        self.running = running
        self.terminated = False
        self.killed = False

    def is_running(self) -> bool:
        return self.running

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class TestProcessTeardown:
    def test_pick_free_ports_returns_distinct_bound_ports(self) -> None:
        ports = sm._pick_free_ports(3)
        assert len(set(ports)) == 3
        assert all(1024 < port <= 65535 for port in ports)

    def test_terminate_proc_group_posix_sigterm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sm.os, "getpgid", lambda _pid: 99, raising=False)
        signals: list[int] = []
        monkeypatch.setattr(
            sm.os, "killpg", lambda _pgid, signum: signals.append(signum), raising=False
        )
        sm._terminate_proc_group(_FakeProc(poll_result=None))
        assert signals == [sm.signal.SIGTERM]

    def test_terminate_proc_group_escalates_to_sigkill_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sm.os, "getpgid", lambda _pid: 99, raising=False)
        signals: list[int] = []
        monkeypatch.setattr(
            sm.os, "killpg", lambda _pgid, signum: signals.append(signum), raising=False
        )
        monkeypatch.setattr(sm, "_await_killed", lambda procs: None)

        class _Stuck(_FakeProc):
            def wait(self, timeout: float | None = None) -> int:
                raise sm.psutil.TimeoutExpired(timeout or 0)

        sm._terminate_proc_group(_Stuck(poll_result=None))
        assert signals == [sm.signal.SIGTERM, sm._SIGKILL]

    _CFG = Path("/data/llama-swap.json")

    def test_stop_own_fleet_windows_hard_stops_each_swap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = _FakeProc(poll_result=None)
        monkeypatch.setattr(sm.sys, "platform", "win32")
        monkeypatch.setattr(sm, "_swaps_for_config", lambda _cfg: [proc])
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        monkeypatch.setattr(sm, "_find_orphan_servers", lambda ports: [])
        monkeypatch.setattr(sm, "_reap_survivors", lambda procs: None)
        sm._stop_own_fleet(self._CFG, ())
        assert proc.terminated is True

    def test_stop_own_fleet_posix_terminates_each_owned_swap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc = _FakeProc(poll_result=None)
        monkeypatch.setattr(sm.sys, "platform", "linux")
        monkeypatch.setattr(sm, "_swaps_for_config", lambda _cfg: [proc])
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        monkeypatch.setattr(sm, "_find_orphan_servers", lambda ports: [])
        groups: list[object] = []
        monkeypatch.setattr(sm, "_terminate_proc_group", lambda p: groups.append(p))
        monkeypatch.setattr(sm, "_reap_survivors", lambda procs: None)
        sm._stop_own_fleet(self._CFG, ())
        assert groups == [proc]

    def test_stop_own_fleet_reaps_children_and_port_servers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # llama-swap puts each server in its own process group, so terminating
        # the swap misses them; both a captured descendant and a member-port
        # server the snapshot missed (a respawned upstream) must be swept.
        child = _FakeChild(running=True)
        port_server = _FakeChild(running=True)
        reaped: list[object] = []
        monkeypatch.setattr(sm.sys, "platform", "linux")
        monkeypatch.setattr(sm, "_swaps_for_config", lambda _cfg: [_FakeProc(poll_result=None)])
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [child])
        monkeypatch.setattr(sm, "_find_orphan_servers", lambda ports: [port_server])
        monkeypatch.setattr(sm, "_terminate_proc_group", lambda p: None)
        monkeypatch.setattr(sm, "_reap_survivors", lambda procs: reaped.extend(procs))
        sm._stop_own_fleet(self._CFG, (1234,))
        assert child in reaped
        assert port_server in reaped

    def test_swaps_for_config_matches_only_our_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Identity is the -config path argument: a llama-swap on a different
        # config (another data_dir) and a non-swap process are both excluded.
        class _Proc:
            def __init__(self, pid: int, argv: list[str]) -> None:
                self.pid = pid
                self._argv = argv
                self.info = {"name": Path(next(iter(argv), "")).name}

            def cmdline(self) -> list[str]:
                return self._argv

        # Build the cmdline -config args the way the swap is launched (str of the
        # Path), so the match holds on Windows too, where str(Path) uses backslashes.
        swap_bin = "/x/bin/llama-swap"
        our_cfg = str(Path("/data/llama-swap.json"))
        other_cfg = str(Path("/other/llama-swap.json"))
        ours = _Proc(10, [swap_bin, "-config", our_cfg, "-listen", "x"])
        other = _Proc(11, [swap_bin, "-config", other_cfg])
        notswap = _Proc(12, ["/x/bin/python", "-config", our_cfg])
        monkeypatch.setattr(sm.psutil, "process_iter", lambda *a, **k: [ours, other, notswap])
        result = sm._swaps_for_config(Path("/data/llama-swap.json"))
        assert [p.pid for p in result] == [10]

    def test_swaps_for_config_reads_no_cmdline_for_unrelated_processes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The scan walks the whole process table; cmdline() is the expensive call
        # (and on macOS blocks on entitlement-protected binaries), so a process
        # whose name is not llama-swap must be filtered on name alone.
        class _Exploding:
            pid = 99
            info: ClassVar[dict[str, str]] = {"name": "Google Chrome Helper"}

            def cmdline(self) -> list[str]:
                raise AssertionError("cmdline() paid for an unrelated process")

        our_cfg = str(Path("/data/llama-swap.json"))

        class _Ours:
            pid = 10
            info: ClassVar[dict[str, str]] = {"name": "llama-swap"}

            def cmdline(self) -> list[str]:
                return ["/x/bin/llama-swap", "-config", our_cfg]

        monkeypatch.setattr(sm.psutil, "process_iter", lambda *a, **k: [_Exploding(), _Ours()])
        result = sm._swaps_for_config(Path("/data/llama-swap.json"))
        assert [p.pid for p in result] == [10]

    def test_swaps_for_config_skips_processes_that_vanish_mid_scan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A process can exit (or deny access) between enumeration and cmdline();
        # the scan must skip it, not abort, or a busy box could never be reaped.
        swap_bin = "/x/bin/llama-swap"
        our_cfg = str(Path("/data/llama-swap.json"))

        class _GoneProc:
            pid = 20
            info: ClassVar[dict[str, str]] = {"name": "llama-swap"}

            def cmdline(self) -> list[str]:
                raise sm.psutil.NoSuchProcess(self.pid)

        class _DeniedProc:
            pid = 21
            info: ClassVar[dict[str, str]] = {"name": "llama-swap"}

            def cmdline(self) -> list[str]:
                raise sm.psutil.AccessDenied(self.pid)

        class _LiveProc:
            pid = 22
            info: ClassVar[dict[str, str]] = {"name": "llama-swap"}

            def cmdline(self) -> list[str]:
                return [swap_bin, "-config", our_cfg]

        monkeypatch.setattr(
            sm.psutil, "process_iter", lambda *a, **k: [_GoneProc(), _DeniedProc(), _LiveProc()]
        )
        result = sm._swaps_for_config(Path("/data/llama-swap.json"))
        assert [p.pid for p in result] == [22]


class _FakePsProcess:
    """A stand-in psutil.Process with a settable cmdline and recorded signals."""

    def __init__(
        self,
        pid: int,
        *,
        cmdline: list[str],
        status: str = "running",
        create_time: float = 0.0,
    ) -> None:
        self.pid = pid
        self._cmdline = cmdline
        self._status = status
        self._create_time = create_time
        self.signals: list[int] = []
        self.wait_raises = False

    def cmdline(self) -> list[str]:
        return self._cmdline

    def status(self) -> str:
        return self._status

    def create_time(self) -> float:
        return self._create_time

    def children(self, recursive: bool = False) -> list:
        return []

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def wait(self, timeout: float | None = None) -> int:
        if self.wait_raises:
            raise sm.psutil.TimeoutExpired(timeout or 0, self.pid)
        return 0


def _patch_psutil_process(monkeypatch: pytest.MonkeyPatch, table: dict[int, object]) -> None:
    def _process(pid: int | None = None) -> object:
        if pid is None:  # no-arg form = the current process (used by _write_state)
            return _FakePsProcess(os.getpid(), cmdline=[], create_time=123.0)
        if pid not in table:
            raise sm.psutil.NoSuchProcess(pid)
        return table[pid]

    monkeypatch.setattr(sm.psutil, "Process", _process)


def _write_state(
    tmp_path: Path,
    *,
    pid: int = 7777,
    pgid: int | None = 7777,
    created_at: float | None = None,
    member_ports: list[int] | None = None,
    filename: str = "llama-swap.state.json",
) -> Path:
    state_path = tmp_path / filename
    state_path.write_text(
        json.dumps(
            {
                "pid": pid,
                "pgid": pgid,
                "created_at": created_at,
                "member_ports": member_ports,
                "name": "llama-swap",
            }
        )
    )
    return state_path


def _own_state_path(tmp_path: Path) -> Path:
    """The state file this process's SwapManager writes for itself."""
    return tmp_path / sm._state_filename(os.getpid(), _GROUP)


def _swap_state(*, pid: int = 123, created_at: float | None = None) -> sm.SwapState:
    """A minimal SwapState for swap-liveness checks."""
    return sm.SwapState(pid=pid, pgid=None, created_at=created_at)


class TestCrossRunReaping:
    def test_start_writes_a_state_file_with_the_swap_pid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        state = json.loads(_own_state_path(tmp_path).read_text())
        assert state["pid"] == 4321
        assert state["name"] == "llama-swap"

    def test_clean_shutdown_removes_the_state_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT)])
        assert _own_state_path(tmp_path).exists()
        mgr.shutdown()
        assert not _own_state_path(tmp_path).exists()

    def test_stale_state_with_dead_pid_cleans_file_without_killing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777)
        _patch_psutil_process(monkeypatch, {})  # pid 7777 no longer exists
        stopped: list[object] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        assert stopped == []  # nothing alive to kill
        # the stale file is gone; this owner's own file records the NEW swap
        assert not (tmp_path / "llama-swap.state.json").exists()
        assert json.loads(_own_state_path(tmp_path).read_text())["pid"] == 4321

    def test_stale_state_with_live_llama_swap_kills_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777)
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap", "-config", "x.json"])
        _patch_psutil_process(monkeypatch, {7777: stale})
        stopped: list[sm.SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        assert [state.pid for state in stopped] == [7777]

    def test_pid_reuse_with_foreign_cmdline_is_not_killed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The recorded pid is alive but now belongs to another program.
        _write_state(tmp_path, pid=7777)
        impostor = _FakePsProcess(7777, cmdline=["/usr/bin/python3", "train.py"])
        _patch_psutil_process(monkeypatch, {7777: impostor})
        stopped: list[object] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        assert stopped == []

    def test_corrupt_state_file_is_skipped_not_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An unparseable file may be a sibling's in-flight write; deleting it
        # would erase a LIVE owner's cross-run reap record.
        corrupt = tmp_path / "llama-swap.state.json"
        corrupt.write_text("{not json")
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        assert corrupt.read_text() == "{not json"
        assert json.loads(_own_state_path(tmp_path).read_text())["pid"] == 4321

    def test_torn_state_file_is_left_in_place_by_reap(self, tmp_path: Path) -> None:
        torn = tmp_path / "llama-swap.state.json"
        torn.write_text('{"pid": 77')  # a sibling's write, caught mid-flight
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert torn.read_text() == '{"pid": 77'

    def test_load_state_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert sm._load_state(tmp_path / "missing.json") is None

    def test_is_live_llama_swap_false_for_dead_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_psutil_process(monkeypatch, {})
        assert sm._is_live_llama_swap(_swap_state(pid=123)) is False

    def test_is_live_llama_swap_false_for_empty_cmdline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A zombie reports an empty cmdline; never treat it as our process.
        _patch_psutil_process(monkeypatch, {123: _FakePsProcess(123, cmdline=[])})
        assert sm._is_live_llama_swap(_swap_state(pid=123)) is False

    def test_is_live_llama_swap_true_when_swap_create_time_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        live = _FakePsProcess(123, cmdline=["/opt/llama-swap"], create_time=42.0)
        _patch_psutil_process(monkeypatch, {123: live})
        assert sm._is_live_llama_swap(_swap_state(pid=123, created_at=42.0)) is True

    def test_swap_pid_reused_by_another_instances_swap_is_not_killed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The recycled pid runs llama-swap, but a DIFFERENT instance's: the
        # create-time mismatch must veto the cmdline match.
        _write_state(tmp_path, pid=7777, created_at=42.0)
        other_swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"], create_time=5000.0)
        _patch_psutil_process(monkeypatch, {7777: other_swap})
        stopped: list[object] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert stopped == []

    def test_swap_with_matching_create_time_is_killed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777, created_at=42.0)
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"], create_time=42.0)
        _patch_psutil_process(monkeypatch, {7777: stale})
        stopped: list[sm.SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert [state.pid for state in stopped] == [7777]

    def test_legacy_state_without_swap_create_time_falls_back_to_cmdline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777, created_at=None)
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"], create_time=5000.0)
        _patch_psutil_process(monkeypatch, {7777: stale})
        stopped: list[sm.SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert [state.pid for state in stopped] == [7777]

    def test_start_records_the_swap_create_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_psutil_process(
            monkeypatch, {4321: _FakePsProcess(4321, cmdline=["llama-swap"], create_time=777.0)}
        )
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        assert json.loads(_own_state_path(tmp_path).read_text())["created_at"] == 777.0

    def test_start_records_the_swap_pgid_on_posix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Platform pinned so the posix pgid lines run on every CI platform.
        monkeypatch.setattr(sm.sys, "platform", "linux")
        monkeypatch.setattr(sm.os, "getpgid", lambda pid: 999, raising=False)
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        assert json.loads(_own_state_path(tmp_path).read_text())["pgid"] == 999

    def test_healthy_engine_is_spared_whoever_started_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Reap must never disagree with bind: an engine answering on its proxy
        # is in use (a reload's own groups, or a bindable sibling engine).
        state_path = _write_state(tmp_path, pid=7777)
        original = state_path.read_text()
        monkeypatch.setattr(sm, "state_is_healthy", lambda _state: True)
        stopped: list[object] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert stopped == []
        assert state_path.read_text() == original

    def test_dead_owner_swap_is_reaped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777)
        swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: swap})  # owner pid 999 is gone
        stopped: list[sm.SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        assert [state.pid for state in stopped] == [7777]

    def test_two_writers_records_coexist_while_healthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A live owner's per-pid file survives a second instance's start.
        owner_a = _FakePsProcess(999, cmdline=["lilbee"], create_time=100.0)
        _patch_psutil_process(monkeypatch, {999: owner_a})
        a_path = _write_state(
            tmp_path,
            pid=7777,
            filename=sm._state_filename(999, _GROUP),
        )
        original = a_path.read_text()
        # A's engine answers on its proxy, so B's reap must spare it.
        monkeypatch.setattr(sm, "state_is_healthy", lambda _state: True)
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr_b = SwapManager(tmp_path, _GROUP)
        mgr_b.start([_launch(WorkerRole.CHAT)])
        assert a_path.read_text() == original  # A's record untouched
        assert _own_state_path(tmp_path).exists()  # B wrote its own
        mgr_b.shutdown()
        assert a_path.read_text() == original  # B's shutdown removes only B's file
        assert not _own_state_path(tmp_path).exists()

    def test_dead_owner_per_pid_state_file_is_reaped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_path = _write_state(tmp_path, pid=7777, filename=sm._state_filename(999, _GROUP))
        swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: swap})  # owner pid 999 is gone
        stopped: list[sm.SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert [state.pid for state in stopped] == [7777]
        assert not state_path.exists()

    def test_legacy_shared_state_file_is_reaped_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pre-per-owner format: the single shared file is still scanned and reaped.
        legacy = _write_state(tmp_path, pid=7777)
        swap = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: swap})
        stopped: list[sm.SwapState] = []
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state))
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert [state.pid for state in stopped] == [7777]
        assert not legacy.exists()
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert len(stopped) == 1  # nothing left to reap a second time


class TestStopStaleSwap:
    def test_terms_the_recorded_process_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: stale})
        monkeypatch.setattr(sm.sys, "platform", "linux")
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        signals: list[tuple[int, int]] = []
        monkeypatch.setattr(
            sm.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)), raising=False
        )
        sm._stop_stale_swap(sm.SwapState(pid=7777, pgid=8888))
        assert signals == [(8888, sm.signal.SIGTERM)]

    def test_escalates_to_sigkill_when_term_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        stale.wait_raises = True
        _patch_psutil_process(monkeypatch, {7777: stale})
        monkeypatch.setattr(sm.sys, "platform", "linux")
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        signals: list[int] = []
        monkeypatch.setattr(sm.os, "killpg", lambda _pgid, sig: signals.append(sig), raising=False)
        waits: list[list[object]] = []

        def _wait_procs(procs: list[object], timeout: float) -> tuple[list, list]:
            waits.append(list(procs))
            return ([], [])

        monkeypatch.setattr(sm.psutil, "wait_procs", _wait_procs)
        sm._stop_stale_swap(sm.SwapState(pid=7777, pgid=8888))
        assert signals == [sm.signal.SIGTERM, sm._SIGKILL]
        # The KILLed swap is awaited so its VRAM is free before the next probe.
        assert [stale] in waits

    def test_signals_the_pid_when_no_pgid_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: stale})
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        sm._stop_stale_swap(sm.SwapState(pid=7777, pgid=None))
        assert stale.signals == [sm.signal.SIGTERM]

    def test_noop_when_process_died_between_checks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_psutil_process(monkeypatch, {})
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        sm._stop_stale_swap(sm.SwapState(pid=7777, pgid=None))  # must not raise

    def test_reaps_surviving_servers_of_the_stale_swap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # llama-swap's servers live in their own process groups, so the group
        # kill misses them; the stale reap must sweep them like shutdown does.
        survivor = _FakeChild(running=True)
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: stale})
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [survivor])
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: ([], []))
        sm._stop_stale_swap(sm.SwapState(pid=7777, pgid=None))
        assert survivor.terminated is True

    def test_reaps_member_port_servers_the_descendant_snapshot_missed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A server that was reparented away from the swap, or respawned after
        # the snapshot, is no longer a descendant; only the recorded member
        # ports find it. Every caller unlinks the record straight after, so a
        # miss here strands it with nothing left to match it against.
        port_server = _FakeChild(running=True)
        stale = _FakePsProcess(7777, cmdline=["/opt/llama-swap"])
        _patch_psutil_process(monkeypatch, {7777: stale})
        monkeypatch.setattr(sm, "_live_children", lambda _pid: [])
        swept: list[tuple[int, ...]] = []
        monkeypatch.setattr(
            sm,
            "_find_orphan_servers",
            lambda ports: swept.append(ports) or [port_server],
        )
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: ([], []))
        sm._stop_stale_swap(sm.SwapState(pid=7777, pgid=None, member_ports=(9101, 9102)))
        assert swept == [(9101, 9102)]
        assert port_server.terminated is True


class TestAtomicStateWrite:
    def test_config_and_state_both_land_via_replace_with_no_tmp_leftovers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A truncating write leaves llama-swap a config it cannot start from."""
        replaced: list[str] = []
        real_replace = os.replace

        def _spy(src: object, dst: object) -> None:
            replaced.append(str(dst))
            real_replace(src, dst)

        monkeypatch.setattr(sm.os, "replace", _spy)
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        SwapManager(tmp_path, _GROUP).start([_launch(WorkerRole.CHAT)])
        config_path = tmp_path / f"llama-swap-{_GROUP.value}.{os.getpid()}.json"
        assert replaced == [str(config_path), str(_own_state_path(tmp_path))]
        assert json.loads(_own_state_path(tmp_path).read_text())["pid"] == 4321
        assert [path for path in tmp_path.iterdir() if path.name.endswith(".tmp")] == []

    def test_half_written_tmp_file_is_invisible_to_the_reap_scan(self, tmp_path: Path) -> None:
        # A live writer's in-flight tmp file must never be parsed as a state
        # record nor removed; dead-writer leftovers are TestStaleTmpCleanup's.
        in_flight = tmp_path / f".llama-swap.state.{os.getpid()}.json.tmp"
        in_flight.write_text('{"pid":')
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert in_flight.exists()


_PARENT_RAISES = "<parent-raises>"


class _FakeParentProc:
    """A stand-in psutil.Process parent for the orphan ownership guard."""

    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class _FakeServerProc:
    """A stand-in psutil.Process for the orphan-server sweep."""

    def __init__(
        self,
        pid: int,
        cmdline: list[str],
        *,
        cmdline_raises: bool = False,
        parent_name: str | None = None,
    ) -> None:
        self.pid = pid
        self._cmdline = cmdline
        self._cmdline_raises = cmdline_raises
        self._parent_name = parent_name
        self.terminated = False
        self.killed = False
        self.info = {"name": Path(next(iter(cmdline), "")).name}

    def cmdline(self) -> list[str]:
        if self._cmdline_raises:
            raise sm.psutil.NoSuchProcess(self.pid)
        return self._cmdline

    def parent(self) -> _FakeParentProc | None:
        if self._parent_name == _PARENT_RAISES:
            raise sm.psutil.NoSuchProcess(self.pid)
        if self._parent_name is None:
            return None
        return _FakeParentProc(self._parent_name)

    def is_running(self) -> bool:
        return True

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class TestOrphanServerReaping:
    def test_start_records_member_ports_in_the_state_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT), _launch(WorkerRole.EMBED)])
        config = json.loads(mgr._config_path.read_text())
        expected = sorted(
            int(entry["cmd"].rsplit(" ", 1)[-1]) for entry in config["models"].values()
        )
        assert json.loads(_own_state_path(tmp_path).read_text())["member_ports"] == expected

    def test_dead_swap_live_server_is_killed_by_name_and_port_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The servers outlive a SIGKILLed swap in their own process groups; the
        # sweep must stop exactly the ones on the recorded ports.
        _write_state(tmp_path, pid=7777, member_ports=[5001, 5002])
        _patch_psutil_process(monkeypatch, {})  # owner and swap are both gone
        orphan = _FakeServerProc(1, ["/opt/llama-server", "-m", "x.gguf", "--port", "5001"])
        recycled = _FakeServerProc(2, ["/usr/bin/python3", "serve.py", "--port", "5002"])
        other_port = _FakeServerProc(3, ["/opt/llama-server", "--port", "9999"])
        no_port = _FakeServerProc(4, ["/opt/llama-server"])
        vanished = _FakeServerProc(5, ["/opt/llama-server"], cmdline_raises=True)
        monkeypatch.setattr(
            sm.psutil,
            "process_iter",
            lambda *a, **k: iter([orphan, recycled, other_port, no_port, vanished]),
        )
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: (list(procs), []))
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert orphan.terminated is True
        assert recycled.terminated is False  # recycled port, foreign binary
        assert other_port.terminated is False
        assert no_port.terminated is False
        assert not (tmp_path / "llama-swap.state.json").exists()

    def test_server_with_a_live_swap_parent_is_spared(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A current run can reuse a stale record's port; its server still has a
        # live llama-swap parent, so the sweep must not touch it.
        _write_state(tmp_path, pid=7777, member_ports=[5001])
        _patch_psutil_process(monkeypatch, {})
        adopted = _FakeServerProc(
            1,
            ["/opt/llama-server", "--port", "5001"],
            parent_name="llama-swap",
        )
        monkeypatch.setattr(sm.psutil, "process_iter", lambda *a, **k: iter([adopted]))
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: (list(procs), []))
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert adopted.terminated is False
        assert not (tmp_path / "llama-swap.state.json").exists()

    def test_server_with_a_foreign_parent_is_still_reaped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777, member_ports=[5001, 5002])
        _patch_psutil_process(monkeypatch, {})
        orphan = _FakeServerProc(
            1,
            ["/opt/llama-server", "--port", "5001"],
            parent_name="launchd",
        )
        parent_vanished = _FakeServerProc(
            2,
            ["/opt/llama-server", "--port", "5002"],
            parent_name=_PARENT_RAISES,
        )
        monkeypatch.setattr(
            sm.psutil, "process_iter", lambda *a, **k: iter([orphan, parent_vanished])
        )
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: (list(procs), []))
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert orphan.terminated is True
        assert parent_vanished.terminated is True

    def test_legacy_state_without_ports_sweeps_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_state(tmp_path, pid=7777)
        _patch_psutil_process(monkeypatch, {})

        def _forbidden() -> object:
            raise AssertionError("process_iter must not run without recorded ports")

        monkeypatch.setattr(sm.psutil, "process_iter", _forbidden)
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert not (tmp_path / "llama-swap.state.json").exists()


class TestStaleTmpCleanup:
    def test_dead_writers_tmp_file_is_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale = tmp_path / ".llama-swap.state.424242.json.tmp"
        stale.write_text("{partial")
        monkeypatch.setattr(sm.psutil, "pid_exists", lambda pid: False)
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert not stale.exists()

    def test_live_writers_tmp_file_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        in_flight = tmp_path / f".llama-swap.state.{os.getpid()}.json.tmp"
        in_flight.write_text("{partial")
        monkeypatch.setattr(sm.psutil, "pid_exists", lambda pid: True)
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert in_flight.exists()

    def test_tmp_file_without_a_pid_is_kept(self, tmp_path: Path) -> None:
        legacy = tmp_path / ".llama-swap.state.json.tmp"
        legacy.write_text("{partial")
        SwapManager(tmp_path, _GROUP).reap_stale()
        assert legacy.exists()


def test_state_owner_pid_parses_state_and_tmp_names() -> None:
    assert sm._state_owner_pid("llama-swap.state.123.json") == 123
    assert sm._state_owner_pid(".llama-swap.state.456.json.tmp") == 456
    assert sm._state_owner_pid("llama-swap.state.json") is None


def test_write_state_is_noop_without_a_process(tmp_path: Path) -> None:
    mgr = SwapManager(tmp_path, _GROUP)
    mgr._write_state()
    assert not _own_state_path(tmp_path).exists()


def test_running_reflects_the_spawned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
    _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
    mgr = SwapManager(tmp_path, _GROUP)
    assert mgr.running is False
    mgr.start([_launch(WorkerRole.CHAT)])
    assert mgr.running is True
    mgr.shutdown()
    assert mgr.running is False


class TestIsLive:
    def test_true_when_proc_alive_and_running_answers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = SwapManager(tmp_path, _GROUP)
        mgr._port = 41999
        mgr._proc = _FakeProc(poll_result=None)  # type: ignore[assignment]
        monkeypatch.setattr(
            sm,
            "_probe_client",
            lambda: _FakeProbeClient(lambda _url: _fake_response(status=200)),
        )
        assert mgr.is_live() is True

    def test_false_when_proc_is_none(self, tmp_path: Path) -> None:
        mgr = SwapManager(tmp_path, _GROUP)
        assert mgr._proc is None
        assert mgr.is_live() is False

    def test_false_when_proc_has_exited(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = SwapManager(tmp_path, _GROUP)
        mgr._port = 41999
        mgr._proc = _FakeProc(poll_result=1)  # type: ignore[assignment]
        assert mgr.is_live() is False

    def test_false_when_running_probe_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mgr = SwapManager(tmp_path, _GROUP)
        mgr._port = 41999
        mgr._proc = _FakeProc(poll_result=None)  # type: ignore[assignment]

        def boom(_url: str) -> object:
            raise OSError("connection refused")

        monkeypatch.setattr(sm, "_probe_client", lambda: _FakeProbeClient(boom))
        assert mgr.is_live() is False

    def test_false_and_no_raise_when_proc_alive_but_port_not_yet_set(self, tmp_path: Path) -> None:
        # Startup-window race: _proc is alive (poll() returns None) but _port is
        # still None. is_live() must return False, not let endpoint()'s
        # ProviderError escape a -> bool method.
        mgr = SwapManager(tmp_path, _GROUP)
        mgr._proc = _FakeProc(poll_result=None)  # type: ignore[assignment]
        assert mgr._port is None
        result = mgr.is_live()
        assert result is False


class TestPerGroupNaming:
    def test_groups_get_distinct_config_and_state_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        chat = SwapManager(tmp_path, SwapGroup.CHAT)
        embed = SwapManager(tmp_path, SwapGroup.EMBED)
        chat.start([_launch(WorkerRole.CHAT)])
        embed.start([_launch(WorkerRole.EMBED)])
        # Each group runs against its own config, so stopping one group's fleet
        # (keyed on config-path identity) can never touch the other's servers.
        assert (tmp_path / sm._config_filename(os.getpid(), "chat")).exists()
        assert (tmp_path / sm._config_filename(os.getpid(), "embed")).exists()
        state_names = {path.name for path in tmp_path.glob("llama-swap.state.*")}
        assert sm._state_filename(os.getpid(), "chat") in state_names
        assert sm._state_filename(os.getpid(), "embed") in state_names

    def test_state_owner_pid_parses_group_qualified_names(self) -> None:
        assert sm._state_owner_pid("llama-swap.state.chat.123.json") == 123
        assert sm._state_owner_pid(".llama-swap.state.embed.456.json.tmp") == 456

    def test_config_owner_pid_parses_and_rejects_legacy(self) -> None:
        assert sm._config_owner_pid("llama-swap-chat.123.json") == 123
        assert sm._config_owner_pid("llama-swap-embed.456.json") == 456
        # A pid-less legacy name has no owner pid.
        assert sm._config_owner_pid("llama-swap-chat.json") is None

    def test_reap_removes_dead_owner_configs_keeps_live_and_legacy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pid = os.getpid()
        live = tmp_path / sm._config_filename(pid, "chat")
        legacy = tmp_path / "llama-swap-embed.json"
        dead = tmp_path / sm._config_filename(pid + 1, "chat")
        for path in (live, legacy, dead):
            path.write_text("{}")
        monkeypatch.setattr(sm.psutil, "pid_exists", lambda p: p == pid)
        sm._clean_stale_configs(tmp_path)
        assert live.exists()  # this owner is alive
        assert legacy.exists()  # pid-less name is left alone
        assert not dead.exists()  # dead owner's orphan config removed


class TestStateFilePersistenceKeys:
    def test_round_trips_proxy_port_and_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT)])
        state = sm._load_state(mgr._state_path)
        assert state is not None
        assert state.proxy_port == mgr._port

    def test_start_writes_an_owned_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT)])
        state = sm._load_state(mgr._state_path)
        assert state is not None and state.pid == 4321

    def test_old_format_files_parse_with_defaults(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps({"pid": 123, "member_ports": [4000]}))
        state = sm._load_state(legacy)
        assert state is not None
        assert state.proxy_port is None
        assert state.engine_pin is None


class TestLiveStateLaunchContract:
    """Live state files must carry the serving contract, so a guest lilbee can
    bind to a running sibling's fleet without reverse-engineering /running."""

    def test_start_records_the_launch_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT)])
        state = sm._load_state(mgr._state_path)
        assert state is not None
        assert len(state.launches) == 1
        assert state.launches[0]["role"] == "chat"
        assert state.launches[0]["model"] == "chat-model"


class TestEnginePinInState:
    def test_start_records_the_engine_pin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_spawn(monkeypatch, _FakeProc(poll_result=None))
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        monkeypatch.setattr(sm, "engine_pin", lambda: "llama-cpp-1.2.3+swap-v9+gguf-v1")
        mgr = SwapManager(tmp_path, _GROUP)
        mgr.start([_launch(WorkerRole.CHAT)])
        state = sm._load_state(mgr._state_path)
        assert state is not None
        assert state.engine_pin == "llama-cpp-1.2.3+swap-v9+gguf-v1"

    def test_legacy_state_without_a_pin_parses_as_none(self, tmp_path: Path) -> None:
        legacy = tmp_path / "legacy.json"
        legacy.write_text(json.dumps({"pid": 123, "member_ports": [4000]}))
        state = sm._load_state(legacy)
        assert state is not None
        assert state.engine_pin is None


class TestBindToLiveEngine:
    """A second lilbee binds to a healthy running engine instead of building one."""

    def _live_state(self, tmp_path: Path, *, pin: str = "pin-a", model: str = "chat-model") -> Path:
        path = tmp_path / sm._state_filename(999_999, _GROUP.value)
        payload = _launch(WorkerRole.CHAT).to_state()
        payload["model"] = model
        path.write_text(
            json.dumps(
                {
                    "pid": 999_998,
                    "member_ports": [4000],
                    "proxy_port": 4100,
                    "launches": [payload],
                    "engine_pin": pin,
                }
            )
        )
        return path

    def test_binds_to_a_healthy_matching_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_path = self._live_state(tmp_path)
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path, _GROUP)
        state = sm._load_state(state_path)
        assert state is not None
        assert mgr.bind(state) is True
        assert mgr.endpoint() == "http://127.0.0.1:4100"
        assert mgr.bound is True

    def test_bind_refuses_a_state_without_a_proxy_port(self, tmp_path: Path) -> None:
        mgr = SwapManager(tmp_path, _GROUP)
        state = sm.SwapState(pid=1, pgid=None)
        assert mgr.bind(state) is False
        assert mgr.bound is False

    def test_bind_refuses_an_unreachable_proxy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_path = self._live_state(tmp_path)
        _patch_http(monkeypatch, _raise_connect_error)
        mgr = SwapManager(tmp_path, _GROUP)
        state = sm._load_state(state_path)
        assert state is not None
        assert mgr.bind(state) is False
        assert mgr.bound is False

    def test_bind_never_writes_a_state_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_path = self._live_state(tmp_path)
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path, _GROUP)
        state = sm._load_state(state_path)
        assert state is not None
        assert mgr.bind(state) is True
        assert not mgr._state_path.exists()
        assert state_path.exists()  # the engine's own record is untouched

    def test_bound_shutdown_never_signals_engine_processes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_path = self._live_state(tmp_path)
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        stopped: list[object] = []
        monkeypatch.setattr(sm, "_stop_own_fleet", lambda *a: stopped.append(a))
        mgr = SwapManager(tmp_path, _GROUP)
        state = sm._load_state(state_path)
        assert state is not None
        assert mgr.bind(state) is True
        mgr.shutdown()
        assert stopped == []
        assert state_path.exists()
        assert mgr._port is None  # binding dropped, manager reusable

    def test_bind_carries_the_contract(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state_path = self._live_state(tmp_path)
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        mgr = SwapManager(tmp_path, _GROUP)
        state = sm._load_state(state_path)
        assert state is not None
        assert mgr.bind(state) is True
        assert mgr._launches_payload[0]["model"] == "chat-model"


class TestStopEngine:
    """The unconditional off switch: stop whatever the dir's state files record."""

    def test_stops_every_recorded_swap_and_unlinks_states(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for group, pid in (("chat", 7001), ("embed", 7002)):
            path = tmp_path / sm._state_filename(999_999, group)
            path.write_text(json.dumps({"pid": pid, "member_ports": [4000]}))
        stopped: list[int] = []
        monkeypatch.setattr(sm, "_is_live_llama_swap", lambda _state: True)  # both alive
        monkeypatch.setattr(sm, "_stop_stale_swap", lambda state: stopped.append(state.pid))
        result = sm.stop_engine(tmp_path)
        assert sorted(stopped) == [7001, 7002]
        assert sorted(result) == ["chat", "embed"]  # both reported as actually stopped
        assert not list(tmp_path.glob(sm._STATE_FILE_GLOB))

    def test_empty_dir_is_a_noop(self, tmp_path: Path) -> None:
        sm.stop_engine(tmp_path)  # no states, no error

    def test_unparseable_state_is_left_alone(self, tmp_path: Path) -> None:
        junk = tmp_path / sm._state_filename(1, "chat")
        junk.write_text("not json{{{")
        sm.stop_engine(tmp_path)
        assert junk.exists()

    def test_reaps_orphan_servers_when_the_swap_is_dead(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The chaos case: llama-swap is dead but its llama-servers (own process
        # groups) survive on the recorded ports. The off switch must reap them by
        # port, not silently drop the record and leave them holding VRAM.
        path = tmp_path / sm._state_filename(999_999, "chat")
        path.write_text(json.dumps({"pid": 999_998, "member_ports": [4000, 4001]}))
        monkeypatch.setattr(sm, "_is_live_llama_swap", lambda _state: False)  # swap dead
        reaped: list[tuple[int, ...]] = []
        monkeypatch.setattr(
            sm, "_find_orphan_servers", lambda ports: reaped.append(ports) or ["srv"]
        )
        monkeypatch.setattr(sm, "_reap_survivors", lambda procs: None)
        result = sm.stop_engine(tmp_path)
        assert reaped == [(4000, 4001)]  # orphans looked up by the recorded ports
        assert result == ["chat"]  # reported as actually stopped (orphans existed)
        assert not path.exists()

    def test_cleans_stale_config_files(self, tmp_path: Path, monkeypatch) -> None:
        # stop_engine leaves the dir as clean as a reap: a dead owner's config file
        # is removed, not stranded for a future ladder reap to find.
        dead_config = tmp_path / sm._config_filename(999_998, "chat")
        dead_config.write_text("{}")
        monkeypatch.setattr(sm.psutil, "pid_exists", lambda _pid: False)  # owner dead
        sm.stop_engine(tmp_path)
        assert not dead_config.exists()


class TestLiveStateHelpers:
    def test_find_live_state_returns_the_newest_for_the_group(self, tmp_path: Path) -> None:
        old = tmp_path / sm._state_filename(111, _GROUP.value)
        old.write_text(json.dumps({"pid": 1, "member_ports": [], "created_at": 100.0}))
        new = tmp_path / sm._state_filename(222, _GROUP.value)
        new.write_text(json.dumps({"pid": 2, "member_ports": [], "created_at": 200.0}))
        state = sm.find_live_state(tmp_path, _GROUP)
        assert state is not None and state.pid == 2

    def test_find_live_state_none_when_group_absent(self, tmp_path: Path) -> None:
        assert sm.find_live_state(tmp_path, _GROUP) is None

    def test_find_live_state_skips_unparseable_files(self, tmp_path: Path) -> None:
        junk = tmp_path / sm._state_filename(1, _GROUP.value)
        junk.write_text("not json{{{")
        assert sm.find_live_state(tmp_path, _GROUP) is None

    def test_state_is_healthy_probes_the_proxy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200))
        state = sm.SwapState(pid=1, pgid=None, proxy_port=4100)
        assert sm.state_is_healthy(state) is True

    def test_state_without_a_port_is_unhealthy(self) -> None:
        state = sm.SwapState(pid=1, pgid=None)
        assert sm.state_is_healthy(state) is False

    def test_refused_probe_is_unhealthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_http(monkeypatch, _raise_connect_error)
        state = sm.SwapState(pid=1, pgid=None, proxy_port=4100)
        assert sm.state_is_healthy(state) is False

    def test_error_status_is_unhealthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A proxy that answers but errors (llama-swap starting up, or a foreign
        # service refusing the path) is not a live engine.
        _patch_http(monkeypatch, lambda _url: _fake_response(status=503))
        state = sm.SwapState(pid=1, pgid=None, proxy_port=4100)
        assert sm.state_is_healthy(state) is False

    def test_recycled_port_responder_is_not_our_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A dead engine's port taken over by an unrelated service that 200s an
        # unknown path (no {"running": [...]} body) must NOT read as healthy, or
        # inference clients would bind to a non-engine endpoint forever.
        _patch_http(monkeypatch, lambda _url: _fake_response(status=200, payload={"ok": True}))
        state = sm.SwapState(pid=1, pgid=None, proxy_port=4100)
        assert sm.state_is_healthy(state) is False

    def test_non_json_responder_is_not_our_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A 200 with a non-JSON body (a plain HTTP server) also fails the shape check.
        def _html(_url):
            resp = _fake_response(status=200)
            resp.json = lambda: (_ for _ in ()).throw(ValueError("not json"))
            return resp

        _patch_http(monkeypatch, _html)
        state = sm.SwapState(pid=1, pgid=None, proxy_port=4100)
        assert sm.state_is_healthy(state) is False


@pytest.mark.parametrize(
    "leak",
    [PermissionError(13, "force permission denied"), SystemError("result with an exception set")],
    ids=["permission-error", "system-error"],
)
def test_owned_swap_scan_skips_processes_that_deny_inspection(monkeypatch, leak):
    """macOS psutil mishandles entitlement-protected binaries (sysctl
    KERN_PROCARGS2), leaking raw PermissionError or C-extension SystemError;
    one such process must not break the sweep."""
    from pathlib import Path
    from unittest import mock

    import psutil

    from lilbee.providers.fleet import swap_manager

    config_path = Path("/tmp/x.json")
    # name() is cheap and readable; only cmdline() (KERN_PROCARGS2) is protected,
    # so both pass the name pre-filter and the leak surfaces at cmdline().
    denied = mock.MagicMock()
    denied.info = {"name": "llama-swap"}
    denied.cmdline.side_effect = leak
    visible = mock.MagicMock()
    visible.info = {"name": "llama-swap"}
    visible.cmdline.return_value = ["/opt/bin/llama-swap", "-config", str(config_path)]
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **k: [denied, visible])

    swaps = swap_manager._swaps_for_config(config_path)
    assert swaps == [visible]


class TestTeardownHelpers:
    """Direct coverage for the process-teardown primitives every stop path uses."""

    def test_live_children_empty_for_a_dead_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _gone(pid: int):
            raise sm.psutil.NoSuchProcess(pid)

        monkeypatch.setattr(sm.psutil, "Process", _gone)
        assert sm._live_children(999_999) == []

    def test_reap_survivors_terminates_then_kills_the_stubborn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Child:
            def __init__(self) -> None:
                self.terminated = False
                self.killed = False

            def is_running(self) -> bool:
                return True

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.killed = True

        child = _Child()
        # First wait: the child survives SIGTERM; second wait (in _await_killed):
        # it is gone.
        waits = iter([([], [child]), ([child], [])])
        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: next(waits))
        sm._reap_survivors([child])
        assert child.terminated and child.killed

    def test_hard_stop_proc_escalates_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Proc:
            def __init__(self) -> None:
                self.terminated = False
                self.killed = False

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout: float) -> None:
                raise sm.psutil.TimeoutExpired(timeout)

            def kill(self) -> None:
                self.killed = True

        proc = _Proc()
        sm._hard_stop_proc(proc)
        assert proc.terminated and proc.killed

    def test_live_children_lists_a_real_processes_children(self) -> None:
        """A list-shaped return proves nothing: every stop path reaps by pid.

        _live_children feeds the orphan reaping for llama-servers that outlive
        their llama-swap, so a version returning [] unconditionally would leave
        those holding VRAM while the suite stayed green.
        """
        import sys

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert proc.pid in [child.pid for child in sm._live_children(os.getpid())]
        finally:
            proc.terminate()
            proc.wait(timeout=30)

    def test_live_children_of_an_exited_process_is_empty(self) -> None:
        """The reaper asks about pids that may already be gone."""
        import sys

        proc = subprocess.Popen([sys.executable, "-c", ""])
        proc.wait(timeout=30)
        assert sm._live_children(proc.pid) == []

    def test_await_killed_warns_for_a_sigkill_survivor(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _Immortal:
            pid = 424_242

        monkeypatch.setattr(sm.psutil, "wait_procs", lambda procs, timeout: ([], [_Immortal()]))
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.swap_manager"):
            sm._await_killed([_Immortal()])
        assert any("survived SIGKILL" in record.message for record in caplog.records)


def test_stale_config_tmp_of_a_dead_writer_is_swept(tmp_path: Path) -> None:
    """The sweep must see config leftovers too, not just state ones."""
    dead = tmp_path / ".llama-swap-chat.999999.json.tmp"
    dead.write_text("half")
    live = tmp_path / f".llama-swap-chat.{os.getpid()}.json.tmp"
    live.write_text("in flight")
    sm._clean_stale_tmp_files(tmp_path)
    assert not dead.exists()
    assert live.exists()  # a live writer's file in flight is never touched


def test_probe_client_is_shared_across_calls() -> None:
    """The engine probes reuse one client: httpx.get would build a fresh Client --
    and a fresh SSL context, loading the system CA bundle -- on every poll, which
    the task bar runs at up to 10 Hz."""
    sm._probe_client.cache_clear()
    client = sm._probe_client()
    try:
        assert sm._probe_client() is client
    finally:
        # Close before dropping the cache entry so the pool is not leaked.
        client.close()
        sm._probe_client.cache_clear()


class TestEstimateReadback:
    """A newly-ready engine is compared to what placement charged it, once."""

    @staticmethod
    def _manager(tmp_path):
        from lilbee.providers.fleet.groups import SwapGroup
        from lilbee.providers.fleet.swap_manager import SwapManager

        return SwapManager(tmp_path, SwapGroup.CHAT)

    @staticmethod
    def _launch(est: int):
        from lilbee.providers.fleet.launch import InstanceLaunch
        from lilbee.providers.roles import WorkerRole

        return InstanceLaunch(
            role=WorkerRole.CHAT,
            argv=["/bin/llama-server"],
            env_overrides={},
            model="org/chat.gguf",
            est_vram_bytes=est,
        )

    def test_a_ready_engine_is_checked_against_its_estimate(self, tmp_path, caplog) -> None:
        from lilbee.providers.fleet.readback import engine_log_path

        manager = self._manager(tmp_path)
        manager._launch_by_model = {"chat-0": self._launch(4 * 1024**3)}
        log_dir = manager._log_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        engine_log_path(log_dir, "chat-0").write_text(
            "load_tensors:  CUDA0 model buffer size =  8192.00 MiB\n"
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            manager._check_estimates({"chat-0"})
        assert "planned for 4.0 GiB" in caplog.text

    def test_it_does_not_repeat_on_every_readiness_poll(self, tmp_path, caplog) -> None:
        from lilbee.providers.fleet.readback import engine_log_path

        manager = self._manager(tmp_path)
        manager._launch_by_model = {"chat-0": self._launch(4 * 1024**3)}
        log_dir = manager._log_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        engine_log_path(log_dir, "chat-0").write_text(
            "load_tensors:  CUDA0 model buffer size =  8192.00 MiB\n"
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            manager._check_estimates({"chat-0"})
            manager._check_estimates({"chat-0"})
        assert caplog.text.count("planned for") == 1

    def test_an_unsized_model_is_skipped(self, tmp_path, caplog) -> None:
        # A model the estimator could not size is enrolled at 0; there is nothing
        # to compare it to, and a warning about the comparison would be noise.
        from lilbee.providers.fleet.readback import engine_log_path

        manager = self._manager(tmp_path)
        manager._launch_by_model = {"chat-0": self._launch(0)}
        log_dir = manager._log_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        engine_log_path(log_dir, "chat-0").write_text(
            "load_tensors: CUDA0 model buffer size = 1.00 MiB\n"
        )
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            manager._check_estimates({"chat-0"})
        assert caplog.text == ""

    def test_a_ready_engine_that_wrote_no_log_says_so(self, tmp_path, caplog) -> None:
        # The engine is up, so an absent log is not "too early": it means the
        # build never accepted the settings that produce one. Silent, that reads
        # as a passing check forever, which is how a wrong variable name hid.
        manager = self._manager(tmp_path)
        manager._launch_by_model = {"chat-0": self._launch(4 * 1024**3)}
        manager._log_path.parent.mkdir(parents=True, exist_ok=True)
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            manager._check_estimates({"chat-0"})
        assert "wrote no log" in caplog.text

    def test_an_unknown_model_id_is_skipped(self, tmp_path, caplog) -> None:
        manager = self._manager(tmp_path)
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.readback"):
            manager._check_estimates({"ghost-0"})
        assert caplog.text == ""
