"""End-to-end multi-gpu tests: a real llama-server stub subprocess + real httpx.

These spawn a real stub server in its own process, drive the actual Fleet, client,
and FleetProvider against it over real sockets, and tear it down. POSIX-only: the
fleet's process-group teardown is a Unix mechanism (the Windows branch is covered
by the unit tests in test_multi_gpu_fleet.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lilbee.providers.multi_gpu import provider as prov_mod
from lilbee.providers.multi_gpu.devices import FleetDevice
from lilbee.providers.multi_gpu.fleet import Fleet, InstanceLaunch
from lilbee.providers.multi_gpu.placement import InstancePlan, ModelPlacementInput, Placement
from lilbee.providers.worker.transport import WorkerRole

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group lifecycle")

_STUB = Path(__file__).parent / "_llama_server_stub.py"
_GB = 1024**3


def _launch(tmp_path: Path, role: WorkerRole) -> InstanceLaunch:
    # argv carries no --port; FleetServer.spawn claims one and appends it, which
    # the stub then reads back from its argv.
    return InstanceLaunch(
        role=role,
        argv=[sys.executable, str(_STUB)],
        env_overrides={},
        model="stub.gguf",
        port_file=tmp_path / f"llama-server-{role.value}.port",
    )


def test_fleet_serves_chat_and_embed_over_real_http(tmp_path: Path) -> None:
    fleet = Fleet(ready_timeout=15.0, data_dir=tmp_path)
    try:
        fleet.start([_launch(tmp_path, WorkerRole.CHAT), _launch(tmp_path, WorkerRole.EMBED)])
        chat = fleet.healthy_clients(WorkerRole.CHAT)[0]
        assert chat.chat([{"role": "user", "content": "hi"}]) == "stub-chat"
        streamed = "".join(chat.chat([{"role": "user", "content": "hi"}], stream=True))
        assert streamed == "stub-chat"
        embeds = fleet.healthy_clients(WorkerRole.EMBED)[0].embed(["a", "b"])
        assert len(embeds) == 2
        assert embeds[0] == [0.5, 0.5]
    finally:
        fleet.shutdown()
    assert not (tmp_path / "llama-server-chat.port").exists()


def test_fleet_restarts_a_killed_server(tmp_path: Path) -> None:
    fleet = Fleet(ready_timeout=15.0, data_dir=tmp_path)
    try:
        fleet.start([_launch(tmp_path, WorkerRole.CHAT)])
        server = fleet._servers[0]
        old_pid = server._proc.pid
        server._proc.kill()  # simulate a crash
        server._proc.wait()
        fleet._restart_dead()  # the monitor's step, driven directly
        assert server.is_alive()
        assert server._proc.pid != old_pid  # a fresh process
        client = fleet.healthy_clients(WorkerRole.CHAT)[0]
        assert client.chat([{"role": "user", "content": "x"}]) == "stub-chat"
    finally:
        fleet.shutdown()


def test_fleet_provider_routes_chat_to_a_real_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
    monkeypatch.setattr(prov_mod, "resolve_llama_server_binary", lambda: Path(sys.executable))
    monkeypatch.setattr(prov_mod, "probe_devices", lambda _binary: [device])
    monkeypatch.setattr(
        prov_mod,
        "_server_model_inputs",
        lambda: ([ModelPlacementInput(WorkerRole.CHAT, 5 * _GB)], {WorkerRole.CHAT: "ref"}),
    )
    monkeypatch.setattr(
        prov_mod,
        "plan_placement",
        lambda inputs, devices: Placement(
            instances=(InstancePlan(WorkerRole.CHAT, (0,)),), in_process_roles=()
        ),
    )
    monkeypatch.setattr(
        prov_mod,
        "_launch_for",
        lambda plan, ref, binary, data_dir, by_index: _launch(tmp_path, plan.role),
    )
    provider = prov_mod.FleetProvider()
    try:
        assert provider.chat([{"role": "user", "content": "hi"}]) == "stub-chat"
    finally:
        provider.shutdown()
