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
from lilbee.providers.multi_gpu.fleet import Fleet, InstanceLaunch, pick_free_port
from lilbee.providers.multi_gpu.placement import InstancePlan, ModelPlacementInput, Placement
from lilbee.providers.worker.transport import WorkerRole

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group lifecycle")

_STUB = Path(__file__).parent / "_llama_server_stub.py"


def _launch(tmp_path: Path, role: WorkerRole, port: int) -> InstanceLaunch:
    return InstanceLaunch(
        role=role,
        argv=[sys.executable, str(_STUB), "--port", str(port)],
        devices=(0,),
        port=port,
        model="stub.gguf",
        port_file=tmp_path / f"llama-server-{role.value}.port",
    )


def test_fleet_serves_chat_and_embed_over_real_http(tmp_path: Path) -> None:
    fleet = Fleet(ready_timeout=15.0)
    chat_port, embed_port = pick_free_port(), pick_free_port()
    try:
        clients = fleet.start(
            [
                _launch(tmp_path, WorkerRole.CHAT, chat_port),
                _launch(tmp_path, WorkerRole.EMBED, embed_port),
            ]
        )
        chat = clients[WorkerRole.CHAT][0]
        assert chat.chat([{"role": "user", "content": "hi"}]) == "stub-chat"
        streamed = "".join(chat.chat([{"role": "user", "content": "hi"}], stream=True))
        assert streamed == "stub-chat"
        embeds = clients[WorkerRole.EMBED][0].embed(["a", "b"])
        assert len(embeds) == 2
        assert embeds[0] == [0.5, 0.5]
    finally:
        fleet.shutdown()
    assert not (tmp_path / "llama-server-chat.port").exists()


def test_fleet_provider_routes_chat_to_a_real_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port = pick_free_port()
    monkeypatch.setattr(
        "lilbee.providers.llama_cpp.gpu_select.enumerate_gpu_vram",
        lambda: [(0, 24 * 1024**3)],
    )
    monkeypatch.setattr(
        prov_mod,
        "_server_model_inputs",
        lambda: ([ModelPlacementInput(WorkerRole.CHAT, 5 * 1024**3)], {WorkerRole.CHAT: "ref"}),
    )
    monkeypatch.setattr(
        prov_mod,
        "plan_placement",
        lambda inputs, devices: Placement(
            instances=(InstancePlan(WorkerRole.CHAT, (0,)),), in_process_roles=()
        ),
    )
    monkeypatch.setattr(prov_mod, "resolve_llama_server_binary", lambda: Path(sys.executable))
    monkeypatch.setattr(
        prov_mod,
        "_launch_for",
        lambda plan, ref, binary, data_dir: _launch(tmp_path, plan.role, port),
    )
    provider = prov_mod.FleetProvider()
    try:
        assert provider.chat([{"role": "user", "content": "hi"}]) == "stub-chat"
    finally:
        provider.shutdown()
