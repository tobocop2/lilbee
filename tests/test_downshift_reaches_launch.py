"""The downshift has to change the launch, not just the estimate.

Recording a step and re-planning is only a recovery if the argv that comes out
is different. Applied to the placement estimate alone, the retry respawns a
byte-identical command and the engine dies exactly as it did before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.roles import WorkerRole

_GB = 1024**3
_LAUNCH_CTX = 8192


def _launch_ctx(tmp_path, monkeypatch, role: WorkerRole) -> int:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 1000)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning_mod, "_vision_mmproj", lambda _r: Path("/m/mmproj.gguf"))
    # The role's own resolver is a pure function of model and config: it is the
    # thing that never hears about the downshift.
    monkeypatch.setattr(planning_mod, "_role_ctx", lambda _r, _p, _m, *_a: _LAUNCH_CTX)
    monkeypatch.setattr(planning_mod, "_slots_for", lambda *_a, **_k: 1)
    device = FleetDevice("CUDA", 0, "gpu", 24 * _GB, 23 * _GB)
    plan = InstancePlan(role=role, devices=(0,))
    launch = planning_mod._launch_for(plan, "ref", Path("/bin/llama-server"), {0: device})
    argv = launch.argv
    return int(argv[argv.index("--ctx-size") + 1])


@pytest.mark.parametrize(
    "role", [WorkerRole.EMBED, WorkerRole.RERANK, WorkerRole.VISION, WorkerRole.CHAT]
)
def test_a_recorded_step_shrinks_the_launched_context(tmp_path, monkeypatch, role) -> None:
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "num_ctx", None, raising=False)
    planning_mod.clear_ctx_downshift()
    before = _launch_ctx(tmp_path, monkeypatch, role)
    planning_mod.record_ctx_downshift(role)
    after = _launch_ctx(tmp_path, monkeypatch, role)
    planning_mod.clear_ctx_downshift()
    assert after < before, f"{role.value} relaunches at the same context after a load OOM"


def test_a_pinned_chat_context_is_still_never_shrunk(tmp_path, monkeypatch) -> None:
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "num_ctx", _LAUNCH_CTX, raising=False)
    planning_mod.clear_ctx_downshift()
    before = _launch_ctx(tmp_path, monkeypatch, WorkerRole.CHAT)
    planning_mod.record_ctx_downshift(WorkerRole.CHAT)
    after = _launch_ctx(tmp_path, monkeypatch, WorkerRole.CHAT)
    planning_mod.clear_ctx_downshift()
    assert after == before
