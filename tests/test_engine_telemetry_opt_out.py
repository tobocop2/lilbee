"""Every engine lilbee starts opts out of telemetry, whatever its adapter returns."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from lilbee.providers.fleet import gpu_env, planning
from lilbee.providers.fleet.engines import EngineAdapter
from lilbee.providers.fleet.engines.choice import ADAPTERS
from lilbee.providers.roles import RoleTarget

OPT_OUT = {"HF_HUB_DISABLE_TELEMETRY": "1", "DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
ADAPTER_PARAMS = [pytest.param(adapter, id=name.value) for name, adapter in ADAPTERS.items()]


@pytest.fixture(autouse=True)
def _telemetry_on_in_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in OPT_OUT:
        monkeypatch.setenv(var, "0")


def test_every_engine_is_covered() -> None:
    assert len(ADAPTER_PARAMS) >= 5


@pytest.mark.parametrize("adapter", ADAPTER_PARAMS)
@pytest.mark.parametrize("target", list(RoleTarget))
def test_launch_env_of_every_adapter_holds_the_opt_outs(
    adapter: EngineAdapter, target: RoleTarget
) -> None:
    env = planning._launch_env(adapter, (), target)
    assert {k: env.get(k) for k in OPT_OUT} == OPT_OUT


@pytest.mark.parametrize("adapter", ADAPTER_PARAMS)
def test_an_adapter_cannot_remove_or_change_an_opt_out(
    adapter: EngineAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    hostile = {k: "0" for k in OPT_OUT}
    monkeypatch.setattr(type(adapter), "runtime_env", lambda self: hostile, raising=False)
    monkeypatch.setattr(type(adapter), "target_env", lambda self, target: hostile, raising=False)
    env = planning._launch_env(adapter, (), RoleTarget.GPU)
    assert {k: env[k] for k in OPT_OUT} == OPT_OUT


def test_the_spawn_environment_overrides_a_telemetry_enabling_parent() -> None:
    env = gpu_env.engine_environ()
    assert {k: env[k] for k in OPT_OUT} == OPT_OUT


def test_opt_out_table_is_read_only() -> None:
    assert isinstance(gpu_env.ENGINE_TELEMETRY_OPT_OUT, Mapping)
    with pytest.raises(TypeError):
        gpu_env.ENGINE_TELEMETRY_OPT_OUT["X"] = "1"  # type: ignore[index]


def test_a_tei_cpu_role_still_hides_every_card() -> None:
    tei = next(a for n, a in ADAPTERS.items() if n.value == "tei")
    env = planning._launch_env(tei, (), RoleTarget.CPU)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert planning._launch_env(tei, (), RoleTarget.GPU).get("CUDA_VISIBLE_DEVICES") != ""


TRACE_EXPORT_VARS = ("OTLP_ENDPOINT",)


@pytest.mark.parametrize("var", TRACE_EXPORT_VARS)
def test_a_host_trace_endpoint_does_not_reach_the_engine_environment(
    var: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(var, "http://collector.invalid:4317")
    assert var not in gpu_env.engine_environ()


@pytest.mark.parametrize("adapter", ADAPTER_PARAMS)
def test_no_adapter_launch_env_carries_a_trace_endpoint(
    adapter: EngineAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in TRACE_EXPORT_VARS:
        monkeypatch.setenv(var, "http://collector.invalid:4317")
    env = {**gpu_env.engine_environ(), **planning._launch_env(adapter, (), RoleTarget.GPU)}
    assert not set(TRACE_EXPORT_VARS) & set(env)
