"""A role names a device target: the parser, the plan, the real adapters and the refusal."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.catalog import runnable
from lilbee.providers.fleet import adapters, planning, readback
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.engines import LaunchRequest
from lilbee.providers.fleet.engines.ik_llama import IK_LLAMA
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.engines.mistral import MISTRAL, MistralAdapter
from lilbee.providers.fleet.engines.tei import TEI, TeiAdapter
from lilbee.providers.fleet.placement import InstancePlan, placement_from_spec
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec, RolePlacement
from lilbee.providers.roles import RoleTarget, WorkerRole

GIB = 1 << 30
LLAMA_HELP = "  --device <dev1,dev2,..>  devices\n  --fit-target N  margin\n"
OLD_HELP = "  --device-draft <dev>  draft devices\n"


def _no_estimate(role: WorkerRole, ratio: tuple[int, ...]) -> tuple[int, ...]:
    raise AssertionError(f"{role} was sized for a card")


def _build(monkeypatch: pytest.MonkeyPatch, adapter: type, backend: str | None) -> None:
    monkeypatch.setattr(runnable, "session_host", lambda: None)
    picked = SimpleNamespace(backend=backend)
    monkeypatch.setattr(adapter, "pick_build", lambda _self, _host: picked)


def _help(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(adapters, "engine_help", lambda _binary: text)
    monkeypatch.setattr(readback, "engine_help", lambda _binary: text)


def _llama_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plan: InstancePlan, engine):
    """The launch the planner builds for *plan* on one card, through the real adapter."""
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 1000)
    binary = tmp_path / "server"
    _help(monkeypatch, LLAMA_HELP)
    monkeypatch.setattr(type(engine), "binary", lambda _self: binary)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _r: model)
    monkeypatch.setattr("lilbee.providers.fleet.engines.llama.read_gguf_metadata", lambda _p: {})
    monkeypatch.setattr(planning, "_role_ctx", lambda _r, _p, _m, *_a, engine: 4096)
    monkeypatch.setattr("lilbee.providers.fleet.engines.llama.llama_server_runtime_env", dict)
    card = FleetDevice("CUDA", 0, "gpu0", 24 * GIB, 23 * GIB)
    return planning._launch_for(plan, "ref", {0: card}, engine=engine)


def _flag(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None


class TestSpecTargetClosedSet:
    """Row spec_target_closed_set."""

    @pytest.mark.parametrize(
        ("raw", "target", "replicas"),
        [
            ('{"embed": {"target": "cpu"}}', RoleTarget.CPU, 1),
            ('{"embed": {"target": "cpu", "replicas": 2}}', RoleTarget.CPU, 2),
            ('{"embed": {"target": "npu"}}', RoleTarget.NPU, 1),
            ('{"embed": {"target": "auto"}}', RoleTarget.AUTO, 1),
        ],
    )
    def test_a_target_with_no_card_parses_and_round_trips(
        self, raw: str, target: RoleTarget, replicas: int
    ) -> None:
        spec = PlacementSpec.from_json(raw)
        assert spec.roles[WorkerRole.EMBED] == RolePlacement(
            devices=(), replicas=replicas, target=target
        )
        assert PlacementSpec.from_json(spec.to_json()) == spec

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            (
                '{"target": "tpu"}',
                "embed: unknown target 'tpu'; choose auto, cpu, gpu, npu",
            ),
            (
                '{"target": null}',
                "embed: unknown target None; choose auto, cpu, gpu, npu",
            ),
            (
                '{"target": ["cpu"]}',
                "embed: unknown target ['cpu']; choose auto, cpu, gpu, npu",
            ),
            (
                '{"target": "cpu", "devices": [0]}',
                "embed: the cpu target takes no devices. "
                "Remove devices, or set the devices with no target.",
            ),
            (
                '{"target": "cpu", "devices": []}',
                "embed: the cpu target takes no devices. "
                "Remove devices, or set the devices with no target.",
            ),
            (
                '{"target": "cpu", "tensor_split": [1]}',
                "embed: the cpu target takes no tensor_split. "
                "Remove tensor_split, or set the devices with no target.",
            ),
            (
                '{"target": "npu", "replicas": 2}',
                "embed: the npu target takes no replicas. "
                "Remove replicas, or set the devices with no target.",
            ),
            (
                '{"target": "auto", "devices": [0], "replicas": 2}',
                "embed: the auto target takes no devices or replicas. "
                "Remove devices, replicas, or set the devices with no target.",
            ),
            ('{"target": "cpu", "replicas": 0}', "embed: replicas must be >= 1"),
            ('{"target": "gpu"}', "embed: at least one device is required"),
            ('{"devices": []}', "embed: at least one device is required"),
        ],
    )
    def test_refused_entries_name_the_value(self, entry: str, message: str) -> None:
        with pytest.raises(PlacementError) as caught:
            PlacementSpec.from_json(f'{{"embed": {entry}}}')
        assert str(caught.value) == message


class TestSpecOldFormSamePlan:
    """Row spec_old_form_same_plan: the plans and JSON below were read at the base."""

    @pytest.mark.parametrize(
        ("raw", "stored"),
        [
            ('{"chat": {"devices": [0]}}', '{"chat": {"devices": [0]}}'),
            (
                '{"chat": {"devices": [0, 1], "tensor_split": [1, 2]}, '
                '"embed": {"devices": [1], "replicas": 2}}',
                '{"chat": {"devices": [0, 1], "tensor_split": [1, 2]}, '
                '"embed": {"devices": [1], "replicas": 2}}',
            ),
            ('{"chat": {"devices": [0], "target": "gpu"}}', '{"chat": {"devices": [0]}}'),
        ],
    )
    def test_stored_form_has_no_target_key(self, raw: str, stored: str) -> None:
        assert PlacementSpec.from_json(raw).to_json() == stored

    def test_plan_of_an_old_spec(self) -> None:
        spec = PlacementSpec.from_json(
            '{"chat": {"devices": [0, 1], "tensor_split": [1, 2]}, '
            '"embed": {"devices": [1], "replicas": 2}}'
        )
        plan = placement_from_spec(
            spec,
            (WorkerRole.CHAT, WorkerRole.EMBED),
            {0: 8 * GIB, 1: 8 * GIB},
            estimate_peak=lambda _role, ratio: tuple(GIB for _ in ratio),
        )
        assert plan.instances == (
            InstancePlan(WorkerRole.CHAT, (0, 1), (1, 2), 0, RoleTarget.GPU),
            InstancePlan(WorkerRole.EMBED, (1,), (), 0, RoleTarget.GPU),
            InstancePlan(WorkerRole.EMBED, (1,), (), 1, RoleTarget.GPU),
        )
        assert plan.unplaceable_roles == ()


class TestPlan:
    def test_cpu_role_leaves_the_card_list_and_is_not_charged(self) -> None:
        spec = PlacementSpec.from_json('{"embed": {"target": "cpu", "replicas": 2}}')
        plan = placement_from_spec(spec, (WorkerRole.EMBED,), {}, estimate_peak=_no_estimate)
        assert plan.instances == (
            InstancePlan(WorkerRole.EMBED, (), (), 0, RoleTarget.CPU),
            InstancePlan(WorkerRole.EMBED, (), (), 1, RoleTarget.CPU),
        )


class TestCpuRoleArgvFromRealAdapter:
    """Row cpu_role_argv_from_real_adapter, through the planner's launch."""

    @pytest.mark.parametrize("engine", [LLAMA, IK_LLAMA], ids=lambda e: e.name.value)
    def test_llama_family_cpu_target(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine
    ) -> None:
        plan = InstancePlan(WorkerRole.EMBED, (), target=RoleTarget.CPU)
        launch = _llama_launch(monkeypatch, tmp_path, plan, engine)
        assert _flag(launch.argv, "--device") == "none"
        assert _flag(launch.argv, "--n-gpu-layers") == "0"
        assert "--fit-target" not in launch.argv
        assert "CUDA_VISIBLE_DEVICES" not in launch.env_overrides

    def test_llama_cpu_chat_drops_the_fit_margin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(planning, "_chat_fit_target_mib", lambda *_a, **_k: (512,))
        plan = InstancePlan(WorkerRole.CHAT, (), target=RoleTarget.CPU)
        launch = _llama_launch(monkeypatch, tmp_path, plan, LLAMA)
        assert "--fit-target" not in launch.argv
        assert _flag(launch.argv, "--n-gpu-layers") == "0"

    def test_llama_gpu_role_keeps_its_card(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        plan = InstancePlan(WorkerRole.EMBED, (0,))
        launch = _llama_launch(monkeypatch, tmp_path, plan, LLAMA)
        assert _flag(launch.argv, "--device") != "none"
        assert _flag(launch.argv, "--n-gpu-layers") == "-1"
        assert launch.env_overrides["CUDA_VISIBLE_DEVICES"] == "0"

    @pytest.mark.parametrize(
        ("backend", "target", "flags"),
        [
            ("cuda128-sm89", RoleTarget.CPU, ["--cpu"]),
            ("cuda128-sm89", RoleTarget.GPU, ["--paged-attn", "off"]),
            ("cpu", RoleTarget.CPU, ["--cpu"]),
            ("cpu", RoleTarget.GPU, ["--cpu"]),
        ],
    )
    def test_mistral_target_decides_before_the_build(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        backend: str,
        target: RoleTarget,
        flags: list[str],
    ) -> None:
        _build(monkeypatch, MistralAdapter, backend)
        request = LaunchRequest(
            binary=tmp_path / "tool",
            model_path=tmp_path / "model",
            devices=(),
            n_gpu_layers=99,
            slots=1,
            ctx_per_slot=4096,
            target=target,
        )
        spec = MISTRAL.role_spec(WorkerRole.EMBED, None, None)
        assert MISTRAL.launch_argv(spec, request) == [
            str(tmp_path / "tool"),
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "{port}",
            "embedding",
            *flags,
            "-m",
            str(tmp_path / "model"),
        ]

    def test_a_request_with_no_target_runs_on_the_gpu_flags(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _build(monkeypatch, MistralAdapter, "cuda128-sm89")
        request = LaunchRequest(
            binary=tmp_path / "tool",
            model_path=tmp_path / "model",
            devices=(),
            n_gpu_layers=99,
            slots=1,
            ctx_per_slot=4096,
        )
        spec = MISTRAL.role_spec(WorkerRole.EMBED, None, None)
        argv = MISTRAL.launch_argv(spec, request)
        assert _flag(argv, "--paged-attn") == "off"
        assert "--cpu" not in argv


class TestDeviceTargets:
    def test_llama_with_the_device_flag_offers_the_cpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _help(monkeypatch, LLAMA_HELP)
        offered = frozenset({RoleTarget.GPU, RoleTarget.CPU})
        assert LLAMA.device_targets(tmp_path / "server") == offered
        assert IK_LLAMA.device_targets(tmp_path / "server") == offered

    def test_llama_help_naming_only_a_longer_flag_offers_no_cpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _help(monkeypatch, OLD_HELP)
        assert LLAMA.device_targets(tmp_path / "server") == frozenset({RoleTarget.GPU})

    @pytest.mark.parametrize(
        ("adapter", "engine", "backend", "offered"),
        [
            (MistralAdapter, MISTRAL, "cuda128-sm89", {RoleTarget.CPU, RoleTarget.GPU}),
            (MistralAdapter, MISTRAL, "cpu", {RoleTarget.CPU}),
            (MistralAdapter, MISTRAL, None, {RoleTarget.CPU}),
            (TeiAdapter, TEI, "cuda", {RoleTarget.GPU}),
            (TeiAdapter, TEI, "cpu", {RoleTarget.CPU}),
        ],
    )
    def test_build_decides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        adapter: type,
        engine,
        backend: str | None,
        offered: set[RoleTarget],
    ) -> None:
        _build(monkeypatch, adapter, backend)
        assert engine.device_targets(tmp_path / "tool") == frozenset(offered)


class _NeverAsked:
    """An engine a gpu role must not question about targets."""

    takes_device_names = True

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        raise AssertionError("a gpu role asked for device targets")

    def binary(self) -> Path:
        raise AssertionError("a gpu role resolved a binary for the target check")


class TestTargetRefusedPlainMessage:
    """Row target_refused_plain_message, through the planner's own resolve."""

    def _resolve(self, raw: str, engine, devices: list[FleetDevice] | None = None):
        return planning._resolve_placement(
            PlacementSpec.from_json(raw),
            [],
            {WorkerRole.EMBED: "ref"},
            devices or [],
            engines={WorkerRole.EMBED: engine},
            unified_budget=None,
        )

    @pytest.mark.parametrize(
        ("target", "message"),
        [
            (
                "cpu",
                "embed: this engine's installed build cannot run on the CPU. "
                "Choose a GPU or install the CPU build.",
            ),
            (
                "npu",
                "embed: this engine's installed build cannot run on an NPU. "
                "Choose a GPU or the CPU.",
            ),
            (
                "auto",
                "embed: the auto target is not available yet. "
                "Name the devices or set the target to cpu.",
            ),
        ],
    )
    def test_refusal_message(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str, message: str
    ) -> None:
        _help(monkeypatch, OLD_HELP)
        monkeypatch.setattr(type(LLAMA), "binary", lambda _self: tmp_path / "server")
        with pytest.raises(PlacementError) as caught:
            self._resolve(f'{{"embed": {{"target": "{target}"}}}}', LLAMA)
        assert str(caught.value) == message

    def test_usable_cpu_target_is_planned_without_the_card(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _help(monkeypatch, LLAMA_HELP)
        monkeypatch.setattr(type(LLAMA), "binary", lambda _self: tmp_path / "server")
        plan = self._resolve('{"embed": {"target": "cpu"}}', LLAMA)
        assert plan.instances == (InstancePlan(WorkerRole.EMBED, (), (), 0, RoleTarget.CPU),)

    def test_gpu_role_never_asks_the_engine_about_targets(self) -> None:
        planning._refuse_unusable_targets(
            PlacementSpec.from_json('{"embed": {"devices": [0]}}'),
            {WorkerRole.EMBED: "ref"},
            {WorkerRole.EMBED: _NeverAsked()},
        )

    def test_role_with_no_entry_is_left_to_the_entry_check(self) -> None:
        planning._refuse_unusable_targets(
            PlacementSpec.from_json('{"chat": {"target": "cpu"}}'),
            {WorkerRole.EMBED: "ref"},
            {WorkerRole.EMBED: _NeverAsked()},
        )

    def test_mixed_many_card_gate_passes_a_cpu_role_and_stops_a_gpu_role(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _help(monkeypatch, LLAMA_HELP)
        monkeypatch.setattr(type(LLAMA), "binary", lambda _self: tmp_path / "server")
        monkeypatch.setattr(planning, "_roles_without_a_pin", lambda *_a: (WorkerRole.EMBED,))
        cards = [FleetDevice("CUDA", i, f"gpu{i}", 8 * GIB, 8 * GIB) for i in range(2)]
        plan = self._resolve('{"embed": {"target": "cpu"}}', LLAMA, cards)
        assert plan.instances == (InstancePlan(WorkerRole.EMBED, (), (), 0, RoleTarget.CPU),)
        monkeypatch.setattr(planning, "_peak_estimator", lambda *_a: _no_estimate)
        with pytest.raises(PlacementError) as caught:
            self._resolve('{"embed": {"devices": [0]}}', LLAMA, cards)
        assert str(caught.value) == "the engine of embed cannot be told which GPU to use"


class TestNoFallbackToAuto:
    """A spec that names a target is never replaced by the automatic plan at launch."""

    def _saved(self, raw: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        _help(monkeypatch, OLD_HELP)
        monkeypatch.setattr(type(LLAMA), "binary", lambda _self: tmp_path / "server")
        return planning._placement_or_auto(
            PlacementSpec.from_json(raw),
            [],
            {WorkerRole.EMBED: "ref"},
            [],
            engines={WorkerRole.EMBED: LLAMA},
            unified_budget=None,
        )

    def test_refused_cpu_target_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING), pytest.raises(PlacementError, match="CPU build"):
            self._saved('{"embed": {"target": "cpu"}}', monkeypatch, tmp_path)
        assert "automatic placement" not in caplog.text

    def test_old_spec_off_the_hardware_still_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            _, applied = self._saved('{"embed": {"devices": [3]}}', monkeypatch, tmp_path)
        assert applied is False
        assert "using automatic placement" in caplog.text


class TestNoDevicePinIsOnePath:
    def test_cpu_target_pins_none_with_no_refusal_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert planning._no_device_pin(LLAMA, RoleTarget.CPU) == ("none",)
        assert caplog.text == ""

    def test_gpu_target_with_no_probe_pins_nothing(self) -> None:
        assert planning._no_device_pin(LLAMA, RoleTarget.GPU) == ()
