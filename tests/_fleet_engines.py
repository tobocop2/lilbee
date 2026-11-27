"""A test engine and host that plan real roles across several cards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.devices import DeviceProbe, FleetDevice
from lilbee.providers.fleet.engines.base import EngineAdapter, MemoryEstimate, MemoryKind
from lilbee.providers.roles import ROLE_REGISTRY, EngineBackend, WorkerRole
from tests.conformance.fake_adapter import FakeAdapter

GIB = 1 << 30
ROLE_FIELDS = {
    WorkerRole.CHAT: "chat_model",
    WorkerRole.EMBED: "embedding_model",
    WorkerRole.RERANK: "reranker_model",
    WorkerRole.VISION: "vision_model",
}


def engines_for_all_roles(engine: EngineAdapter) -> dict[WorkerRole, EngineAdapter]:
    """The engine mapping that gives *engine* every role, for tests of one-engine plans."""
    return dict.fromkeys(ROLE_REGISTRY, engine)


def model_file(role: WorkerRole) -> str:
    """The file name a role's model resolves to on the test host."""
    return f"{role.value}.gguf"


def cards_of(gib: tuple[int, ...], backend: str = "CUDA") -> list[FleetDevice]:
    """One discrete card per entry, sized in GiB."""
    return [
        FleetDevice(backend, index, f"gpu{index}", size * GIB, size * GIB)
        for index, size in enumerate(gib)
    ]


@dataclass(frozen=True)
class Role:
    """One role's model on the test host: its footprint in GiB and its replica knob."""

    role: WorkerRole
    gib: float
    replicas: int = 1


class CardEngine(FakeAdapter):
    """An engine that charges each role a stated footprint and may span cards."""

    served_roles = frozenset(ROLE_FIELDS)
    memory_kind = MemoryKind.SIZED

    def __init__(
        self,
        name: str,
        binary: Path,
        roles: tuple[Role, ...],
        devices: list[FleetDevice],
        *,
        spans: bool = False,
        backend: EngineBackend = EngineBackend.CUDA,
        takes_device_names: bool = False,
        cpu_only: bool = False,
        host_gib: float = 0,
    ) -> None:
        self.name = name
        self.takes_device_names = takes_device_names
        self.concatenates_pooling = False
        self._cpu_only = cpu_only
        self._host_bytes = int(host_gib * GIB)
        self._binary = binary
        self._sizes = {model_file(r.role): int(r.gib * GIB) for r in roles}
        self._devices = devices
        self._spans = spans
        self._backend = backend
        binary.write_bytes(name.encode())

    def __hash__(self) -> int:
        return hash(self.name)

    def binary(self) -> Path:
        return self._binary

    def launch_argv(self, spec, request) -> list[str]:
        argv = super().launch_argv(spec, request)
        if request.tensor_split:
            argv += ["--tensor-split", ",".join(str(part) for part in request.tensor_split)]
        if self.takes_device_names and request.device_names:
            argv += ["--device", ",".join(request.device_names)]
        return argv

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(
            devices=list(self._devices), output="", spoke_protocol=False, backend=self._backend
        )

    def estimate_memory(self, request) -> MemoryEstimate:
        size = self._sizes[request.model_path.name]
        if self._cpu_only:
            return MemoryEstimate(
                kind=self.memory_kind, device_bytes=0, host_bytes=size, shared_bytes=size
            )
        ratio = request.tensor_split
        if not ratio:
            per_device = (size,)
        elif self._spans:
            per_device = tuple(size * part // sum(ratio) for part in ratio)
        else:
            per_device = ()
        return MemoryEstimate(
            kind=self.memory_kind,
            device_bytes=size,
            host_bytes=self._host_bytes,
            shared_bytes=size,
            per_device=per_device,
            per_device_shared=per_device,
        )

    def model_meta(self, path: Path):
        return {"architecture": "fakearch", "context_length": "4096"}

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def runtime_env(self):
        return {}

    def target_env(self, target, cards):
        return {}

    def swap_extras(self, launch):
        return {}


def prepare_host(monkeypatch, tmp_path: Path, roles: tuple[Role, ...], total: int) -> None:
    """Configure *roles*, their model files and *total* bytes of system memory."""
    for role in roles:
        (tmp_path / model_file(role.role)).write_bytes(b"x" * 4096)
    monkeypatch.setattr(
        engine_params,
        "resolve_model_path",
        lambda ref: tmp_path / Path(ref).name,
    )
    planning.clear_read_device_cache()
    for module in (planning, planning.model_cache):
        monkeypatch.setattr(module, "total_system_memory", lambda: total)
        monkeypatch.setattr(planning.model_cache, "free_system_memory", lambda: total)
    monkeypatch.setattr(cfg, "system_memory_reserve_gb", 0)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 1.0)
    monkeypatch.setattr(cfg, "usable_vram_fraction", 0.9)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "placement", None)
    monkeypatch.setattr(cfg, "embed_replicas", 1)
    monkeypatch.setattr(cfg, "vision_replicas", 1)
    for field in ROLE_FIELDS.values():
        monkeypatch.setattr(cfg, field, "")
    for role in roles:
        monkeypatch.setattr(cfg, ROLE_FIELDS[role.role], f"org/repo/{model_file(role.role)}")
        if role.role is WorkerRole.EMBED:
            monkeypatch.setattr(cfg, "embed_replicas", role.replicas)
    monkeypatch.setattr(planning, "vision_role_wanted", lambda _ref: True)
    monkeypatch.setattr(planning, "vision_mmproj", lambda _ref: tmp_path / "mmproj.gguf")
    monkeypatch.setattr(planning, "host_lacks_nvlink", lambda: False)


class SizedEngine(FakeAdapter):
    """An engine that charges a fixed footprint for every model it loads."""

    served_roles = frozenset({WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION})

    def __init__(self, name: str, binary: Path, kind: MemoryKind, footprint: int, device) -> None:
        self.name = name
        self.memory_kind = kind
        self._binary = binary
        self._footprint = footprint
        self._device = device

    def __hash__(self) -> int:
        return hash(self.name)

    def binary(self) -> Path:
        return self._binary

    def discover_devices(self, binary: Path) -> DeviceProbe:
        return DeviceProbe(
            devices=list(self._device), output="", spoke_protocol=False, backend=EngineBackend.CUDA
        )

    def estimate_memory(self, request) -> MemoryEstimate:
        return MemoryEstimate(
            kind=self.memory_kind,
            device_bytes=self._footprint,
            host_bytes=0,
            shared_bytes=self._footprint,
            per_device=(self._footprint,),
            per_device_shared=(self._footprint,),
        )

    def model_meta(self, path: Path):
        return {"architecture": "fakearch", "context_length": "4096"}

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def runtime_env(self):
        return {}

    def target_env(self, target, cards):
        return {}

    def swap_extras(self, launch):
        return {}


def one_machine(
    monkeypatch, tmp_path: Path, *, unified: bool, total: int, cards: int = 1
) -> list[FleetDevice]:
    """Patch one machine with *cards* devices of *total* bytes each."""
    model = tmp_path / "model.bin"
    model.write_bytes(b"x" * 4096)
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref: model)
    planning.clear_read_device_cache()
    for module in (planning, planning.model_cache):
        monkeypatch.setattr(module, "total_system_memory", lambda: total)
        monkeypatch.setattr(planning.model_cache, "free_system_memory", lambda: total)
    monkeypatch.setattr(cfg, "system_memory_reserve_gb", 0)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 1.0)
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "chat_model", "org/repo/chat.gguf")
    monkeypatch.setattr(cfg, "embedding_model", "org/repo/embed.gguf")
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")
    monkeypatch.setattr(cfg, "placement", None)
    return [
        FleetDevice("CUDA", index, f"gpu{index}", total, total, unified=unified)
        for index in range(cards)
    ]


def engine_pair(
    tmp_path: Path, device, chat_gib: int, other_gib: int, *, other_kind=MemoryKind.SIZED
):
    """Two engines with their own binaries: one for chat, one for the other role."""
    first = tmp_path / "engine-a-server"
    second = tmp_path / "engine-b-server"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    chat = SizedEngine("engine-a", first, MemoryKind.SIZED, chat_gib * GIB, device)
    other = SizedEngine("engine-b", second, other_kind, other_gib * GIB, device)
    return chat, other
