"""The mistral.rs adapter; the only module that names mistral.rs's tools."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from lilbee.catalog.hf_client import HfToken
from lilbee.catalog.types import ModelFormat
from lilbee.providers.base import ProviderError
from lilbee.providers.fleet.adapters import ROLE_SPECS, RoleServerSpec
from lilbee.providers.fleet.devices import DeviceProbe, visible_nvidia_cards
from lilbee.providers.fleet.engine_manifest import pack_digest, pack_platforms
from lilbee.providers.fleet.engines.base import (
    PORT_SLOT,
    EngineName,
    LaunchRequest,
    RequestProfile,
    ServerProcess,
    SplitRefusedError,
)
from lilbee.providers.fleet.engines.host_gate import PASS, PLATFORM_UNREAD, HostGate
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.model_memory import estimate_memory, fixed_weights_estimate
from lilbee.providers.folder_embed_width import read_folder_config
from lilbee.providers.gguf_meta import (
    is_hf_cache_blob,
    read_gguf_metadata,
    snapshot_entry_of_blob,
)
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

_MIN_SPLIT_CARDS = 2
# With two or more visible cards the engine starts tensor parallelism unless this is "1".
_NO_COLLECTIVE_ENV = "MISTRALRS_NO_NCCL"
# A split uses peer-to-peer only over NVLink; a PCIe pair in a VM can stall without this.
_P2P_LEVEL_ENV = "NCCL_P2P_LEVEL"
_P2P_NVLINK_ONLY = "NVL"
# A user who sets either variable keeps their choice.
_P2P_USER_ENV = (_P2P_LEVEL_ENV, "NCCL_P2P_DISABLE")
_LAYERS_KEY = "num_hidden_layers"
_HEADS_KEY = "num_attention_heads"
_KV_HEADS_KEY = "num_key_value_heads"
_ARCHITECTURES_KEY = "architectures"
_WINDOW_KEY = "max_position_embeddings"
# The multimodal loaders of mistral.rs v0.9.5 that cap their window to ``--max-model-len``
# (their ``runtime_config``); the trait default of every other loader refuses the flag.
_CAPPED_VISION_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen4ExpForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Gemma4UnifiedForConditionalGeneration",
        "Gemma4UnifiedForCausalLM",
        "MuseGlimmerForConditionalGeneration",
    }
)
_TEXT_CONFIG_KEY = "text_config"
_GGUF_LAYERS_KEY = "block_count"
_GGUF_HEADS_KEY = "head_count"
_GGUF_KV_HEADS_KEY = "head_count_kv"
_SPLIT_FIX = f"Use the {EngineName.LLAMA.value} engine for this role, or a smaller model."
_VERSION = "0.9.5"
_TOOL = "mistralrs"
_HOST = "127.0.0.1"
_LICENCES = ["MIT"]
# The expression a CUDA pack declares: it bundles NVIDIA runtime libraries.
_CUDA_LICENCES = ["MIT AND LicenseRef-NVIDIA-CUDA-EULA AND BSD-3-Clause"]
_CPU = "cpu"
# The one id the engine serves a model under; it answers 400 to any other.
_SERVED_NAME = "default"
_GGUF_SUFFIX = ".gguf"
# The model subcommand each role names; the device flags go after it.
_EMBEDDING = "embedding"
_MULTIMODAL = "multimodal"
_SUBCOMMAND = {
    WorkerRole.EMBED: _EMBEDDING,
    WorkerRole.VISION: _MULTIMODAL,
}
_TEXT = "text"
_METAL = "metal"
_NO_CUDA_PACK = "The mistral.rs CUDA pack is not offered, so the cpu pack runs."
_SUPPORTED_HOSTS = frozenset(
    {
        (HostOS.LINUX, HostArch.X86_64),
        (HostOS.MACOS, HostArch.ARM64),
        (HostOS.WINDOWS, HostArch.X86_64),
    }
)

_PROFILE = RequestProfile(
    embed_fields={},
    stream_fields={},
    tokenize_path=None,
    detokenize_path=None,
    apply_template_path="/apply-template",
    label="mistral.rs",
    context_overflow=re.compile("Prompt sequence length is greater than", re.IGNORECASE),
    batch_overflow=re.compile(r"(?!)"),
    unreadable_embedding_hint="mistral.rs answers embeddings as lists of floats.",
    reasoning_field="reasoning_content",
)


@dataclass(frozen=True)
class BuildPick:
    """The archive for one host: its backend name, or ``None`` with the reason it has none."""

    backend: str | None
    reason: str


def _linux_pick(host: HostCaps) -> BuildPick:
    """The CPU archive, with the reason when an NVIDIA GPU could have asked for a CUDA one."""
    if not host.nvidia_known:
        return BuildPick(_CPU, f"{host.nvidia_reason}, so the CPU archive runs.")
    if not host.gpus:
        return BuildPick(_CPU, "This host has no NVIDIA GPU, so the CPU archive runs.")
    return BuildPick(_CPU, _NO_CUDA_PACK)


def _positive_int(value: object) -> int | None:
    """*value* as a whole number above zero, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, int | str):
        return None
    try:
        count = int(value)
    except ValueError:
        return None
    return count if count > 0 else None


def _config_sections(path: Path) -> list[Mapping[str, Any]]:
    """The top level and the text section of a model folder's config, as far as they are objects."""
    config = read_folder_config(path)
    return [s for s in (config, config.get(_TEXT_CONFIG_KEY)) if isinstance(s, dict)]


def _layer_count(path: Path) -> int | None:
    """The layer count a GGUF file or a model folder declares, or ``None``."""
    if path.suffix.lower() == _GGUF_SUFFIX:
        meta = read_gguf_metadata(path)
        return _positive_int(meta.get(_GGUF_LAYERS_KEY)) if meta else None
    for section in _config_sections(path):
        count = _positive_int(section.get(_LAYERS_KEY))
        if count is not None:
            return count
    return None


def _vision_context(path: Path, planned: int) -> int:
    """The context flag of a vision folder, or 0 for none.

    The loaders that take the flag refuse a figure above the window their config
    declares, so the flag is the smaller of the planned context and that window. A
    config that declares none gets no flag.
    """
    sections = _config_sections(path)
    if _architecture(sections) not in _CAPPED_VISION_ARCHITECTURES:
        return 0
    window = _positive_int(sections[-1].get(_WINDOW_KEY))
    return min(planned, window) if window else 0


def _architecture(sections: list[Mapping[str, Any]]) -> str | None:
    """The one architecture the top level of a config names, or ``None``."""
    names = sections[0].get(_ARCHITECTURES_KEY) if sections else None
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], str):
        return names[0]
    return None


def _heads_in(section: Mapping[str, Any], heads_key: str, kv_key: str) -> tuple[int, int] | None:
    """Query and key-value head counts of *section*; an absent key-value count equals the first."""
    heads = _positive_int(section.get(heads_key))
    if heads is None:
        return None
    kv_heads = section.get(kv_key)
    return heads, heads if kv_heads is None else (_positive_int(kv_heads) or 0)


def _head_counts(path: Path) -> tuple[int, int] | None:
    """Query and key-value head counts a GGUF file or a model folder declares, or ``None``."""
    if path.suffix.lower() == _GGUF_SUFFIX:
        meta = read_gguf_metadata(path)
        return _heads_in(meta, _GGUF_HEADS_KEY, _GGUF_KV_HEADS_KEY) if meta else None
    for section in _config_sections(path):
        counts = _heads_in(section, _HEADS_KEY, _KV_HEADS_KEY)
        if counts is not None:
            return counts
    return None


def _heads_divide(heads: int, kv_heads: int, cards: int) -> bool:
    """Whether the heads shard over *cards*: queries divide, key-value heads divide or repeat."""
    if heads % cards or kv_heads <= 0:
        return False
    return kv_heads % cards == 0 if cards <= kv_heads else cards % kv_heads == 0


def _apportion(layers: int, ratio: tuple[int, ...]) -> tuple[int, ...] | None:
    """Layers per card by largest remainder; ``None`` when a card gets none."""
    total = sum(ratio)
    if total <= 0:
        return None
    shares = [layers * part // total for part in ratio]
    by_remainder = sorted(range(len(ratio)), key=lambda i: (-(layers * ratio[i] % total), i))
    for i in by_remainder[: layers - sum(shares)]:
        shares[i] += 1
    return tuple(shares) if min(shares) > 0 else None


class MistralAdapter:
    """mistral.rs behind the engine contract; it serves chat, embed and vision, never rerank."""

    name = EngineName.MISTRAL
    formats = frozenset({ModelFormat.GGUF, ModelFormat.SAFETENSORS})
    architectures: frozenset[str] = frozenset()
    served_roles = frozenset({WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION})
    refused_role_formats: Mapping[WorkerRole, frozenset[str]] = {
        WorkerRole.EMBED: frozenset({ModelFormat.GGUF})
    }
    memory_kind = MemoryKind.FIXED_RESERVE
    equal_split_only = True
    takes_device_names = False
    concatenates_pooling = True
    bundled = False
    server_process = ServerProcess(name=_TOOL)

    def supports_architecture(self, architecture: str) -> bool:
        return True

    def host_gate(self, host: HostCaps) -> HostGate:
        if host.os is None or host.arch is None:
            return PLATFORM_UNREAD
        if (host.os, host.arch) in _SUPPORTED_HOSTS:
            reason = self._archive(host, host.os).reason
            return HostGate(True, reason) if reason else PASS
        return HostGate(False, f"mistral.rs is not offered on {host.os.value} {host.arch.value}.")

    def pick_build(self, host: HostCaps) -> BuildPick:
        """The archive that runs on *host*; ``backend`` is ``None`` when the gate refuses it."""
        gate = self.host_gate(host)
        if not gate.ok or host.os is None:
            return BuildPick(None, gate.reason)
        return self._archive(host, host.os)

    @staticmethod
    def _archive(host: HostCaps, os: HostOS) -> BuildPick:
        if os is HostOS.MACOS:
            return BuildPick(_METAL, "")
        if os is HostOS.WINDOWS:
            return BuildPick(_CPU, "The release has no CUDA archive for Windows.")
        return _linux_pick(host)

    def pin_record(self) -> Mapping[str, Any]:
        from lilbee.catalog.runnable import session_host

        host = session_host()
        backend = self.pick_build(host).backend or _CPU
        platforms = pack_platforms(self.name.value, _VERSION, host, backend)
        return {
            "engine": self.name.value,
            "version": _VERSION,
            "backend": backend,
            "sha256": pack_digest(self.name.value, _VERSION, host, backend),
            "tool": _TOOL,
            "launch": [_TOOL, "serve"],
            "platforms": platforms,
            "memory": {"sizing": None, "rule": MemoryKind.FIXED_RESERVE.value},
            "licenses": _LICENCES if backend == _CPU else _CUDA_LICENCES,
        }

    def binary(self) -> Path:
        from lilbee.catalog.runnable import session_host
        from lilbee.providers.fleet.engines.choice import install_advice, store_tool

        tool = store_tool(self.pin_record(), _TOOL, session_host())
        if tool is None:
            raise ProviderError(f"The mistral.rs engine is not installed. {install_advice(self)}")
        return tool

    def identity(self, binary: Path) -> str:
        return f"{self.name.value} {_VERSION} {binary}"

    def role_spec(
        self, role: WorkerRole, rerank_mode: RerankMode | None, meta: dict[str, str] | None
    ) -> RoleServerSpec:
        if role not in self.served_roles:
            raise ProviderError(f"Engine {self.name.value} does not serve {role}.")
        return RoleServerSpec(role, ROLE_SPECS[role].endpoint_path, (), server_capable=True)

    def launch_argv(self, spec: RoleServerSpec, request: LaunchRequest) -> list[str]:
        """The serve command; an equal split over cards adds no flag.

        Chat passes the planned context as ``--max-model-len``, the figure the memory
        estimate prices. An embedding model takes none: its context is its tokenizer
        limit, and no cache grows with it. Vision takes it only for the architectures in
        ``_CAPPED_VISION_ARCHITECTURES``: any other multimodal loader refuses the flag and
        the launch fails. The estimate prices the planned context in every case.

        A cpu target gets ``--cpu`` on any build. Otherwise the backend of the build
        decides the device flags: a CPU build gets ``--cpu``, and
        any GPU build turns paged attention off, because its cache takes the whole card
        and the first chat then answers 503. The port flag
        belongs to ``serve``, so it sits before the model subcommand. The device flags
        belong to the subcommand: before it the engine ignores ``--cpu`` and takes the
        whole card (20536 MiB measured), so every role names its subcommand. A GGUF file
        goes by ``-f`` with its folder as ``-m``. The cache stores the file as a blob
        that has no name, so the launch names the snapshot file that points at it.
        """
        argv = [str(request.binary), "serve", "--host", _HOST, "--port", PORT_SLOT]
        subcommand = _SUBCOMMAND.get(spec.role, _TEXT)
        argv.append(subcommand)
        argv += self._device_flags(request.target)
        self._check_split(request)
        path = snapshot_entry_of_blob(request.model_path)
        context = request.ctx_per_slot
        if subcommand == _MULTIMODAL:
            context = _vision_context(path, context)
        elif subcommand != _TEXT:
            context = 0
        if context:
            argv += ["--max-model-len", str(context)]
        if path.suffix.lower() == _GGUF_SUFFIX:
            return [*argv, "-m", str(path.parent), "-f", path.name]
        return [*argv, "-m", str(path)]

    def _device_flags(self, target: RoleTarget) -> list[str]:
        from lilbee.catalog.runnable import session_host

        if target is RoleTarget.CPU or self.pick_build(session_host()).backend in (None, _CPU):
            return ["--cpu"]
        return ["--paged-attn", "off"]

    def _check_split(self, request: LaunchRequest) -> None:
        """Refuse a launch split over several cards that the engine cannot run."""
        if request.target is not RoleTarget.CPU:
            self.check_split(request.model_path, request.tensor_split)

    def check_split(self, model_path: Path, ratio: tuple[int, ...]) -> None:
        """Refuse a split over several cards unless the engine can run it as equal shards."""
        if len(ratio) < _MIN_SPLIT_CARDS:
            return
        from lilbee.catalog.runnable import session_host

        path = snapshot_entry_of_blob(model_path)
        name = path.name
        cards = len(ratio)
        backend = self.pick_build(session_host()).backend
        if backend in (None, _CPU, _METAL):
            label = "Metal" if backend == _METAL else "CPU"
            raise SplitRefusedError(
                f"mistral.rs cannot split {name} across {cards} cards on the {label} build. "
                f"{_SPLIT_FIX}"
            )
        if len(set(ratio)) != 1:
            raise SplitRefusedError(
                f"mistral.rs splits {name} only evenly across cards. "
                f"Use the {EngineName.LLAMA.value} engine for an uneven split."
            )
        if is_hf_cache_blob(path):
            raise SplitRefusedError(
                f"mistral.rs cannot split this model across {cards} cards: "
                "its model file is incomplete or damaged. "
                "Run 'lilbee model list', remove that model with 'lilbee model rm', "
                "then pull it again."
            )
        heads = _head_counts(path)
        if heads is None or not _heads_divide(*heads, cards):
            raise SplitRefusedError(
                f"mistral.rs cannot split {name} across {cards} cards: "
                f"the attention heads of the model do not divide over {cards} cards, "
                f"or the model does not list its heads. {_SPLIT_FIX}"
            )

    def device_targets(self, binary: Path) -> frozenset[RoleTarget]:
        from lilbee.catalog.runnable import session_host

        if self.pick_build(session_host()).backend in (None, _CPU):
            return frozenset({RoleTarget.CPU})
        return frozenset({RoleTarget.CPU, RoleTarget.GPU})

    def discover_devices(self, binary: Path) -> DeviceProbe:
        from lilbee.catalog.runnable import session_host

        if self.pick_build(session_host()).backend in (None, _CPU):
            return DeviceProbe(devices=[], output="", has_listing=False)
        return DeviceProbe(devices=visible_nvidia_cards(), output="", has_listing=False)

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        whole = self._whole_estimate(request)
        if len(request.tensor_split) < _MIN_SPLIT_CARDS:
            return whole
        layers = _layer_count(snapshot_entry_of_blob(request.model_path))
        if layers is None:
            return whole
        shares = _apportion(layers, request.tensor_split)
        if shares is None:
            return whole
        per_device = tuple(whole.device_bytes * count // layers for count in shares)
        return replace(whole, per_device=per_device)

    @staticmethod
    def _whole_estimate(request: MemoryRequest) -> MemoryEstimate:
        if request.model_path.is_dir():
            return estimate_memory(ModelFormat.SAFETENSORS, request)
        try:
            weights = request.model_path.stat().st_size
        except OSError:
            weights = None
        return fixed_weights_estimate(weights)

    def check_readback(self, binary: Path) -> bool:
        return False

    def supports_fit_target(self, binary: Path) -> bool:
        return False

    def supports_shared_kv(self, binary: Path) -> bool:
        return False

    def model_meta(self, path: Path) -> dict[str, str] | None:
        return None

    def assert_loadable(self, repo: str, filename: str, token: HfToken) -> None:
        return None

    def runtime_env(self) -> Mapping[str, str]:
        return {}

    def target_env(self, target: RoleTarget, cards: int) -> Mapping[str, str]:
        """Tensor parallelism needs the collective library; any other launch turns it off."""
        if target is not RoleTarget.CPU and cards >= _MIN_SPLIT_CARDS:
            if any(name in os.environ for name in _P2P_USER_ENV):
                return {}
            return {_P2P_LEVEL_ENV: _P2P_NVLINK_ONLY}
        return {_NO_COLLECTIVE_ENV: "1"}

    def swap_extras(self, launch: InstanceLaunch) -> Mapping[str, Any]:
        return {"useModelName": _SERVED_NAME}

    def request_profile(self, role: WorkerRole) -> RequestProfile:
        return _PROFILE


MISTRAL = MistralAdapter()
ADAPTER = MISTRAL
