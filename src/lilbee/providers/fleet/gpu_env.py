"""The environment a fleet child runs in: GPU visibility and Vulkan-loader safety.

Binding-free: composes the backend visible-device env vars (from
``cfg.gpu_devices``) and the dual-vendor Vulkan crash mitigations over a copy of
this process's environment. Nothing here writes ``os.environ``: ``setenv``
reallocates the C ``environ`` array, and a native thread reading it at that moment
(lancedb's runtime) can fault. A caller passes the returned mapping as ``env=`` at
spawn, and a reader that needs the effective value reads it from that mapping.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import MutableMapping

from lilbee.runtime.shard_process import shard_process

log = logging.getLogger(__name__)

# Backend env vars set from ``cfg.gpu_devices``. Vulkan, CUDA, and ROCm each read
# their own; a user-set ``cfg.gpu_devices`` is applied to all four because the
# user is specifying their own indexes and opting in to all wheel flavors.
_GPU_VISIBLE_ENV_VARS = (
    "GGML_VK_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)
# The AMD pair is deliberately absent: ROCr filters before HIP re-indexes within
# the survivors, so a pin may only ever be written to one of them. Which one is
# devices.amd_visible_var's decision.
_NON_AMD_VISIBLE_ENV_VARS = ("GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
_CUDA_VISIBLE_VAR = "CUDA_VISIBLE_DEVICES"
# What the NVIDIA container runtime sets to say which GPUs it gave this
# container. Its own words for "none" are below.
_NVIDIA_RUNTIME_VISIBLE_VAR = "NVIDIA_VISIBLE_DEVICES"
_NVIDIA_RUNTIME_NO_GPU_VALUES = frozenset({"void", "none"})

_VK_LOADER_LAYERS_DISABLE_ENV_VAR = "VK_LOADER_LAYERS_DISABLE"

# Layers with documented crashes against multi-VkDevice apps:
#   https://github.com/ggml-org/llama.cpp/issues/18109 (RTSS / OBS / HudSight)
#   https://github.com/ValveSoftware/steam-for-linux/issues/9120 (Steam overlay)
#   https://alegruz.github.io/graphics/2025/03/22/galaxyoverlayvklayer-issue.html (Galaxy)
# Vendor-dispatch layers (NV_optimus, AMD_switchable_graphics, MESA_device_select)
# and user-opt-in overlays (MangoHud) are intentionally absent so GPU routing
# stays identical to what every other Vulkan app on the host sees.
_VK_LOADER_LAYERS_DISABLE_GLOBS: tuple[str, ...] = (
    "VK_LAYER_VALVE_steam_overlay*",
    "VK_LAYER_VALVE_steam_fossilize*",
    "VK_LAYER_RTSS*",
    "VK_LAYER_OBS_HOOK*",
    "VK_LAYER_HudSight*",
    "GalaxyOverlayVkLayer*",
    "VK_LAYER_GalaxyOverlay*",
    "VK_LAYER_DISCORD_overlay*",
    "VK_LAYER_EOS_Overlay*",
    "VK_LAYER_RESHADE*",
    "VK_LAYER_VKBASALT*",
)
_VK_LOADER_LAYERS_DISABLE_VALUE = ",".join(_VK_LOADER_LAYERS_DISABLE_GLOBS)


def _apply_vulkan_loader_safety(env: MutableMapping[str, str]) -> None:
    """Disable known-crashing overlay layers and conflicting dual-vendor ICDs in *env*.

    Must reach every child that calls ``vkCreateInstance``: the loader loads every
    ICD and layer at instance creation, before ``GGML_VK_VISIBLE_DEVICES`` is
    consulted, so device pinning alone cannot stop a buggy second-vendor ICD from
    corrupting the heap. A value already in *env* is kept.
    """
    from lilbee.providers.fleet.gpu_select import (
        VulkanIcdEnvVar,
        disable_conflicting_vulkan_icds,
    )

    if sys.platform == "win32" or sys.platform.startswith("linux"):
        env.setdefault(_VK_LOADER_LAYERS_DISABLE_ENV_VAR, _VK_LOADER_LAYERS_DISABLE_VALUE)
    disable_glob = disable_conflicting_vulkan_icds()
    if disable_glob is not None:
        env.setdefault(VulkanIcdEnvVar.LOADER_DRIVERS_DISABLE, disable_glob)
        log.info("Disabling conflicting Vulkan ICDs: %s", disable_glob)


def _apply_gpu_devices_pin(env: MutableMapping[str, str]) -> bool:
    """Put the user's ``cfg.gpu_devices`` pin in *env* for every backend's visible-devices var.

    Returns ``True`` when a pin was applied (so the caller can skip autodetect).
    The pin goes to all four backend vars because the user is naming indexes
    that match their own wheel's enumeration. A non-empty value *env* already
    carries is kept, since that is an equally explicit instruction arriving
    closer to the process.

    An empty one is replaced. It carries no index to respect, and the pin is the
    more specific statement of the two: somebody wrote it into this lilbee's own
    configuration, where an empty mask is usually inherited from whatever
    launched the process.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.fleet.devices import amd_visible_var

    if not cfg.gpu_devices:
        return False
    for name in (*_NON_AMD_VISIBLE_ENV_VARS, amd_visible_var(env)):
        if env.get(name, "").strip():
            continue
        env[name] = cfg.gpu_devices
    return True


def shard_visible_devices(device: int) -> dict[str, str]:
    """The backend visible-device vars that pin a process to one card.

    Composed against whatever mask this process already carries, so *device* names
    the card at that position in lilbee's own enumeration rather than the host's.
    """
    from lilbee.providers.fleet.devices import amd_visible_var

    env = gpu_environ()
    return {
        name: _mask_entry(env.get(name, ""), device)
        for name in (*_NON_AMD_VISIBLE_ENV_VARS, amd_visible_var(env))
    }


def _mask_entry(mask: str, device: int) -> str:
    """The *device*-th entry of a visible-devices *mask*, or its index when unmasked."""
    entries = [entry.strip() for entry in mask.split(",") if entry.strip()]
    if not entries:
        return str(device)
    return entries[device % len(entries)]


def _clear_empty_visible_device_vars(env: MutableMapping[str, str]) -> None:
    """Drop an empty ``CUDA_VISIBLE_DEVICES`` from *env* when the container runtime contradicts it.

    An empty visibility variable is not a mistake to be corrected. It is the
    documented way to say "no devices", it is what SLURM and Kubernetes export on
    an allocation without a GPU, and it is what a user writes to force CPU. These
    variables are read-only filters; deleting one overrides a decision somebody
    made on purpose and can hand backend selection to a vendor that was fenced
    off deliberately.

    The one exception is a genuine contradiction: the NVIDIA container runtime
    exposes a card through ``NVIDIA_VISIBLE_DEVICES`` while leaving
    ``CUDA_VISIBLE_DEVICES`` empty, so the two disagree and the runtime's own
    statement is the newer one. That marker speaks only for NVIDIA, so only the
    CUDA variable is touched; it says nothing about an AMD or Vulkan opt-out, and
    those are always left alone.
    """
    if not _container_runtime_exposes_a_gpu(env):
        return
    if _CUDA_VISIBLE_VAR in env and not env[_CUDA_VISIBLE_VAR].strip():
        del env[_CUDA_VISIBLE_VAR]
        log.debug(
            "%s was empty while %s exposes a GPU; clearing the empty mask so the engine "
            "can see the card the container runtime provided.",
            _CUDA_VISIBLE_VAR,
            _NVIDIA_RUNTIME_VISIBLE_VAR,
        )


def _container_runtime_exposes_a_gpu(env: MutableMapping[str, str]) -> bool:
    """Whether the NVIDIA container runtime says this container was given a GPU.

    ``void`` and ``none`` are its own words for "no GPU", so they confirm the
    empty mask rather than contradict it.
    """
    value = env.get(_NVIDIA_RUNTIME_VISIBLE_VAR, "").strip().casefold()
    return bool(value) and value not in _NVIDIA_RUNTIME_NO_GPU_VALUES


def gpu_environ() -> dict[str, str]:
    """This process's environment with the ``cfg.gpu_devices`` pin and the empty-mask fix applied.

    A copy: the process environment is not touched. Cheap and free of disk reads,
    so a sampler may call it every second. Anything that reads a visible-devices
    variable to learn which cards the engine sees reads it here, not from
    ``os.environ``, because the pin lives only in this mapping.

    Nothing here chooses a device. The fleet selects through its own placement,
    and anything pinning ``GGML_VK_VISIBLE_DEVICES`` before ``probe_devices``
    runs would hide every other GPU from it and switch off ggml's own device
    filtering besides. A ``cfg.gpu_devices`` pin is still honored, since there
    the user is naming their own indexes. An empty backend visible-devices var from
    the orchestrator is cleared first so it does not hide a present GPU, and so a
    pin can replace it rather than be blocked.
    """
    env = dict(os.environ)
    env.update(shard_process.visible_devices)
    _clear_empty_visible_device_vars(env)
    _apply_gpu_devices_pin(env)
    return env


def engine_environ() -> dict[str, str]:
    """The environment for a child that loads the engine or the Vulkan loader.

    :func:`gpu_environ` plus the Vulkan loader safety. The loader reads its
    variables when the child creates its first instance, so this must be the
    ``env=`` of every such spawn. The ICD discovery reads the host's device tree,
    which is why the sampling readers use :func:`gpu_environ` instead.
    """
    env = gpu_environ()
    _apply_vulkan_loader_safety(env)
    return env
