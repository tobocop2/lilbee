"""Apply GPU-visibility and Vulkan-loader environment before the engine starts.

Binding-free: sets the backend visible-device env vars (from ``cfg.gpu_devices``
or Vulkan autodetect) and the dual-vendor Vulkan crash mitigations. These are
process-wide ``setdefault`` writes, so a child llama-server inherits them.
"""

from __future__ import annotations

import logging
import os
import sys

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


def _apply_vulkan_loader_safety() -> None:
    """Disable known-crashing overlay layers and conflicting dual-vendor ICDs.

    Must precede every ``vkCreateInstance``: the loader loads every ICD and layer
    at instance creation, before ``GGML_VK_VISIBLE_DEVICES`` is consulted, so
    device pinning alone cannot stop a buggy second-vendor ICD from corrupting the
    heap. ``setdefault`` preserves any user-set value.
    """
    from lilbee.providers.fleet.gpu_select import (
        VulkanIcdEnvVar,
        disable_conflicting_vulkan_icds,
    )

    if sys.platform == "win32" or sys.platform.startswith("linux"):
        os.environ.setdefault(_VK_LOADER_LAYERS_DISABLE_ENV_VAR, _VK_LOADER_LAYERS_DISABLE_VALUE)
    disable_glob = disable_conflicting_vulkan_icds()
    if disable_glob is not None:
        os.environ.setdefault(VulkanIcdEnvVar.LOADER_DRIVERS_DISABLE, disable_glob)
        log.info("Disabling conflicting Vulkan ICDs: %s", disable_glob)


def _apply_gpu_devices_pin() -> bool:
    """Apply the user's ``cfg.gpu_devices`` pin to every backend's visible-devices var.

    Returns ``True`` when a pin was applied (so the caller can skip autodetect).
    The pin goes to all four backend vars because the user is naming indexes
    that match their own wheel's enumeration. ``setdefault`` keeps an explicit
    env var the user already set.
    """
    from lilbee.core.config import cfg

    if not cfg.gpu_devices:
        return False
    for name in _GPU_VISIBLE_ENV_VARS:
        os.environ.setdefault(name, cfg.gpu_devices)
    return True


def apply_fleet_gpu_env() -> None:
    """Fleet engine bootstrap: loader safety plus the ``cfg.gpu_devices`` pin only.

    The single-device Vulkan autodetect is skipped: the fleet selects devices via
    its own placement, and autodetect would pin ``GGML_VK_VISIBLE_DEVICES`` to one
    adapter before ``probe_devices`` runs and hide every other GPU. A
    ``cfg.gpu_devices`` pin is still honored (the probe inherits this environment).
    """
    _apply_vulkan_loader_safety()
    _apply_gpu_devices_pin()
