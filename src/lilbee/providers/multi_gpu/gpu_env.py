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

# Subset the Vulkan autodetect applies to. CUDA and HIP/ROCm enumerate
# single-vendor adapter sets (NVIDIA-only, AMD-only); a Vulkan device index
# doesn't translate, so writing the autodetect result to
# ``CUDA_VISIBLE_DEVICES`` on a CUDA wheel + dual-GPU host would hide the only
# NVIDIA card and silently fall back to CPU.
_VULKAN_AUTODETECT_ENV_VARS = ("GGML_VK_VISIBLE_DEVICES",)


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


def apply_gpu_device_env() -> None:
    """Apply ``cfg.gpu_devices`` (or autodetect) to backend visibility env vars.

    Must run before the engine's first ``vkCreateInstance`` (the llama-server
    subprocess inherits this process's environment). Resolution order:

    1. Explicit env var (``GGML_VK_VISIBLE_DEVICES`` etc.) -- always wins,
       including when the user intentionally set it empty.
    2. ``cfg.gpu_devices`` -- the user's lilbee-level pin. Applied to every
       backend (the user is opting in and naming indexes that match their
       wheel's enumeration).
    3. Vulkan autodetection -- pick the highest-ranked Vulkan device (discrete
       > integrated). Applied **only** to ``GGML_VK_VISIBLE_DEVICES``; the
       Vulkan index doesn't translate to CUDA / HIP / ROCm enumeration order,
       so writing it there could mask the only visible NVIDIA / AMD card on a
       single-vendor wheel.

    Steps 2 and 3 only set vars that aren't already in the environment, so step
    1 is preserved.
    """
    from lilbee.core.config import cfg
    from lilbee.providers.multi_gpu.gpu_select import (
        VulkanIcdEnvVar,
        autoselect_best_gpu_index,
        disable_conflicting_vulkan_icds,
    )

    # Suppress known-crashing third-party Vulkan overlay layers on Windows +
    # Linux. Must precede every subsequent vkCreateInstance call (our own probe
    # in autoselect plus llama.cpp's), otherwise the overlay piggy-backs on the
    # first instance and stays resident even if disabled later. setdefault
    # preserves a user-set VK_LOADER_LAYERS_DISABLE, and the loader composes our
    # globs with the user's own ENABLE token per spec.
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        os.environ.setdefault(_VK_LOADER_LAYERS_DISABLE_ENV_VAR, _VK_LOADER_LAYERS_DISABLE_VALUE)

    # Dual-vendor Vulkan crash mitigation runs first and unconditionally: the
    # loader loads every registered ICD at vkCreateInstance, before
    # GGML_VK_VISIBLE_DEVICES is consulted, so device pinning alone cannot
    # prevent a buggy second-vendor ICD from corrupting the heap.
    disable_glob = disable_conflicting_vulkan_icds()
    if disable_glob is not None:
        os.environ.setdefault(VulkanIcdEnvVar.LOADER_DRIVERS_DISABLE, disable_glob)
        log.info("Disabling conflicting Vulkan ICDs: %s", disable_glob)

    if cfg.gpu_devices:
        for name in _GPU_VISIBLE_ENV_VARS:
            os.environ.setdefault(name, cfg.gpu_devices)
        return
    autoselected = autoselect_best_gpu_index()
    if not autoselected:
        return
    for name in _VULKAN_AUTODETECT_ENV_VARS:
        os.environ.setdefault(name, autoselected)
