"""An empty visibility variable is an instruction, not an accident."""

from __future__ import annotations

import logging

import pytest

from lilbee.providers.fleet import gpu_env


@pytest.fixture(autouse=True)
def _no_pin(monkeypatch):
    """No cfg.gpu_devices pin, and an NVIDIA card physically present."""
    from lilbee.core.config import cfg

    monkeypatch.setattr(cfg, "gpu_devices", "")
    monkeypatch.setattr("lilbee.providers.model_cache.has_nvidia_gpu", lambda: True)
    for var in gpu_env._GPU_VISIBLE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)


class TestADeliberateCpuOptOutSurvives:
    """SLURM and Kubernetes export an empty CUDA_VISIBLE_DEVICES on a no-GPU
    allocation, and it is the documented idiom for "no devices". Deleting it
    un-hides hardware the caller was told not to have."""

    def test_an_empty_mask_with_no_orchestrator_marker_is_kept(self, monkeypatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        gpu_env._clear_empty_visible_device_vars()
        assert gpu_env.os.environ["CUDA_VISIBLE_DEVICES"] == ""

    def test_an_empty_mask_is_cleared_when_the_runtime_says_a_gpu_is_exposed(
        self, monkeypatch, caplog
    ) -> None:
        # The case this was written for: the NVIDIA container runtime exposes the
        # card through its own variable while leaving CUDA_VISIBLE_DEVICES empty.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
        with caplog.at_level(logging.INFO, logger="lilbee.providers.fleet.gpu_env"):
            gpu_env._clear_empty_visible_device_vars()
        assert "CUDA_VISIBLE_DEVICES" not in gpu_env.os.environ

    def test_a_void_runtime_marker_is_not_a_gpu(self, monkeypatch) -> None:
        # "void" and "none" are the container runtime's own words for no GPU.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "void")
        gpu_env._clear_empty_visible_device_vars()
        assert gpu_env.os.environ["CUDA_VISIBLE_DEVICES"] == ""

    @pytest.mark.parametrize(
        "var",
        ["GGML_VK_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"],
    )
    def test_another_vendors_opt_out_is_never_touched(self, monkeypatch, var: str) -> None:
        # The NVIDIA container runtime says nothing about AMD or Vulkan, so its
        # marker cannot justify un-hiding those.
        monkeypatch.setenv(var, "")
        monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
        gpu_env._clear_empty_visible_device_vars()
        assert gpu_env.os.environ[var] == ""


class TestTheConfiguredPinAndTheInheritedMask:
    """Which of two explicit instructions wins, and why."""

    def test_a_pin_replaces_an_empty_mask(self, monkeypatch) -> None:
        # An empty mask carries no index to respect, and the pin was written into
        # this lilbee's own config rather than inherited from the launcher.
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "gpu_devices", "0")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert gpu_env._apply_gpu_devices_pin() is True
        assert gpu_env.os.environ["CUDA_VISIBLE_DEVICES"] == "0"

    def test_a_pin_yields_to_a_mask_that_names_devices(self, monkeypatch) -> None:
        # Equally explicit, and arriving closer to the process.
        from lilbee.core.config import cfg

        monkeypatch.setattr(cfg, "gpu_devices", "0")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
        assert gpu_env._apply_gpu_devices_pin() is True
        assert gpu_env.os.environ["CUDA_VISIBLE_DEVICES"] == "3"
