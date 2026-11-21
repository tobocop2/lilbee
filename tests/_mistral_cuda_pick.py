"""A stand-in for the mistral.rs build pick that lets a Linux host with a GPU pick a CUDA build."""

from __future__ import annotations

import pytest

from lilbee.providers.fleet.engines import mistral
from lilbee.providers.fleet.host_caps import HostCaps, HostOS

_CUDA_BACKEND = "cuda128-sm89"


def pick_cuda_on_gpu_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a Linux host with a GPU pick a CUDA build, to test the launch rules of a GPU build."""
    real = mistral.MistralAdapter.pick_build

    def pick(self: mistral.MistralAdapter, host: HostCaps) -> mistral.BuildPick:
        if host.os is HostOS.LINUX and host.gpus:
            return mistral.BuildPick(_CUDA_BACKEND, "")
        return real(self, host)

    monkeypatch.setattr(mistral.MistralAdapter, "pick_build", pick)
