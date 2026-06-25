"""Tests for placement and gpus HTTP routes."""

from __future__ import annotations

from litestar.testing import create_test_client

import lilbee.server.handlers as handlers
from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
from lilbee.providers.roles import WorkerRole
from lilbee.server.routes.placement import (
    gpus_route,
    placement_clear_route,
    placement_preview_route,
    placement_route,
    placement_set_route,
)

GIB = 1024**3


def _view(manual=False):
    return PlacementView(
        gpus=(GpuInfo(0, "CUDA", "CUDA0", "NVIDIA A100", 80 * GIB, 72 * GIB),),
        roles=(RolePlacementView(WorkerRole.CHAT, "org/chat.gguf", (0,), None, 1),),
        unplaceable=(),
        manual=manual,
        spec_json=None,
    )


def test_get_placement(monkeypatch):
    async def _placement():
        return _view()

    monkeypatch.setattr(handlers, "placement", _placement)
    with create_test_client([placement_route]) as client:
        r = client.get("/api/placement")
        assert r.status_code == 200
        assert r.json()["gpus"][0]["label"] == "CUDA0"


def test_put_refused_on_daemon():
    """PUT placement is refused on the HTTP daemon (rebuilds the shared fleet)."""
    with create_test_client([placement_set_route]) as client:
        r = client.put("/api/placement", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 409
        assert "CLI" in r.text


def test_delete_refused_on_daemon():
    """DELETE placement is refused on the HTTP daemon (rebuilds the shared fleet)."""
    with create_test_client([placement_clear_route]) as client:
        r = client.delete("/api/placement")
        assert r.status_code == 409
        assert "CLI" in r.text


def test_preview_requires_auth():
    """preview is not read-only: it runs subprocess probes and must require auth (bb-895)."""
    from lilbee.server.auth import is_read_only

    assert not is_read_only(placement_preview_route.fn)
    assert is_read_only(placement_route.fn)
    assert is_read_only(gpus_route.fn)


def test_preview_provider_error_returns_503(monkeypatch):
    from lilbee.providers.base import ProviderError

    async def boom(spec_json):
        raise ProviderError("llama-server binary not found")

    monkeypatch.setattr(handlers, "placement_preview", boom)
    with create_test_client([placement_preview_route]) as client:
        r = client.post("/api/placement/preview", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 503
        assert "llama-server" in r.text


def test_get_placement_provider_error_returns_503(monkeypatch):
    from lilbee.providers.base import ProviderError

    async def boom():
        raise ProviderError("no engine")

    monkeypatch.setattr(handlers, "placement", boom)
    with create_test_client([placement_route]) as client:
        r = client.get("/api/placement")
        assert r.status_code == 503


def test_preview_unfit_returns_422(monkeypatch):
    from lilbee.providers.fleet.placement_spec import PlacementError

    async def boom(spec_json):
        raise PlacementError("chat needs 70 GiB but device 0 has 40 GiB free")

    monkeypatch.setattr(handlers, "placement_preview", boom)
    with create_test_client([placement_preview_route]) as client:
        r = client.post("/api/placement/preview", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 422
        assert "40 GiB free" in r.text


def test_preview_auto(monkeypatch):
    async def _preview(spec_json):
        return _view()

    monkeypatch.setattr(handlers, "placement_preview", _preview)
    with create_test_client([placement_preview_route]) as client:
        r = client.post("/api/placement/preview", json={})
        assert r.status_code == 200
        assert r.json()["manual"] is False


def test_preview_with_spec(monkeypatch):
    captured = {}

    async def _preview(spec_json):
        captured["json"] = spec_json
        return _view(True)

    monkeypatch.setattr(handlers, "placement_preview", _preview)
    with create_test_client([placement_preview_route]) as client:
        r = client.post("/api/placement/preview", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 200
        assert '"chat"' in captured["json"]


def test_put_refused_regardless_of_body():
    """PUT is refused even with an empty body (refusal precedes validation)."""
    with create_test_client([placement_set_route]) as client:
        r = client.put("/api/placement", json={})
        assert r.status_code == 409


def test_get_gpus(monkeypatch):
    from lilbee.server.models import GpuInfoResponse

    async def _gpus():
        return [
            GpuInfoResponse(
                index=0,
                backend="CUDA",
                label="CUDA0",
                name="A100",
                total_bytes=80 * GIB,
                free_bytes=72 * GIB,
            )
        ]

    monkeypatch.setattr(handlers, "gpus", _gpus)
    with create_test_client([gpus_route]) as client:
        r = client.get("/api/gpus")
        assert r.status_code == 200
        assert r.json()[0]["label"] == "CUDA0"
