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


def test_put_sets(monkeypatch):
    seen = {}

    async def _set(spec_json):
        seen["json"] = spec_json
        return _view(True)

    monkeypatch.setattr(handlers, "placement_set", _set)
    with create_test_client([placement_set_route]) as client:
        r = client.put("/api/placement", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 200
        assert '"chat"' in seen["json"]


def test_delete_clears(monkeypatch):
    cleared = {"called": False}

    async def _clear():
        cleared["called"] = True
        return _view()

    monkeypatch.setattr(handlers, "placement_clear", _clear)
    with create_test_client([placement_clear_route]) as client:
        r = client.delete("/api/placement")
        assert r.status_code == 200
        assert r.json()["manual"] is False
        assert r.json()["gpus"][0]["label"] == "CUDA0"
        assert cleared["called"]


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


def test_put_missing_spec_returns_422(monkeypatch):
    async def _set(spec_json):
        return _view(True)

    monkeypatch.setattr(handlers, "placement_set", _set)
    with create_test_client([placement_set_route]) as client:
        r = client.put("/api/placement", json={})
        assert r.status_code == 422


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
