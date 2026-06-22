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
    monkeypatch.setattr(handlers, "placement", lambda: _view())
    with create_test_client([placement_route]) as client:
        r = client.get("/api/placement")
        assert r.status_code == 200
        assert r.json()["gpus"][0]["label"] == "CUDA0"


def test_put_sets(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        handlers,
        "placement_set",
        lambda spec_json: seen.setdefault("json", spec_json) or _view(True),
    )
    with create_test_client([placement_set_route]) as client:
        r = client.put("/api/placement", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 200
        assert '"chat"' in seen["json"]


def test_delete_clears(monkeypatch):
    monkeypatch.setattr(handlers, "placement_clear", lambda: _view())
    with create_test_client([placement_clear_route]) as client:
        assert client.delete("/api/placement").status_code == 200


def test_preview_unfit_returns_422(monkeypatch):
    from lilbee.providers.fleet.placement_spec import PlacementError

    def boom(spec_json):
        raise PlacementError("chat needs 70 GiB but device 0 has 40 GiB free")

    monkeypatch.setattr(handlers, "placement_preview", boom)
    with create_test_client([placement_preview_route]) as client:
        r = client.post("/api/placement/preview", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 422
        assert "40 GiB free" in r.text


def test_preview_auto(monkeypatch):
    monkeypatch.setattr(handlers, "placement_preview", lambda spec_json: _view())
    with create_test_client([placement_preview_route]) as client:
        r = client.post("/api/placement/preview", json={})
        assert r.status_code == 200
        assert r.json()["manual"] is False


def test_preview_with_spec(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        handlers,
        "placement_preview",
        lambda spec_json: captured.setdefault("json", spec_json) or _view(True),
    )
    with create_test_client([placement_preview_route]) as client:
        r = client.post("/api/placement/preview", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 200
        assert '"chat"' in captured["json"]


def test_put_missing_spec_returns_422(monkeypatch):
    monkeypatch.setattr(handlers, "placement_set", lambda spec_json: _view(True))
    with create_test_client([placement_set_route]) as client:
        r = client.put("/api/placement", json={})
        assert r.status_code == 422


def test_get_gpus(monkeypatch):
    from lilbee.server.models import GpuInfoResponse

    monkeypatch.setattr(
        handlers,
        "gpus",
        lambda: [
            GpuInfoResponse(
                index=0,
                backend="CUDA",
                label="CUDA0",
                name="A100",
                total_bytes=80 * GIB,
                free_bytes=72 * GIB,
            )
        ],
    )
    with create_test_client([gpus_route]) as client:
        r = client.get("/api/gpus")
        assert r.status_code == 200
        assert r.json()[0]["label"] == "CUDA0"
