"""Tests for placement and gpus HTTP routes."""

from __future__ import annotations

from litestar.testing import create_test_client

import lilbee.server.handlers as handlers
from lilbee.app.placement import GpuInfo, PlacementView, RolePlacementView
from lilbee.providers.roles import WorkerRole
from lilbee.server.routes.placement import (
    gpu_stats_stream_route,
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


def test_placement_routes_require_auth():
    """No placement route is read-only: they all run subprocess device probes."""
    from lilbee.server.auth import is_read_only

    assert not is_read_only(placement_preview_route.fn)
    assert not is_read_only(placement_route.fn)
    assert not is_read_only(gpus_route.fn)


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


def test_get_gpus_provider_error_returns_503(monkeypatch):
    from lilbee.providers.base import ProviderError

    async def boom():
        raise ProviderError("no engine")

    monkeypatch.setattr(handlers, "gpus", boom)
    with create_test_client([gpus_route]) as client:
        r = client.get("/api/gpus")
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


def _enable_http_placement(monkeypatch):
    """Flip the allow_http_placement gate the routes read."""
    from lilbee.server.routes import placement as placement_routes

    monkeypatch.setattr(placement_routes.cfg, "allow_http_placement", True)


def test_put_applies_when_enabled(monkeypatch):
    """With the flag on, PUT applies the spec and returns the new placement."""
    _enable_http_placement(monkeypatch)
    captured = {}

    async def _set(spec_json):
        captured["json"] = spec_json
        return _view(True)

    monkeypatch.setattr(handlers, "placement_set", _set)
    with create_test_client([placement_set_route]) as client:
        r = client.put("/api/placement", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 200
        assert r.json()["manual"] is True
        assert '"chat"' in captured["json"]


def test_put_missing_spec_returns_422_when_enabled(monkeypatch):
    """With the flag on, PUT without a spec is a 422 (clear via DELETE for auto)."""
    _enable_http_placement(monkeypatch)
    with create_test_client([placement_set_route]) as client:
        r = client.put("/api/placement", json={})
        assert r.status_code == 422
        assert "spec is required" in r.text


def test_put_unfit_spec_returns_422_when_enabled(monkeypatch):
    """An infeasible spec surfaces PlacementError as a 422 even with the flag on."""
    from lilbee.providers.fleet.placement_spec import PlacementError

    _enable_http_placement(monkeypatch)

    async def boom(spec_json):
        raise PlacementError("chat needs 70 GiB but device 0 has 40 GiB free")

    monkeypatch.setattr(handlers, "placement_set", boom)
    with create_test_client([placement_set_route]) as client:
        r = client.put("/api/placement", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 422
        assert "40 GiB free" in r.text


def test_put_provider_error_returns_503_when_enabled(monkeypatch):
    """A provider failure while applying surfaces as a 503."""
    from lilbee.providers.base import ProviderError

    _enable_http_placement(monkeypatch)

    async def boom(spec_json):
        raise ProviderError("llama-server binary not found")

    monkeypatch.setattr(handlers, "placement_set", boom)
    with create_test_client([placement_set_route]) as client:
        r = client.put("/api/placement", json={"spec": {"chat": {"devices": [0]}}})
        assert r.status_code == 503
        assert "llama-server" in r.text


def test_delete_clears_when_enabled(monkeypatch):
    """With the flag on, DELETE clears placement and returns the auto plan."""
    _enable_http_placement(monkeypatch)

    async def _clear():
        return _view(False)

    monkeypatch.setattr(handlers, "placement_clear", _clear)
    with create_test_client([placement_clear_route]) as client:
        r = client.delete("/api/placement")
        assert r.status_code == 200
        assert r.json()["manual"] is False


def test_delete_provider_error_returns_503_when_enabled(monkeypatch):
    """A provider failure while clearing surfaces as a 503."""
    from lilbee.providers.base import ProviderError

    _enable_http_placement(monkeypatch)

    async def boom():
        raise ProviderError("no engine")

    monkeypatch.setattr(handlers, "placement_clear", boom)
    with create_test_client([placement_clear_route]) as client:
        r = client.delete("/api/placement")
        assert r.status_code == 503


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


def test_gpu_stats_stream_provider_error_returns_503(monkeypatch):
    """ProviderError from get_placement() before the stream starts maps to 503."""
    import lilbee.app.placement as placement_mod
    from lilbee.providers.base import ProviderError

    def _boom():
        raise ProviderError("llama-server binary not found")

    monkeypatch.setattr(placement_mod, "get_placement", _boom)
    with create_test_client([gpu_stats_stream_route]) as client:
        r = client.get("/api/gpus/stream")
        assert r.status_code == 503
        assert "llama-server" in r.text


def test_gpu_stats_stream_streams_events(monkeypatch):
    """Happy-path: stream route returns SSE events from the generator."""
    import lilbee.app.placement as placement_mod
    from lilbee.app.placement import PlacementView

    view = PlacementView(gpus=(), roles=(), unplaceable=(), manual=False, spec_json=None)

    async def _fake_stream(devices):
        yield 'event: gpu_stats\ndata: {"gpus": []}\n\n'

    monkeypatch.setattr(placement_mod, "get_placement", lambda: view)
    monkeypatch.setattr(handlers, "gpu_stats_stream", _fake_stream)
    with create_test_client([gpu_stats_stream_route]) as client:
        r = client.get("/api/gpus/stream")
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
