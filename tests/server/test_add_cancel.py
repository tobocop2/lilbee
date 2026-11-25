"""POST /api/add/cancel, the add_id of the add routes, and what a cancel by the user undoes."""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest
from litestar.testing import AsyncTestClient

from lilbee.app.services import get_services
from lilbee.core.config import cfg
from lilbee.server.handlers import SseStream, cancel_add
from lilbee.server.handlers.ingest import _ingest_stream
from lilbee.server.models import AddSummary
from tests.server.conftest import parse_sse_events
from tests.server.test_handlers import (  # noqa: F401 (autouse fixtures)
    _auth_headers,
    isolated_env,
    mock_svc,
    reset_ingest_locks,
)


class _Stub:
    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


def _app():
    from lilbee.server.app import create_app

    return create_app()


class TestTheCancelRoute:
    async def test_a_cancel_stops_the_add_that_runs_under_the_id(self):
        run = _Stub()
        with get_services().add_registry.running("a-1", run.stop):
            async with AsyncTestClient(_app()) as client:
                resp = await client.post(
                    "/api/add/cancel", json={"add_id": "a-1"}, headers=_auth_headers()
                )
        assert resp.status_code == 200
        assert resp.json() == {"add_id": "a-1", "cancelling": True}
        assert run.stops == 1

    async def test_a_cancel_of_an_id_that_does_not_run_answers_404(self):
        async with AsyncTestClient(_app()) as client:
            resp = await client.post(
                "/api/add/cancel", json={"add_id": "gone"}, headers=_auth_headers()
            )
        assert resp.status_code == 404
        assert "gone" in resp.json()["detail"]

    @pytest.mark.parametrize("add_id", ["", "has space", "a/b", "x" * 65])
    async def test_a_cancel_with_a_bad_id_answers_400(self, add_id):
        async with AsyncTestClient(_app()) as client:
            resp = await client.post(
                "/api/add/cancel", json={"add_id": add_id}, headers=_auth_headers()
            )
        assert resp.status_code == 400


class TestTheAddIdOfTheAddRoutes:
    async def test_an_add_with_a_bad_id_answers_400(self, tmp_path):
        async with AsyncTestClient(_app()) as client:
            resp = await client.post(
                "/api/add",
                json={"paths": [str(tmp_path / "a.txt")], "add_id": "bad id"},
                headers=_auth_headers(),
            )
        assert resp.status_code == 400

    async def test_an_upload_with_a_bad_id_answers_400(self):
        async with AsyncTestClient(_app()) as client:
            resp = await client.post(
                "/api/add/upload",
                files={"data": ("a.md", b"# a\n")},
                params={"add_id": "x" * 65},
                headers=_auth_headers(),
            )
        assert resp.status_code == 400
        assert not (cfg.documents_dir / "a.md").exists()

    async def test_an_add_with_an_id_that_runs_answers_409(self, tmp_path):
        with get_services().add_registry.running("busy", _Stub().stop):
            async with AsyncTestClient(_app()) as client:
                resp = await client.post(
                    "/api/add",
                    json={"paths": [str(tmp_path / "a.txt")], "add_id": "busy"},
                    headers=_auth_headers(),
                )
        assert resp.status_code == 409
        assert "busy" in resp.json()["detail"]

    async def test_an_upload_with_an_id_that_runs_answers_409_and_writes_nothing(self):
        with get_services().add_registry.running("busy", _Stub().stop):
            async with AsyncTestClient(_app()) as client:
                resp = await client.post(
                    "/api/add/upload",
                    files={"data": ("a.md", b"# a\n")},
                    params={"add_id": "busy"},
                    headers=_auth_headers(),
                )
        assert resp.status_code == 409
        assert not (cfg.documents_dir / "a.md").exists()

    async def test_the_schema_publishes_the_answers_of_the_cancel(self):
        async with AsyncTestClient(_app()) as client:
            paths = (await client.get("/schema/openapi.json", headers=_auth_headers())).json()[
                "paths"
            ]
        assert {"200", "400", "404"} <= set(paths["/api/add/cancel"]["post"]["responses"])
        assert "409" in paths["/api/add"]["post"]["responses"]
        assert "409" in paths["/api/add/upload"]["post"]["responses"]
        params = {p["name"] for p in paths["/api/add/upload"]["post"]["parameters"]}
        assert "add_id" in params


class TestTheStreamHoldsTheId:
    async def test_the_id_runs_while_the_stream_runs_and_is_free_after(self):
        seen: list[bool] = []

        async def run(_locked, sse) -> AddSummary:
            seen.append(get_services().add_registry.is_running("s-1"))
            sse.queue.put_nowait(None)
            return AddSummary(copied=[], errors=[])

        frames = [f async for f in _ingest_stream([("a.md", "a")], run, "Test stream", "s-1")]

        assert seen == [True]
        assert not get_services().add_registry.is_running("s-1")
        assert [event for event, _data in parse_sse_events("".join(frames).encode())] == ["done"]

    async def test_a_stream_whose_id_another_add_holds_ends_with_an_error(self):
        run = mock.AsyncMock()
        with get_services().add_registry.running("s-1", _Stub().stop):
            frames = [f async for f in _ingest_stream([("a.md", "a")], run, "Test stream", "s-1")]

        events = parse_sse_events("".join(frames).encode())
        assert [event for event, _data in events] == ["error"]
        assert "'s-1' is already running" in events[0][1]["message"]
        run.assert_not_called()
        assert get_services().ingest_lock_registry._locks == {}

    async def test_a_stop_by_the_user_cancels_the_run_and_still_ends_with_done(self):
        summary = AddSummary(copied=[], errors=[], cancelled=True, not_added=["a.md"])
        parked = asyncio.Event()

        async def run(_locked, sse) -> AddSummary:
            try:
                parked.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                assert sse.stopped_by_user
            sse.queue.put_nowait(None)
            return summary

        async def _cancel_when_parked() -> bool:
            await parked.wait()
            return cancel_add("s-2")

        canceller = asyncio.create_task(_cancel_when_parked())
        frames = [f async for f in _ingest_stream([("a.md", "a")], run, "Test stream", "s-2")]

        assert await canceller is True
        events = parse_sse_events("".join(frames).encode())
        assert [data for event, data in events if event == "done"] == [summary.model_dump()]


class TestAStopByTheUser:
    async def test_the_stop_cancels_a_running_task(self):
        sse = SseStream()
        task = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)

        sse.stop_by_user(task)

        with pytest.raises(asyncio.CancelledError):
            await task
        assert sse.stopped_by_user and sse.cancel.is_set()

    async def test_the_stop_leaves_a_task_that_has_not_started_to_see_the_cancel(self):
        sse = SseStream()

        async def run() -> bool:
            return sse.cancel.is_set()

        task = asyncio.create_task(run())
        sse.stop_by_user(task)

        assert await task is True
        assert sse.stopped_by_user

    async def test_the_stop_leaves_a_run_that_already_stops_alone(self):
        sse = SseStream()
        sse.cancel.set()
        task = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)

        sse.stop_by_user(task)

        assert not sse.stopped_by_user and not task.cancelling()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
