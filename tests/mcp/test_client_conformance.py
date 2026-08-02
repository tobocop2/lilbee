"""End-to-end conformance: a real MCP client against the real lilbee server.

The rest of the MCP suite calls the server's methods in-process or drives the
mount with hand-built JSON. This drives a genuine ``ClientSession`` over the
SDK's memory transport instead, so the handshake, the tools/list wire encoding
and the call/response envelopes all run through the client half of the SDK.
That is the layer the mcp 2.x migration moved, and it is the layer an agent
actually speaks.

Every tool exercised here is chosen to need no services: an engine start in the
unit suite would leak llama-server processes onto the developer's machine.
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import anyio
from mcp.shared.memory import create_client_server_memory_streams
from tools.qa.mcp_stdio_probe import CORE_TOOLS

from lilbee.mcp_server import _offload_sync, build_mcp_server
from lilbee.mcp_server import sync as mcp_sync
from mcp import ClientSession

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def _client(server: Any = None) -> AsyncIterator[tuple[ClientSession, Any]]:
    """A connected client session and the result of its initialize handshake."""
    server = server if server is not None else build_mcp_server()
    lowlevel = server._lowlevel_server
    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        anyio.create_task_group() as tg,
    ):

        async def _serve() -> None:
            await lowlevel.run(
                server_streams[0],
                server_streams[1],
                lowlevel.create_initialization_options(),
                raise_exceptions=False,
            )

        tg.start_soon(_serve)
        async with ClientSession(client_streams[0], client_streams[1]) as session:
            yield session, await session.initialize()
        tg.cancel_scope.cancel()


async def test_the_handshake_identifies_lilbee() -> None:
    async with _client() as (_session, init):
        pass
    assert init.server_info.name == "lilbee"


async def test_the_core_tools_reach_a_real_client() -> None:
    async with _client() as (session, _init):
        names = {tool.name for tool in (await session.list_tools()).tools}
    assert names >= CORE_TOOLS


async def test_the_wire_schema_stays_trimmed_through_the_client() -> None:
    """The strip runs in list_tools, so only a real client sees the wire form.

    Guards the schema attribute the 2.x rename moved: reading the wrong one
    would surface here as untrimmed titles rather than as an import error.
    """
    async with _client() as (session, _init):
        tools = (await session.list_tools()).tools
    for tool in tools:
        schema = tool.input_schema
        assert "title" not in schema, f"{tool.name}: top-level title on the wire"
        for prop_name, prop in schema.get("properties", {}).items():
            assert "title" not in prop, f"{tool.name}.{prop_name}: property title on the wire"
            assert "default" not in prop, f"{tool.name}.{prop_name}: default on the wire"


async def test_tool_descriptions_arrive_flattened() -> None:
    async with _client() as (session, _init):
        tools = (await session.list_tools()).tools
    for tool in tools:
        if tool.description:
            assert "\n  " not in tool.description, f"{tool.name}: unflattened indentation"


async def test_a_tool_call_round_trips_a_result() -> None:
    async with _client() as (session, _init):
        result = await session.call_tool("settings_list", {})
    assert not result.is_error
    assert result.content


async def test_a_refused_call_round_trips_the_error_envelope() -> None:
    """The envelope is a normal result, not a protocol error, so agents can read it."""
    async with _client() as (session, _init):
        result = await session.call_tool("search", {"query": "   "})
    assert not result.is_error
    assert "query must not be empty" in str(result.content)


async def test_an_unknown_tool_is_a_protocol_error() -> None:
    async with _client() as (session, _init):
        result = await session.call_tool("no_such_tool", {})
    assert result.is_error


async def test_a_cancelled_sync_handler_releases_the_caller() -> None:
    """A cancelling caller is freed while the worker thread is still blocked.

    Deterministic rather than timed: the handler parks until *release* is set,
    so the cancel scope can only exit if the wait was abandoned. Without
    abandon_on_cancel the scope stays open until the thread returns, which on
    the shared mount holds a connection open for a request nobody awaits.
    """
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def parked() -> str:
        started.set()
        release.wait(timeout=10)
        finished.set()
        return "done"

    runner = _offload_sync(parked)
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(runner)
            while not started.is_set():  # the thread is inside the handler
                await anyio.sleep(0.01)
            tg.cancel_scope.cancel()
        # The discriminator: without abandon_on_cancel the scope cannot close
        # until the thread returns, so finished would be set by the time we
        # get here. Still parked means the wait really was abandoned.
        assert not finished.is_set(), "the cancel waited for the worker thread"
    finally:
        release.set()


async def test_a_cancelled_ingest_tool_signals_the_pipeline(monkeypatch) -> None:
    """Cancelling sync hands the pipeline the stop token, as a disconnect does on HTTP.

    A worker thread cannot be interrupted, so this token is the only thing that
    actually stops an extraction: the pipeline polls it and halts between files.
    """
    import lilbee.data.ingest as ingest_mod

    seen: dict[str, threading.Event] = {}

    async def fake_sync(*_args, cancel=None, **_kwargs):
        assert cancel is not None, "sync tool called the pipeline with no cancel token"
        seen["token"] = cancel
        await anyio.sleep(10)  # still running when the caller gives up

    monkeypatch.setattr(ingest_mod, "sync", fake_sync)
    with anyio.move_on_after(0.2):
        await mcp_sync()
    assert seen["token"].is_set(), "the pipeline was never told to stop"


async def test_a_running_crawl_can_be_cancelled_through_a_real_client(monkeypatch) -> None:
    """crawl runs detached, so an explicit tool is the only way to stop it.

    Cancelling the crawl tool call cannot work: it returns a task id
    immediately and the work outlives the request.
    """
    from lilbee.crawler import task as task_mod

    task = task_mod.CrawlTask(task_id="probe", url="https://example.com", depth=1, max_pages=5)
    task.status = task_mod.TaskStatus.RUNNING
    monkeypatch.setitem(task_mod._registry.tasks, "probe", task)

    async with _client() as (session, _init):
        result = await session.call_tool("crawl_cancel", {"task_id": "probe"})
        missing = await session.call_tool("crawl_cancel", {"task_id": "no-such-task"})

    assert not result.is_error
    assert task.cancel.is_set(), "the running crawl was never told to stop"
    assert "No task found" in str(missing.content)


async def test_a_cancelled_model_pull_aborts_the_download(monkeypatch) -> None:
    """The download runs on a thread, so the stop has to reach it through the
    progress callback: returning would leave a multi-GB pull running."""
    import lilbee.app.models as models_mod
    from lilbee.mcp_server import model_pull
    from lilbee.runtime.cancellation import TaskCancelledError

    state = {"ticks": 0, "aborted": False}

    def fake_pull(_ref, _src, *, on_update, allow_unsupported=False):
        for _ in range(200):
            time.sleep(0.01)
            state["ticks"] += 1
            try:
                on_update(SimpleNamespace(percent=1.0, detail="x"))
            except TaskCancelledError:
                state["aborted"] = True
                raise

    monkeypatch.setattr(models_mod, "pull_model_data", fake_pull)
    with anyio.move_on_after(0.3):
        await model_pull("some/model")
    at_cancel = state["ticks"]
    await anyio.sleep(0.5)
    assert state["aborted"], "the download never saw the cancellation"
    assert state["ticks"] < at_cancel + 40, "the download kept going well past the cancel"


async def test_a_cancelled_dataset_import_stops_re_embedding(monkeypatch) -> None:
    """Import re-embeds on a worker thread, so cancelling the await stops the
    loop only between sources. The progress hook is what stops it mid-batch."""
    import lilbee.app.dataset as dataset_mod
    from lilbee.mcp_server import import_dataset
    from lilbee.runtime.cancellation import TaskCancelledError
    from lilbee.runtime.progress import EventType

    state = {"rows": 0, "aborted": False}

    async def fake_import(_path, _fmt, *, on_progress):
        def _embed_batch() -> None:  # the real embed runs off the loop
            for _ in range(200):
                time.sleep(0.01)
                state["rows"] += 1
                try:
                    on_progress(EventType.EMBED, object())
                except TaskCancelledError:
                    state["aborted"] = True
                    raise

        await anyio.to_thread.run_sync(_embed_batch, abandon_on_cancel=True)

    monkeypatch.setattr(dataset_mod, "import_from_path", fake_import)
    with anyio.move_on_after(0.3):
        await import_dataset("/tmp/does-not-matter.jsonl")
    at_cancel = state["rows"]
    await anyio.sleep(0.5)
    assert state["aborted"], "the import never saw the cancellation"
    assert state["rows"] < at_cancel + 30, "the import kept re-embedding past the cancel"
