"""The wait for work on a thread through a cancel of the caller: order and state, no durations."""

from __future__ import annotations

import asyncio
import functools
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from unittest import mock

import anyio
import anyio.lowlevel
import anyio.to_thread
import pytest

from lilbee.runtime.cancellation import TaskCancelledError
from lilbee.runtime.thread_end import run_to_thread_end

_Offload = Callable[[Callable[[], str]], Awaitable[str]]

# The turns of the event loop given to a cancel to land and to be delivered again.
_TURNS = 20


@pytest.fixture(params=["asyncio.to_thread", "anyio.to_thread.run_sync"])
def offload(request: pytest.FixtureRequest) -> _Offload:
    """Each way the product puts work on a thread."""
    if request.param == "asyncio.to_thread":
        return asyncio.to_thread
    return anyio.to_thread.run_sync


class _ParkedWork:
    """Work that holds its thread until the test releases it, then ends as it was told."""

    def __init__(self, order: list[str], ending: BaseException | None = None) -> None:
        self._loop = asyncio.get_running_loop()
        self._order = order
        self._ending = ending
        self._release = threading.Event()
        self.reached = asyncio.Event()

    def __call__(self) -> str:
        self._loop.call_soon_threadsafe(self.reached.set)
        assert self._release.wait(60), "the test never released the thread"
        self._order.append("thread ended")
        if self._ending is not None:
            raise self._ending
        return "done"

    def release(self) -> None:
        self._order.append("thread released")
        self._release.set()

    def let_go(self) -> None:
        """Free the thread of a test that failed before its release."""
        self._release.set()


_Park = Callable[..., _ParkedWork]


@pytest.fixture
async def parked() -> AsyncIterator[_Park]:
    """Make parked work; no thread stays parked after the test, whatever its result."""
    made: list[_ParkedWork] = []

    def _park(order: list[str], ending: BaseException | None = None) -> _ParkedWork:
        made.append(_ParkedWork(order, ending))
        return made[-1]

    yield _park
    for work in made:
        work.let_go()


class _Call:
    """One call of the helper on a task of its own, inside an anyio cancel scope."""

    def __init__(self, offload: _Offload, work: Callable[[], str], order: list[str]) -> None:
        self.token = threading.Event()
        self.scope = anyio.CancelScope()
        self.after_the_helper: list[str] = []
        self._start = functools.partial(offload, work)
        self._order = order
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> str:
        with self.scope:
            try:
                result = await run_to_thread_end(self._start, self.token)
            finally:
                self._order.append("call ended")
            self.after_the_helper.append("returned")
            await anyio.lowlevel.checkpoint()
            self.after_the_helper.append("passed a checkpoint")
            return result
        return "the scope caught its cancel"


async def _settle() -> None:
    """Let every task run until it waits again."""
    await anyio.wait_all_tasks_blocked()


async def test_a_caller_already_cancelled_starts_nothing(offload):
    started: list[str] = []
    token = threading.Event()

    with anyio.CancelScope() as scope:
        scope.cancel()
        await run_to_thread_end(functools.partial(offload, lambda: started.append("work")), token)

    assert scope.cancelled_caught
    assert started == []
    assert not token.is_set()


async def test_a_task_cancelled_before_the_call_starts_nothing(offload):
    started: list[str] = []
    token = threading.Event()

    async def _cancel_itself_then_call() -> None:
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        await run_to_thread_end(functools.partial(offload, lambda: started.append("work")), token)

    task = asyncio.create_task(_cancel_itself_then_call())
    await asyncio.wait([task])

    assert task.cancelled()
    assert started == []
    assert not token.is_set()


async def test_no_cancel_returns_the_result_and_leaves_the_token_unset(offload, parked):
    order: list[str] = []
    work = parked(order)
    call = _Call(offload, work, order)

    await work.reached.wait()
    work.release()

    assert await call.task == "done"
    assert order == ["thread released", "thread ended", "call ended"]
    assert not call.token.is_set()


@pytest.mark.parametrize("cancel", ["scope", "task"])
async def test_a_cancel_sets_the_token_before_the_thread_is_released(offload, parked, cancel):
    order: list[str] = []
    work = parked(order)
    call = _Call(offload, work, order)
    await work.reached.wait()
    assert not call.token.is_set()

    if cancel == "scope":
        call.scope.cancel()
    else:
        call.task.cancel()
    await _settle()

    assert call.token.is_set()
    assert order == []
    assert not call.task.done()
    work.release()
    await asyncio.wait([call.task])
    assert order == ["thread released", "thread ended", "call ended"]


async def test_a_second_cancel_does_not_end_the_wait_or_abandon_the_thread(offload, parked):
    order: list[str] = []
    work = parked(order)
    call = _Call(offload, work, order)
    await work.reached.wait()
    call.task.cancel()
    await _settle()

    call.task.cancel()
    await _settle()

    assert not call.task.done()
    assert order == []
    work.release()
    assert await call.task == "done"
    assert order == ["thread released", "thread ended", "call ended"]


async def test_a_cancelled_scope_does_not_wake_the_wait_again(offload, parked):
    order: list[str] = []
    work = parked(order)
    with mock.patch.object(asyncio, "wait", wraps=asyncio.wait) as waits:
        call = _Call(offload, work, order)
        await work.reached.wait()

        call.scope.cancel()
        for _ in range(_TURNS):
            await asyncio.sleep(0)

        assert call.token.is_set()
        assert waits.await_count == 2
        assert order == []
        work.release()
        await call.task
    assert order == ["thread released", "thread ended", "call ended"]


async def test_an_exception_with_no_cancel_reaches_the_caller_unchanged(offload, parked):
    order: list[str] = []
    boom = ValueError("the work failed")
    work = parked(order, ending=boom)
    call = _Call(offload, work, order)
    await work.reached.wait()

    work.release()
    await asyncio.wait([call.task])

    assert call.task.exception() is boom
    assert not call.token.is_set()
    assert order == ["thread released", "thread ended", "call ended"]


async def test_a_stop_error_with_no_cancel_reaches_the_caller_unchanged(offload, parked):
    order: list[str] = []
    stopped = TaskCancelledError("stopped by something else")
    work = parked(order, ending=stopped)
    call = _Call(offload, work, order)
    await work.reached.wait()

    work.release()
    await asyncio.wait([call.task])

    assert call.task.exception() is stopped
    assert not call.token.is_set()


async def test_work_that_completes_under_a_cancelled_scope_returns_and_the_scope_then_cancels(
    offload, parked
):
    order: list[str] = []
    work = parked(order)
    call = _Call(offload, work, order)
    await work.reached.wait()
    call.scope.cancel()
    await _settle()

    work.release()

    assert await call.task == "the scope caught its cancel"
    assert call.after_the_helper == ["returned"]
    assert call.scope.cancelled_caught
    assert call.token.is_set()
    assert order == ["thread released", "thread ended", "call ended"]


async def test_work_that_completes_under_a_task_cancel_returns_and_the_task_is_not_cancelled(
    offload, parked
):
    order: list[str] = []
    work = parked(order)
    call = _Call(offload, work, order)
    await work.reached.wait()
    call.task.cancel()
    await _settle()

    work.release()

    assert await call.task == "done"
    assert not call.task.cancelled()
    assert call.after_the_helper == ["returned", "passed a checkpoint"]
    assert not call.scope.cancelled_caught
    assert call.token.is_set()
    assert order == ["thread released", "thread ended", "call ended"]


async def test_work_that_fails_under_a_cancel_raises_its_own_error(offload, parked):
    order: list[str] = []
    boom = ValueError("the work failed")
    work = parked(order, ending=boom)
    call = _Call(offload, work, order)
    await work.reached.wait()
    call.task.cancel()
    await _settle()

    work.release()
    await asyncio.wait([call.task])

    assert call.task.exception() is boom
    assert call.token.is_set()


@pytest.mark.parametrize("cancel", ["scope", "task"])
async def test_work_that_stops_on_the_cancel_raises_the_cancel(offload, parked, cancel):
    order: list[str] = []
    work = parked(order, ending=TaskCancelledError("stopped at the token"))
    call = _Call(offload, work, order)
    await work.reached.wait()
    if cancel == "scope":
        call.scope.cancel()
    else:
        call.task.cancel()
    await _settle()

    work.release()
    await asyncio.wait([call.task])

    assert call.after_the_helper == []
    assert order == ["thread released", "thread ended", "call ended"]
    if cancel == "scope":
        assert call.task.result() == "the scope caught its cancel"
        assert call.scope.cancelled_caught
    else:
        assert call.task.cancelled()
