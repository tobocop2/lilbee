"""Windows probe: where the signal wakeup descriptor points when a pool thread presses Ctrl+C.

Diagnosis only, never merged. Prints PROBE lines; asserts nothing about the add.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
import time
import traceback
from unittest import mock

import pytest

import lilbee.app.services as svc_mod
import tests.test_cli as test_cli
from lilbee.runtime import asyncio_loop
from tests.test_cli import _skip_model_validation, isolated_env, mock_svc  # noqa: F401  fixtures

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows probe")

_ORIG_SET_WAKEUP_FD = signal.set_wakeup_fd


def _p(text: str) -> None:
    print(f"PROBE {text}", flush=True)


def _site() -> str:
    frames = traceback.extract_stack()[:-2]
    keep = [f for f in frames if "test_probe_winctrlc" not in f.filename][-5:]
    return " <- ".join(
        f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}" for f in reversed(keep)
    )


def _csock(loop: asyncio.AbstractEventLoop) -> object:
    sock = getattr(loop, "_csock", None)
    try:
        return sock.fileno() if sock is not None else None
    except OSError as exc:
        return f"closed({exc})"


class _Recorder:
    def __init__(self) -> None:
        self.loops: list[tuple[asyncio.AbstractEventLoop, str]] = []
        self.log: list[str] = []
        self.t0 = time.monotonic()

    def note(self, text: str) -> None:
        self.log.append(f"t={time.monotonic() - self.t0:.3f} {text}")

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        base = asyncio.base_events.BaseEventLoop
        orig_init, orig_close = base.__init__, base.close
        recorder = self

        def _init(loop, *args, **kwargs):
            orig_init(loop, *args, **kwargs)
            name = threading.current_thread().name
            recorder.loops.append((loop, name))
            recorder.note(
                f"loop-built #{len(recorder.loops) - 1} {type(loop).__name__} id={id(loop):x}"
                f" thread={name} at {_site()}"
            )

        def _close(loop):
            name = threading.current_thread().name
            recorder.note(
                f"loop-close {type(loop).__name__} id={id(loop):x} csock={_csock(loop)}"
                f" thread={name} at {_site()}"
            )
            return orig_close(loop)

        def _set_wakeup_fd(fd, *args, **kwargs):
            name = threading.current_thread().name
            try:
                previous = _ORIG_SET_WAKEUP_FD(fd, *args, **kwargs)
            except BaseException as exc:
                recorder.note(f"set_wakeup_fd({fd}) thread={name} raised {exc!r} at {_site()}")
                raise
            recorder.note(f"set_wakeup_fd({fd}) -> {previous} thread={name} at {_site()}")
            return previous

        monkeypatch.setattr(base, "__init__", _init)
        monkeypatch.setattr(base, "close", _close)
        monkeypatch.setattr(signal, "set_wakeup_fd", _set_wakeup_fd)


@pytest.mark.parametrize("arm", ["selector", "proactor"])
def test_probe_wakeup_fd_at_the_planning_press(tmp_path, monkeypatch, arm):
    from lilbee.data.ingest import pipeline
    from tests._ingesting_services import ingesting_services
    from tests._signals import press_ctrl_c_in_a_run

    _p(f"arm={arm} python={sys.version.split()[0]} policy={type(asyncio.get_event_loop_policy()).__name__}")
    previous_policy = asyncio.get_event_loop_policy()
    if arm == "proactor":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    # A fresh process has no background loop yet; start the arm from the same state.
    asyncio_loop.shutdown()
    at_start = _ORIG_SET_WAKEUP_FD(-1)
    _ORIG_SET_WAKEUP_FD(at_start)
    _p(f"arm={arm} wakeup_fd_at_start={at_start}")

    recorder = _Recorder()
    recorder.install(monkeypatch)

    corpus = tmp_path / "source" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "notes.txt").write_text("hello world " * 50, encoding="utf-8")
    services, _sources = ingesting_services()
    svc_mod.set_services(services)
    monkeypatch.setenv("LILBEE_DATA", str(tmp_path))
    events: list[threading.Event] = []
    real_discover = pipeline.discover_corpus
    seen: dict[str, object] = {}

    def _discover(*args, **kwargs):
        if seen:
            return real_discover(*args, **kwargs)
        here = threading.current_thread().name
        main_loops = [(i, lp) for i, (lp, name) in enumerate(recorder.loops) if name == "MainThread"]
        running = [(i, lp) for i, lp in main_loops if lp.is_running()]
        _p(f"arm={arm} press_thread={here} loops_built={len(recorder.loops)} main_thread_loops="
           f"{[(i, type(lp).__name__, lp.is_running(), lp.is_closed(), _csock(lp)) for i, lp in main_loops]}")
        if not running:
            _p(f"arm={arm} no running loop on the main thread")
        else:
            run_index, run_loop = running[-1]
            later = [
                (i, type(lp).__name__, name)
                for i, (lp, name) in enumerate(recorder.loops)
                if i > run_index
            ]
            got = threading.Event()

            def _read_on_main() -> None:
                fd = _ORIG_SET_WAKEUP_FD(-1)
                _ORIG_SET_WAKEUP_FD(fd)
                seen["fd"] = fd
                seen["thread"] = threading.current_thread().name
                seen["handler"] = repr(signal.getsignal(signal.SIGINT))
                got.set()

            run_loop.call_soon_threadsafe(_read_on_main)
            ran = got.wait(5)
            _p(
                f"arm={arm} callback_ran={ran} callback_thread={seen.get('thread')}"
                f" wakeup_fd={seen.get('fd')} run_loop_csock={_csock(run_loop)}"
                f" run_loop={type(run_loop).__name__}#{run_index}"
                f" loops_built_after_run_loop={later}"
                f" sigint_handler={seen.get('handler')}"
            )
        seen.setdefault("pressed", True)
        start = time.monotonic()
        try:
            press_ctrl_c_in_a_run(events[0])
            outcome = "taken"
        except AssertionError as exc:
            outcome = f"lost ({exc})"
        _p(f"arm={arm} press={outcome} after={time.monotonic() - start:.3f}s")
        return real_discover(*args, **kwargs)

    sync_patch = test_cli.TestSyncCancelledExit._sync_recording_its_cancel(events, ctrl_c_after=False)
    try:
        with mock.patch.object(pipeline, "discover_corpus", _discover), sync_patch:
            result = test_cli.runner.invoke(
                test_cli.app, ["add", "--data-dir", str(tmp_path), str(corpus)]
            )
    finally:
        for line in recorder.log:
            _p(f"arm={arm} log {line}")
        if arm == "proactor":
            asyncio_loop.shutdown()
            asyncio.set_event_loop_policy(previous_policy)
    _p(f"arm={arm} exit_code={result.exit_code} exception={result.exception!r}")
    _p(f"arm={arm} output_tail={' '.join(result.output.split())[-200:]!r}")
