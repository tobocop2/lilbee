"""The stall guard aborts and resumes transfers that stop reporting bytes.

A wedged transfer blocks its worker thread forever while the task shows
active: hf_xet can deadlock before the first byte, and a dead socket never
wakes the plain path's read. The guard is the only mechanism that turns
that state into a visible failure, so these tests pin its firing rule, its
pulse plumbing, and the resume loop around it.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from lilbee.catalog import download as dl
from lilbee.catalog.models import CatalogModel
from lilbee.runtime.cancellation import TaskCancelledError


def _entry() -> CatalogModel:
    return CatalogModel(
        hf_repo="user/repo",
        gguf_filename="f.gguf",
        size_gb=1.0,
        min_ram_gb=2,
        description="d",
        featured=False,
        downloads=0,
        task="chat",
    )


def _config() -> dl.DownloadConfig:
    return dl.DownloadConfig(repo_id="user/repo", filename="f.gguf", token=None)


class TestStallGuard:
    def test_fires_after_a_quiet_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        aborts: list[str] = []
        monkeypatch.setattr(dl, "_abort_stalled_transfer", lambda: aborts.append("abort"))

        with dl._StallGuard(window_s=0.0, poll_s=0.01, floor_bytes=1) as guard:
            deadline = time.monotonic() + 5.0
            while not guard.fired and time.monotonic() < deadline:
                time.sleep(0.01)

        assert guard.fired
        assert aborts == ["abort"]

    def test_does_not_fire_inside_the_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        aborts: list[str] = []
        monkeypatch.setattr(dl, "_abort_stalled_transfer", lambda: aborts.append("abort"))

        with dl._StallGuard(window_s=60.0, poll_s=0.01) as guard:
            time.sleep(0.05)

        assert not guard.fired
        assert aborts == []

    def test_wrapped_tqdm_pulses_the_byte_count(self) -> None:
        guard = dl._StallGuard()
        pulses: list[float] = []
        guard.pulse = lambda n=0: pulses.append(n)  # type: ignore[method-assign]

        cls = guard.wrap_tqdm(None)
        bar = cls(total=10, disable=True)
        bar.update(5)

        assert pulses == [5]

    def test_wrapped_tqdm_pulses_on_the_transfer_stream(self) -> None:
        """A base exposing update_transfer (the xet stream) keeps it, pulsed."""
        calls: list[tuple[str, float]] = []

        class _Base:
            def update(self, n: float = 1) -> None:
                calls.append(("update", n))

            def update_transfer(self, n: float = 1) -> None:
                calls.append(("update_transfer", n))

        guard = dl._StallGuard()
        pulses: list[float] = []
        guard.pulse = lambda n=0: pulses.append(n)  # type: ignore[method-assign]

        bar = guard.wrap_tqdm(_Base)()
        bar.update(3)
        bar.update_transfer(4)

        assert calls == [("update", 3), ("update_transfer", 4)]
        assert pulses == [3, 4]

    def test_wrapping_does_not_invent_the_transfer_stream(self) -> None:
        """The hub feature-detects update_transfer; a base without it must not
        grow one, or the hub would feed a stream the base cannot aggregate."""

        class _Base:
            def update(self, n: float = 1) -> None:
                pass

        cls = dl._StallGuard().wrap_tqdm(_Base)
        assert not hasattr(cls, "update_transfer")

    def test_abort_stops_xet_and_closes_the_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import huggingface_hub.utils._http as hub_http

        calls: list[str] = []
        monkeypatch.setattr(dl, "abort_active_download", lambda: calls.append("xet"))
        monkeypatch.setattr(hub_http, "close_session", lambda: calls.append("client"))

        dl._abort_stalled_transfer()

        assert calls == ["xet", "client"]


class _GuardStub:
    """A guard whose firing is scripted, with inert plumbing."""

    fired_script: bool = True

    def __init__(self, *_a: object, **_k: object) -> None:
        self.fired = False

    def wrap_tqdm(self, tqdm_class: object) -> object:
        return tqdm_class

    def __enter__(self) -> _GuardStub:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.fired = self.fired_script


class TestStallRetries:
    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        outcomes: list[Exception | Path],
        fired: bool,
    ) -> tuple[Path | None, int]:
        calls = {"n": 0}

        def _fake_transfer(entry: CatalogModel, config: dl.DownloadConfig) -> Path:
            outcome = outcomes[calls["n"]]
            calls["n"] += 1
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        stub = type("_Stub", (_GuardStub,), {"fired_script": fired})
        monkeypatch.setattr(dl, "_StallGuard", stub)
        monkeypatch.setattr(dl, "_hf_download_or_translate", _fake_transfer)
        return dl._download_with_stall_guard(_entry(), _config()), calls["n"]

    def test_a_stall_resumes_and_finishes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        path = Path("/models/f.gguf")
        result, attempts = self._run(
            monkeypatch, [RuntimeError("aborted"), RuntimeError("aborted"), path], fired=True
        )
        assert result == path
        assert attempts == 3

    def test_a_persistent_stall_fails_with_the_resume_note(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(RuntimeError, match=r"stalled.*resumes where it stopped"):
            self._run(
                monkeypatch,
                [RuntimeError("a"), RuntimeError("b"), RuntimeError("c")],
                fired=True,
            )

    def test_a_real_error_propagates_without_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(RuntimeError, match="repo gone"):
            self._run(monkeypatch, [RuntimeError("repo gone")], fired=False)

    def test_cancellation_is_never_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(TaskCancelledError):
            self._run(monkeypatch, [TaskCancelledError()], fired=True)


class TestGuardIsOnEveryTransfer:
    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda entry: dl.download_model(entry), id="model"),
            pytest.param(lambda entry: dl.download_mmproj(entry), id="mmproj"),
        ],
    )
    def test_the_transfer_runs_under_the_guard(
        self, call, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Both download entry points must go through the stall guard."""
        guarded: list[str] = []
        gguf = tmp_path / "f.gguf"

        def _fake_guarded(entry: CatalogModel, config: dl.DownloadConfig) -> Path:
            guarded.append(config.filename)
            gguf.write_bytes(b"g")  # lands only when the transfer ran
            return gguf

        monkeypatch.setattr(dl, "_download_with_stall_guard", _fake_guarded)
        monkeypatch.setattr(dl, "_models_dir", lambda: tmp_path)
        monkeypatch.setattr(dl, "resolve_filename", lambda entry: "f.gguf")
        monkeypatch.setattr(dl, "fetch_expected_file_size", lambda *a: 1)
        monkeypatch.setattr(dl, "_resolve_mmproj_filename", lambda *a: "mmproj.gguf")
        monkeypatch.setattr(dl, "_finalize_download", lambda entry, dest, **k: dest)

        call(_entry())

        assert guarded, "the transfer bypassed the stall guard"


class TestWatchTick:
    def test_a_window_under_the_floor_fires_the_abort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        aborts: list[str] = []
        monkeypatch.setattr(dl, "_abort_stalled_transfer", lambda: aborts.append("abort"))
        guard = dl._StallGuard(window_s=0.0, floor_bytes=1024)
        guard._window_start -= 1.0
        guard.pulse(1023)  # a trickle is not progress

        assert guard._keep_watching() is False
        assert guard.fired
        assert aborts == ["abort"]

    def test_a_window_before_expiry_keeps_watching(self) -> None:
        guard = dl._StallGuard(window_s=60.0)
        assert guard._keep_watching() is True
        assert not guard.fired

    def test_a_window_at_the_floor_starts_the_next_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        aborts: list[str] = []
        monkeypatch.setattr(dl, "_abort_stalled_transfer", lambda: aborts.append("abort"))
        guard = dl._StallGuard(window_s=0.0, floor_bytes=1024)
        guard._window_start -= 1.0
        guard.pulse(1024)

        assert guard._keep_watching() is True
        assert not guard.fired
        assert aborts == []
        assert guard._window_bytes == 0

    def test_a_finished_transfer_is_not_aborted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The exit path can win the race against an expiring tick; the tick
        must stand down instead of closing the next transfer's client."""
        aborts: list[str] = []
        monkeypatch.setattr(dl, "_abort_stalled_transfer", lambda: aborts.append("abort"))
        guard = dl._StallGuard(window_s=0.0, floor_bytes=1024)
        guard._window_start -= 1.0
        guard._stop.set()

        assert guard._keep_watching() is False
        assert not guard.fired
        assert aborts == []
