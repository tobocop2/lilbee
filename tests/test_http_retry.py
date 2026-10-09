"""The live catalog check retries a dropped connection and never retries a wrong answer."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from tests import _http_retry
from tests.integration import test_catalog_integration as live

_LISTING = {"siblings": [{"rfilename": "README.md"}, {"rfilename": "m-Q4_K_M.gguf"}]}


class _Script:
    """A fake transport that plays one scripted outcome per request and counts them."""

    def __init__(self, *outcomes: httpx.Response | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


@pytest.fixture(autouse=True)
def no_pause(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    pauses: list[float] = []
    monkeypatch.setattr(_http_retry.time, "sleep", pauses.append)
    return pauses


def _reset() -> httpx.ConnectError:
    return httpx.ConnectError("Connection reset by peer")


def test_a_reset_then_success_passes(no_pause: list[float]) -> None:
    script = _Script(_reset(), httpx.Response(200, json=_LISTING))
    assert live.repo_gguf_files("o/r", script.transport) == ["m-Q4_K_M.gguf"]
    assert script.calls == 2
    assert len(no_pause) == 1


def test_a_handshake_timeout_then_a_5xx_then_success_passes() -> None:
    script = _Script(
        httpx.ConnectTimeout("handshake timed out"),
        httpx.Response(503),
        httpx.Response(200, json=_LISTING),
    )
    assert live.repo_gguf_files("o/r", script.transport) == ["m-Q4_K_M.gguf"]
    assert script.calls == 3


def test_a_404_fails_with_no_retry(no_pause: list[float]) -> None:
    script = _Script(httpx.Response(404), httpx.Response(200, json=_LISTING))
    with pytest.raises(httpx.HTTPStatusError, match="404"):
        live.repo_gguf_files("o/gone", script.transport)
    assert script.calls == 1
    assert no_pause == []


def test_a_reset_on_every_attempt_fails_after_the_bound() -> None:
    script = _Script(_reset())
    with pytest.raises(httpx.ConnectError, match="Connection reset by peer"):
        live.repo_gguf_files("o/r", script.transport)
    assert script.calls == _http_retry.ATTEMPTS


def test_a_5xx_on_every_attempt_fails_after_the_bound() -> None:
    script = _Script(httpx.Response(502))
    with pytest.raises(httpx.HTTPStatusError, match="502"):
        live.repo_gguf_files("o/r", script.transport)
    assert script.calls == _http_retry.ATTEMPTS


def test_a_repo_without_a_gguf_lists_nothing() -> None:
    script = _Script(httpx.Response(200, json={"siblings": [{"rfilename": "README.md"}]}))
    assert live.repo_gguf_files("o/r", script.transport) == []


def test_the_pick_sweep_survives_a_reset_but_not_a_missing_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ok = httpx.Response(200, json=_LISTING)
    script = _Script(_reset(), ok, httpx.Response(404))
    real_client = httpx.Client
    monkeypatch.setattr(
        _http_retry.httpx,
        "Client",
        lambda **kw: real_client(**{**kw, "transport": script.transport}),
    )
    picks = [SimpleNamespace(hf_repo="o/a"), SimpleNamespace(hf_repo="o/gone")]
    with pytest.raises(httpx.HTTPStatusError, match="404"):
        live.test_every_pick_has_a_gguf(picks)
    assert script.calls == 3
