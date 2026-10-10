"""The live pick sweep asks the retrying client and still fails on a wrong answer."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from tests.integration import test_catalog_integration as live

_WITH_GGUF = {"siblings": [{"rfilename": "README.md"}, {"rfilename": "m-Q4_K_M.gguf"}]}
_WITHOUT_GGUF = {"siblings": [{"rfilename": "README.md"}, {"rfilename": "model.safetensors"}]}


class _Host:
    """Stands in for the retrying client: answers by repo and records each call."""

    def __init__(self, answers: dict[str, tuple[int, dict]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append((method, url, kwargs))
        status, body = self.answers[url.removeprefix(f"{live.HF_API_URL}/")]
        return httpx.Response(status, json=body, request=httpx.Request(method, url))


def _picks(*repos: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(hf_repo=r) for r in repos]


def _install(monkeypatch: pytest.MonkeyPatch, answers: dict[str, tuple[int, dict]]) -> _Host:
    host = _Host(answers)
    monkeypatch.setattr(live, "http_backoff", host)
    return host


def test_the_sweep_asks_the_retrying_client_once_per_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _install(monkeypatch, {"o/a": (200, _WITH_GGUF), "o/b": (200, _WITH_GGUF)})
    live.test_every_pick_has_a_gguf(_picks("o/a", "o/b"))
    assert [(m, u.removeprefix(f"{live.HF_API_URL}/")) for m, u, _ in host.calls] == [
        ("GET", "o/a"),
        ("GET", "o/b"),
    ]
    assert all(kw["max_retries"] >= 1 for _, _, kw in host.calls)


def test_a_pick_whose_listing_has_no_gguf_fails_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, {"o/a": (200, _WITH_GGUF), "o/bare": (200, _WITHOUT_GGUF)})
    with pytest.raises(AssertionError, match=r"o/bare has no \.gguf files"):
        live.test_every_pick_has_a_gguf(_picks("o/a", "o/bare"))


def test_a_pick_the_host_no_longer_serves_fails_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, {"o/gone": (404, {})})
    with pytest.raises(httpx.HTTPStatusError, match="404"):
        live.test_every_pick_has_a_gguf(_picks("o/gone"))
