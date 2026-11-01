"""The recorder reaches a server that needs a bearer token and keeps the token off the tape."""

from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.conformance.preset_check import recording
from tests.conformance.record_presets import build_parser, token_from

_TOKEN = "s3cret-token-value"  # pragma: allowlist secret


@contextmanager
def _server(name: str, seen: list[tuple[str, str, str]]) -> Iterator[str]:
    """A local server that logs (name, path, Authorization) and answers 401 without the token."""

    class Handler(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            auth = self.headers.get("Authorization", "")
            seen.append((name, self.path, auth))
            ok = name != "lilbee" or auth == f"Bearer {_TOKEN}"
            body = json.dumps({"ok": ok, "from": name}).encode()
            self.send_response(200 if ok else 401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _answer  # noqa: N815

        def log_message(self, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _post(base: str, path: str) -> tuple[int, dict]:
    request = urllib.request.Request(base + path, data=b"{}", method="POST")  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


class TestRecordingProxy:
    def test_without_a_token_the_server_answers_401(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with _server("lilbee", seen) as upstream, recording(upstream) as (proxy, _tape):
            assert _post(proxy, "/v1/chat/completions")[0] == 401

    def test_the_token_reaches_the_server_and_never_the_tape(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with (
            _server("lilbee", seen) as upstream,
            recording(upstream, token=_TOKEN) as (proxy, tape),
        ):
            assert _post(proxy, "/v1/chat/completions")[0] == 200
        assert seen == [("lilbee", "/v1/chat/completions", f"Bearer {_TOKEN}")]
        assert tape.exchanges
        assert _TOKEN not in json.dumps([e.as_json() for e in tape.exchanges])

    def test_the_embeddings_route_goes_to_the_embed_server_without_the_token(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with (
            _server("lilbee", seen) as upstream,
            _server("proxy", seen) as embed,
            recording(upstream, token=_TOKEN, embed_url=embed) as (proxy, tape),
        ):
            assert _post(proxy, "/v1/embeddings") == (200, {"ok": True, "from": "proxy"})
            assert _post(proxy, "/v1/chat/completions")[0] == 200
        assert seen == [
            ("proxy", "/v1/embeddings", ""),
            ("lilbee", "/v1/chat/completions", f"Bearer {_TOKEN}"),
        ]
        assert [e.path for e in tape.exchanges] == ["/v1/embeddings", "/v1/chat/completions"]

    def test_without_an_embed_server_the_embeddings_route_stays_on_the_main_server(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with _server("lilbee", seen) as upstream, recording(upstream, token=_TOKEN) as (proxy, _):
            _post(proxy, "/v1/embeddings")
        assert seen == [("lilbee", "/v1/embeddings", f"Bearer {_TOKEN}")]


def _post_body(base: str, path: str, body: dict) -> int:
    request = urllib.request.Request(  # noqa: S310
        base + path, data=json.dumps(body).encode(), method="POST"
    )
    with urllib.request.urlopen(request, timeout=10) as resp:  # noqa: S310
        return resp.status


_IMAGE_CHAT = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "colour?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
}
_TEXT_CHAT = {"messages": [{"role": "user", "content": "hi"}]}


class TestVisionUrl:
    def test_a_chat_request_with_an_image_goes_to_the_vision_server(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with (
            _server("lilbee", seen) as upstream,
            _server("vision", seen) as vision,
            recording(upstream, token=_TOKEN, vision_url=vision) as (proxy, _),
        ):
            _post_body(proxy, "/v1/chat/completions", _IMAGE_CHAT)
            _post_body(proxy, "/v1/chat/completions", _TEXT_CHAT)
        assert seen == [
            ("vision", "/v1/chat/completions", ""),
            ("lilbee", "/v1/chat/completions", f"Bearer {_TOKEN}"),
        ]

    def test_without_a_vision_server_an_image_request_stays_on_the_main_server(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with _server("lilbee", seen) as upstream, recording(upstream, token=_TOKEN) as (proxy, _):
            _post_body(proxy, "/v1/chat/completions", _IMAGE_CHAT)
        assert seen == [("lilbee", "/v1/chat/completions", f"Bearer {_TOKEN}")]

    def test_the_option_reaches_the_parser(self) -> None:
        base = ["--server", "s", "--version", "1", "--url", "u", "--roles", "chat"]
        args = build_parser().parse_args(
            [*base, "--models", "chat=m", "--out", "o.json", "--vision-url", "http://v"]
        )
        assert args.vision_url == "http://v"


class TestTokenOption:
    @staticmethod
    def _args(*extra: str) -> argparse.Namespace:
        base = ["--server", "s", "--version", "1", "--url", "u", "--roles", "chat"]
        return build_parser().parse_args([*base, "--models", "chat=m", "--out", "o.json", *extra])

    def test_no_option_sends_no_token(self) -> None:
        assert token_from(self._args()) is None

    def test_the_option_reads_the_named_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REC_TOKEN", _TOKEN)
        assert token_from(self._args("--token-env", "REC_TOKEN")) == _TOKEN

    def test_an_empty_variable_is_refused_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REC_TOKEN", raising=False)
        with pytest.raises(SystemExit, match="--token-env REC_TOKEN"):
            token_from(self._args("--token-env", "REC_TOKEN"))


@contextmanager
def _redirector(target: str) -> Iterator[str]:
    """A local server that answers every request with a 302 to *target*."""

    class Handler(BaseHTTPRequestHandler):
        def _redirect(self) -> None:
            self.send_response(302)
            self.send_header("Location", target + self.path)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_POST = _redirect  # noqa: N815

        def log_message(self, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class TestRedirects:
    def test_a_redirect_to_another_origin_never_carries_the_token(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with (
            _server("elsewhere", seen) as elsewhere,
            _redirector(elsewhere) as upstream,
            recording(upstream, token=_TOKEN) as (proxy, tape),
        ):
            status, _ = _post(proxy, "/v1/chat/completions")
        assert status == 302
        assert seen == []
        assert [e.status for e in tape.exchanges] == [302]

    def test_a_same_origin_embed_url_gets_the_token(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with (
            _server("lilbee", seen) as upstream,
            recording(upstream, token=_TOKEN, embed_url=upstream) as (proxy, _),
        ):
            assert _post(proxy, "/v1/embeddings")[0] == 200
        assert seen == [("lilbee", "/v1/embeddings", f"Bearer {_TOKEN}")]
