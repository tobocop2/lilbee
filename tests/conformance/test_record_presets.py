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
from typing import Any

import pytest

from tests.conformance.preset_check import Setup, fixture_json, recording
from tests.conformance.record_presets import build_parser, token_from
from tests.conformance.request_set import JSON_ASK

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


class TestFixtureJson:
    def test_the_fixture_holds_the_status_and_body_of_each_error_answer(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with _server("lilbee", seen) as upstream, recording(upstream) as (proxy, tape):
            assert _post(proxy, "/v1/chat/completions")[0] == 401
        setup = Setup(("chat",), {"chat": "m"})
        doc = fixture_json("lilbee", "1", setup, {}, tape)
        assert doc["error_answers"] == [[401, '{"ok": false, "from": "lilbee"}']]


class TestOverflowStep:
    def test_a_served_window_refuses_the_prompt_before_any_request(self, tmp_path) -> None:
        from tests.conformance.preset_check import _overflow, configured, replaying

        setup = Setup(("chat",), {"chat": "m"}, window=40960)
        with (
            replaying([]) as url,
            recording(url) as (proxy, tape),
            configured(proxy, setup, tmp_path),
        ):
            outcome = _overflow(setup)
        assert tape.exchanges == []
        assert outcome.ok
        assert outcome.detail == "kind=context_overflow"

    def test_a_step_with_no_answer_fails_at_the_cap(self, tmp_path, monkeypatch) -> None:
        import time

        from tests.conformance import preset_check

        class Hangs:
            def chat(self, *args: object, **kwargs: object) -> None:
                time.sleep(5)

        monkeypatch.setattr(preset_check, "create_sdk_provider", lambda: Hangs())
        monkeypatch.setattr(preset_check, "_OVERFLOW_CAP_S", 0.2)
        setup = Setup(("chat",), {"chat": "m"})
        started = time.monotonic()
        outcome = preset_check._overflow(setup)
        assert time.monotonic() - started < 2
        assert not outcome.ok
        assert outcome.detail == "no answer within the cap"

    def test_the_window_is_read_from_the_fixture(self, tmp_path) -> None:
        from tests.conformance.preset_check import load_fixture

        doc = fixture_json(
            "s", "1", Setup(("chat",), {"chat": "m"}, window=4096), {}, _empty_tape()
        )
        path = tmp_path / "s.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert load_fixture(path)[0].window == 4096


def _empty_tape():
    from tests.conformance.preset_check import _Tape

    return _Tape()


@contextmanager
def _chat_server(content: str, bodies: list[dict], finish: str = "stop") -> Iterator[str]:
    """A local chat server that keeps each request body and answers *content* to every one."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            size = int(self.headers.get("Content-Length") or 0)
            bodies.append(json.loads(self.rfile.read(size)))
            message = {"role": "assistant", "content": content}
            reply = {"choices": [{"message": message, "finish_reason": finish}]}
            body = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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


_FENCED = '```json\n{"city": "Paris", "pop": 2100000}\n```'
_BARE = '{"city": "Paris", "pop": 2100000}'


def _run_json_step(
    tmp_path, engine: str, content: str, step: str, finish: str = "stop"
) -> tuple[Any, list[dict]]:
    from tests.conformance import preset_check

    bodies: list[dict] = []
    setup = Setup(("chat",), {"chat": "m"}, strict=True, engine=engine)
    server = _chat_server(content, bodies, finish)
    with server as url, preset_check.configured(url, setup, tmp_path):
        outcome = preset_check._STEP_FUNCTIONS[step](setup)
    return outcome, bodies


def _formats(bodies: list[dict]) -> list[Any]:
    return [body["response_format"] for body in bodies if "response_format" in body]


class TestEngineJsonSteps:
    def test_mlx_refuses_a_schema_before_the_chat_request(self, tmp_path) -> None:
        outcome, bodies = _run_json_step(tmp_path, "mlx", _BARE, "json_schema")
        assert outcome.ok
        assert outcome.detail == "refused kind=bad_request"
        assert _formats(bodies) == []
        assert all(body["messages"][0]["content"] != JSON_ASK for body in bodies)

    def test_mlx_drops_json_object_and_the_scan_finds_the_fenced_object(self, tmp_path) -> None:
        outcome, bodies = _run_json_step(tmp_path, "mlx", _FENCED, "json_object")
        assert outcome.ok
        assert "scan found an object=True" in outcome.detail
        assert _formats(bodies) == []
        assert any(body["messages"][0]["content"] == JSON_ASK for body in bodies)

    @pytest.mark.parametrize("engine", ["mlx", "llama.cpp"])
    def test_the_json_ask_goes_out_as_lilbee_sends_one(self, tmp_path, engine: str) -> None:
        """lilbee asks for JSON with thinking off, so a thinking model does not spend the cap."""
        _, bodies = _run_json_step(tmp_path, engine, _FENCED, "json_object")
        ask = next(body for body in bodies if body["messages"][0]["content"] == JSON_ASK)
        assert ask["chat_template_kwargs"] == {"enable_thinking": False}
        assert ask["max_tokens"] == 256

    def test_mlx_passes_json_object_on_a_full_answer_the_scan_cannot_read(self, tmp_path) -> None:
        """MLX promises no JSON: the step judges that lilbee drops the format, not the model."""
        broken = '```json\n{"city": "Paris", "population": 16,000,000}\n```'
        outcome, bodies = _run_json_step(tmp_path, "mlx", broken, "json_object")
        assert outcome.ok
        assert "scan found an object=False" in outcome.detail
        assert _formats(bodies) == []

    def test_mlx_fails_json_object_when_the_answer_is_cut_off(self, tmp_path) -> None:
        outcome, _ = _run_json_step(tmp_path, "mlx", _FENCED, "json_object", finish="length")
        assert not outcome.ok
        assert "finish=length" in outcome.detail

    def test_llama_cpp_sends_the_schema_and_judges_the_reply(self, tmp_path) -> None:
        outcome, bodies = _run_json_step(tmp_path, "llama.cpp", _BARE, "json_schema")
        assert outcome.ok
        assert [f["type"] for f in _formats(bodies)] == ["json_schema"]

    def test_llama_cpp_fails_a_fenced_reply_to_json_object(self, tmp_path) -> None:
        outcome, bodies = _run_json_step(tmp_path, "llama.cpp", _FENCED, "json_object")
        assert not outcome.ok
        assert [f["type"] for f in _formats(bodies)] == ["json_object"]

    def test_the_engine_is_read_from_the_fixture(self, tmp_path) -> None:
        from tests.conformance.preset_check import load_fixture

        setup = Setup(("chat",), {"chat": "m"}, engine="mlx")
        doc = fixture_json("s", "1", setup, {}, _empty_tape())
        path = tmp_path / "s.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert load_fixture(path)[0].engine == "mlx"

    def test_the_parser_takes_an_engine_name_and_refuses_another_word(self) -> None:
        base = ["--server", "s", "--version", "1", "--url", "u", "--roles", "chat"]
        rest = ["--models", "chat=m", "--out", "o.json"]
        assert build_parser().parse_args([*base, *rest, "--engine", "mlx"]).engine == "mlx"
        with pytest.raises(SystemExit):
            build_parser().parse_args([*base, *rest, "--engine", "mlx-lm"])
