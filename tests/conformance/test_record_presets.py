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
from pathlib import Path
from typing import Any

import pytest

from lilbee.providers.base import ChatResult, FinishReason
from tests.conformance.preset_check import Setup, fixture_json, recording
from tests.conformance.record_presets import build_parser, setup_from, token_from
from tests.conformance.request_set import JSON_ASK

_TOKEN = "s3cret-token-value"  # pragma: allowlist secret


def _drain(handler: BaseHTTPRequestHandler) -> None:
    """Read the whole request body, so closing the socket after the answer sends no reset.

    On Windows a close with unread request data resets the connection and drops the answer.
    """
    handler.rfile.read(int(handler.headers.get("Content-Length") or 0))


@contextmanager
def _server(name: str, seen: list[tuple[str, str, str]]) -> Iterator[str]:
    """A local server that logs (name, path, Authorization) and answers 401 without the token."""

    class Handler(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            _drain(self)
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


class TestRerankUrl:
    def test_the_rerank_route_goes_to_the_rerank_server(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with (
            _server("lilbee", seen) as upstream,
            _server("rerank", seen) as rerank,
            recording(upstream, token=_TOKEN, rerank_url=rerank) as (proxy, _),
        ):
            _post_body(proxy, "/v1/rerank", {"query": "q", "documents": ["a"]})
            _post_body(proxy, "/v1/chat/completions", _TEXT_CHAT)
        assert seen == [
            ("rerank", "/v1/rerank", ""),
            ("lilbee", "/v1/chat/completions", f"Bearer {_TOKEN}"),
        ]

    def test_without_a_rerank_server_the_rerank_route_stays_on_the_main_server(self) -> None:
        seen: list[tuple[str, str, str]] = []
        with _server("lilbee", seen) as upstream, recording(upstream, token=_TOKEN) as (proxy, _):
            _post_body(proxy, "/v1/rerank", {"query": "q", "documents": ["a"]})
        assert seen == [("lilbee", "/v1/rerank", f"Bearer {_TOKEN}")]

    def test_the_option_reaches_the_parser(self) -> None:
        base = ["--server", "s", "--version", "1", "--url", "u", "--roles", "chat"]
        args = build_parser().parse_args(
            [*base, "--models", "chat=m", "--out", "o.json", "--rerank-url", "http://r"]
        )
        assert args.rerank_url == "http://r"


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
            _drain(self)
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


class TestReplayMatching:
    @staticmethod
    def _post_to_replay(exchanges: list, sent: dict) -> tuple[int, str]:
        from tests.conformance.preset_check import replaying

        with replaying(exchanges) as url:
            request = urllib.request.Request(  # noqa: S310
                url + "/x", data=json.dumps(sent).encode(), method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as resp:  # noqa: S310
                    return resp.status, resp.read().decode()
            except urllib.error.HTTPError as exc:
                return exc.code, ""

    @pytest.mark.parametrize(
        ("recorded", "sent", "status"),
        [
            ({"m": "a"}, {"m": "a", "temperature": 0, "max_tokens": 1024}, 200),
            ({"m": "a"}, {"m": "a"}, 200),
            ({"m": "a"}, {"m": "b", "temperature": 0}, 599),
            ({"m": "a", "temperature": 0}, {"m": "a", "temperature": 0}, 200),
            ({"m": "a", "temperature": 0}, {"m": "a", "temperature": 0.7}, 599),
            ({"m": "a", "temperature": 0}, {"m": "a"}, 599),
            ({"m": "a", "max_tokens": 256}, {"m": "a", "max_tokens": 1}, 599),
            ({"m": "a", "temperature": 0}, {"m": "a", "temperature": 0, "max_tokens": 5}, 200),
        ],
        ids=[
            "knobs-added",
            "as-recorded",
            "other-question",
            "same-knob-value",
            "other-knob-value",
            "knob-missing-from-sent",
            "other-cap",
            "one-knob-added",
        ],
    )
    def test_a_marked_fixture_drops_a_knob_only_where_it_lacks_it(
        self, recorded: dict, sent: dict, status: int
    ) -> None:
        from tests.conformance.preset_check import UnrecordedKnobsExchange

        exchange = UnrecordedKnobsExchange("POST", "/x", recorded, 200, "application/json", "{}")
        assert self._post_to_replay([exchange], sent)[0] == status

    @pytest.mark.parametrize(
        "sent",
        [{"m": "a", "temperature": 0, "max_tokens": 1024}, {"m": "a", "temperature": 0}],
        ids=["two-knobs-added", "one-knob-added"],
    )
    def test_a_recording_that_is_not_marked_matches_on_the_whole_body(self, sent: dict) -> None:
        from tests.conformance.preset_check import Exchange

        exchange = Exchange("POST", "/x", {"m": "a"}, 200, "application/json", "{}")
        assert self._post_to_replay([exchange], sent)[0] == 599

    @staticmethod
    def _post_chat(path: Path, sent: dict) -> int:
        from tests.conformance.preset_check import load_fixture, replaying

        exchanges = load_fixture(path)[2]
        chat = next(
            e for e in exchanges if e.path.endswith("/chat/completions") and e.status == 200
        )
        with replaying(exchanges) as url:
            request = urllib.request.Request(  # noqa: S310
                url + chat.path, data=json.dumps(sent).encode(), method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as resp:  # noqa: S310
                    return resp.status
            except urllib.error.HTTPError as exc:
                return exc.code

    _FIXTURE_DIR = Path(__file__).with_name("fixtures")
    _PINNED_FIXTURES = (
        "engines/llama.cpp-cpu.json",
        "engines/llama.cpp-cpu-inline.json",
        "engines/mistralrs-cpu.json",
        "presets/lemonade.json",
        "presets/mistralrs-chat.json",
    )

    @pytest.mark.parametrize("name", _PINNED_FIXTURES)
    def test_a_recorded_fixture_with_pins_misses_on_a_wrong_knob(self, name: str) -> None:
        from tests.conformance.preset_check import (
            SAMPLING_KNOBS_KEY,
            load_fixture,
        )

        path = self._FIXTURE_DIR / name
        doc = load_fixture(path)[3]
        assert SAMPLING_KNOBS_KEY not in doc, "a re-recorded fixture carries no marker"
        chat = next(
            e
            for e in load_fixture(path)[2]
            if e.path.endswith("/chat/completions") and e.status == 200
        )
        assert chat.request["temperature"] == 0
        assert chat.request["seed"] == 7
        assert self._post_chat(path, chat.request) == 200
        assert self._post_chat(path, {**chat.request, "seed": 8}) == 599
        assert self._post_chat(path, {**chat.request, "temperature": 0.9}) == 599

    def test_a_marked_fixture_answers_any_seed(self) -> None:
        from tests.conformance.preset_check import (
            KNOBS_NOT_RECORDED,
            SAMPLING_KNOBS_KEY,
            load_fixture,
        )

        path = self._FIXTURE_DIR / "engines/llama.cpp-cuda.json"
        assert load_fixture(path)[3][SAMPLING_KNOBS_KEY] == KNOBS_NOT_RECORDED
        chat = next(
            e
            for e in load_fixture(path)[2]
            if e.path.endswith("/chat/completions") and e.status == 200
        )
        for seed in (7, 8):
            sent = {**chat.request, "seed": seed, "temperature": 0}
            assert self._post_chat(path, sent) == 200

    def test_a_wrong_temperature_does_not_take_the_reply_of_another_recording(self) -> None:
        from tests.conformance.preset_check import Exchange

        template = Exchange(
            "POST", "/x", {"m": "a", "temperature": 0, "max_tokens": 1024}, 200, "text/plain", "t"
        )
        named = Exchange("POST", "/x", {"m": "a", "max_tokens": 1024}, 200, "text/plain", "n")
        right = {"m": "a", "temperature": 0, "max_tokens": 1024}
        wrong = {"m": "a", "temperature": 0.7, "max_tokens": 1024}
        assert self._post_to_replay([template, named], right) == (200, "t")
        assert self._post_to_replay([template, named], wrong)[0] == 599

    def test_a_fixture_file_marks_its_exchanges(self, tmp_path) -> None:
        from tests.conformance.preset_check import (
            KNOBS_NOT_RECORDED,
            SAMPLING_KNOBS_KEY,
            Exchange,
            UnrecordedKnobsExchange,
            load_fixture,
        )

        doc = {
            "server": "s",
            "roles": [],
            "models": {},
            "rerank_wire": "v1",
            "embed_dim": 1,
            "results": {},
            "exchanges": [
                {
                    "method": "POST",
                    "path": "/x",
                    "request": {},
                    "status": 200,
                    "content_type": "text/plain",
                    "body": "",
                }
            ],
        }
        plain = tmp_path / "plain.json"
        plain.write_text(json.dumps(doc), encoding="utf-8")
        marked = tmp_path / "marked.json"
        marked.write_text(
            json.dumps({**doc, SAMPLING_KNOBS_KEY: KNOBS_NOT_RECORDED}), encoding="utf-8"
        )
        assert type(load_fixture(plain)[2][0]) is Exchange
        assert type(load_fixture(marked)[2][0]) is UnrecordedKnobsExchange


class TestOverflowStep:
    def test_a_served_window_refuses_the_prompt_before_any_request(self, tmp_path) -> None:
        from tests.conformance.preset_check import _overflow, configured, replaying

        setup = Setup(("chat",), {"chat": "m"}, window=4096)
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
def _chat_server(
    content: str, bodies: list[dict], finish: str = "stop", ask_status: int = 200
) -> Iterator[str]:
    """A local chat server that keeps each request body and answers *content* to every one.

    The JSON ask gets *ask_status*; every other request gets 200.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            size = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(size))
            bodies.append(request)
            message = {"role": "assistant", "content": content}
            reply = {"choices": [{"message": message, "finish_reason": finish}]}
            body = json.dumps(reply).encode()
            asked = request.get("messages", [{}])[0].get("content") == JSON_ASK
            self.send_response(ask_status if asked else 200)
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
    tmp_path, engine: str, content: str, step: str, finish: str = "stop", ask_status: int = 200
) -> tuple[Any, list[dict]]:
    from tests.conformance import preset_check

    bodies: list[dict] = []
    setup = Setup(("chat",), {"chat": "m"}, strict=True, engine=engine)
    server = _chat_server(content, bodies, finish, ask_status)
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

    def test_a_server_error_on_the_schema_ask_fails_the_step(self, tmp_path) -> None:
        """Only lilbee's own refusal on an engine without the format passes; an error fails."""
        outcome, bodies = _run_json_step(
            tmp_path, "llama.cpp", _BARE, "json_schema", ask_status=500
        )
        assert not outcome.ok
        assert outcome.detail.startswith("refused kind=")
        assert outcome.detail != "refused kind=bad_request"
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


_OVMS_OVERFLOW = "Number of prompt tokens: 60009 exceeds model max length: 40960"
_OVMS_MODELS = {"chat": "chat-m", "embed": "embed-m", "vision": "vision-m"}


def _ovms_chat_reply(request: dict, finish: str) -> tuple[int, dict]:
    """The chat reply of OVMS 2026.4.1, which has no ``id`` field."""
    from tests.conformance.fake_engine import image_colour
    from tests.conformance.request_set import CONVERSATION, OVERSIZE, TOOL_NAME

    content = request["messages"][0]["content"]
    message: dict[str, Any] = {"role": "assistant", "content": "42", "reasoning_content": "17+25"}
    if content == OVERSIZE:
        return 400, {"error": _OVMS_OVERFLOW}
    if request["messages"] == CONVERSATION:
        message = {"role": "assistant", "content": "Rome"}
    elif request.get("tools"):
        call = {"name": TOOL_NAME, "arguments": '{"city": "Paris"}'}
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": call}],
        }
    elif isinstance(content, list):
        url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
        message = {"role": "assistant", "content": image_colour(url)}
    return 200, {"choices": [{"index": 0, "message": message, "finish_reason": finish}]}


def _ovms_embed_reply(request: dict) -> dict:
    """The embeddings reply of OVMS 2026.4.1, which has no ``model`` field."""
    import base64
    import struct

    texts = request["input"] if isinstance(request["input"], list) else [request["input"]]
    data = []
    for index in range(len(texts)):
        vector = [1.0 if slot == index else 0.0 for slot in range(3)]
        raw = base64.b64encode(struct.pack("<3f", *vector)).decode()
        encoded = raw if request.get("encoding_format") == "base64" else vector
        data.append({"object": "embedding", "embedding": encoded, "index": index})
    return {"object": "list", "data": data}


@contextmanager
def _ovms_server(bodies: list[dict], finish: str = "stop") -> Iterator[str]:
    """A local server that answers as OVMS 2026.4.1 does and keeps each request body."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            bodies.append(request)
            status, reply = 404, {"error": "Unsupported endpoint"}
            if self.path.endswith("/chat/completions"):
                status, reply = _ovms_chat_reply(request, finish)
            elif self.path.endswith("/embeddings"):
                status, reply = 200, _ovms_embed_reply(request)
            body = json.dumps(reply).encode()
            self.send_response(status)
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


def _run_ovms_step(
    tmp_path, step: str, finish: str = "stop", bodies: list[dict] | None = None, **fields: Any
) -> Any:
    from tests.conformance import preset_check

    tmp_path.mkdir(parents=True, exist_ok=True)
    setup = Setup(
        ("chat", "embed", "vision"),
        _OVMS_MODELS,
        embed_dim=3,
        strict=True,
        engine="ovms",
        **fields,
    )
    with (
        _ovms_server([] if bodies is None else bodies, finish) as url,
        preset_check.configured(url, setup, tmp_path),
    ):
        return preset_check._STEP_FUNCTIONS[step](setup)


def _asked(bodies: list[dict]) -> list[dict]:
    """The requests the step asked for, without the template probe the client sends first."""
    return [body for body in bodies if body.get("max_tokens") != 1]


class TestOvmsRepliesWithoutIdOrModel:
    """OVMS sends chat with no ``id`` and embeddings with no ``model``; the SDK refuses both."""

    @pytest.mark.parametrize("step", ["template", "tool", "embed", "vision", "overflow"])
    def test_the_step_reads_the_reply_of_ovms(self, tmp_path, step: str) -> None:
        outcome = _run_ovms_step(tmp_path, step)
        assert outcome.ok, outcome.detail

    def test_the_overflow_step_reads_the_ovms_text_as_a_context_overflow(self, tmp_path) -> None:
        outcome = _run_ovms_step(tmp_path, "overflow")
        assert outcome.detail == "kind=context_overflow"

    def test_the_tool_request_asks_for_auto_and_the_template_request_names_no_choice(
        self, tmp_path
    ) -> None:
        tool_bodies: list[dict] = []
        template_bodies: list[dict] = []
        _run_ovms_step(tmp_path / "tool", "tool", bodies=tool_bodies)
        _run_ovms_step(tmp_path / "template", "template", bodies=template_bodies)
        assert [body["tool_choice"] for body in _asked(tool_bodies)] == ["auto"]
        assert all("tool_choice" not in body for body in template_bodies)

    def test_the_cap_goes_out_as_max_tokens_the_name_that_ovms_reads(self, tmp_path) -> None:
        bodies: list[dict] = []
        _run_ovms_step(tmp_path, "template", bodies=bodies)
        assert [body["max_tokens"] for body in _asked(bodies)] == [1024]
        assert all("num_predict" not in body for body in bodies)

    def test_the_vision_request_caps_the_answer(self, tmp_path) -> None:
        bodies: list[dict] = []
        _run_ovms_step(tmp_path, "vision", bodies=bodies)
        assert [body["max_tokens"] for body in _asked(bodies)] == [64, 64]
        assert {body["model"] for body in bodies} == {"vision-m"}

    def test_the_embed_request_goes_to_the_embed_model(self, tmp_path) -> None:
        bodies: list[dict] = []
        _run_ovms_step(tmp_path, "embed", bodies=bodies)
        assert [body["model"] for body in bodies] == ["embed-m"]

    def test_another_engine_keeps_the_sdk_client(self, tmp_path, monkeypatch) -> None:
        from tests.conformance import preset_check

        seen: list[str] = []

        class Sdk:
            def chat(self, messages: Any, **kwargs: Any) -> ChatResult:
                seen.append(kwargs["model"])
                return ChatResult("Rome", (), FinishReason.STOP)

        monkeypatch.setattr(preset_check, "create_sdk_provider", lambda: Sdk())
        setup = Setup(("chat",), {"chat": "m"}, engine="llama.cpp")
        with preset_check.configured("http://127.0.0.1:9", setup, tmp_path):
            outcome = preset_check._template(setup)
        assert outcome.ok
        assert seen == [f"{setup.preset}/m"]

    def test_the_named_step_sends_the_rerank_role_for_ovms(self) -> None:
        from tests.conformance import preset_check

        roles = ("chat", "embed", "rerank")
        assert preset_check._named_role(Setup(roles, {}, engine="ovms")) == "rerank"
        assert preset_check._named_role(Setup(roles, {}, engine="llama.cpp")) == "chat"
        assert preset_check._named_role(Setup(("chat",), {}, engine="ovms")) == "chat"


class TestReasoningTokens:
    def test_a_bound_goes_out_as_max_tokens_and_accepts_an_answer_the_bound_cut(
        self, tmp_path
    ) -> None:
        outcome = _run_ovms_step(tmp_path, "reasoning", finish="length", reasoning_tokens=64)
        assert outcome.ok, outcome.detail

    def test_no_bound_sends_no_max_tokens_and_fails_a_cut_answer(self, tmp_path) -> None:
        outcome = _run_ovms_step(tmp_path, "reasoning", finish="length")
        assert not outcome.ok
        assert "finish is length" in outcome.detail

    def test_the_bound_is_on_the_wire(self, tmp_path) -> None:
        from tests.conformance import preset_check

        bodies: list[dict] = []
        setup = Setup(("chat",), {"chat": "default_model"}, engine="ovms", reasoning_tokens=64)
        with _ovms_server(bodies) as url, preset_check.configured(url, setup, tmp_path):
            preset_check._reasoning_text(setup)
        assert [body["max_tokens"] for body in bodies] == [64]

    def test_the_bound_is_read_from_the_fixture_and_the_parser(self, tmp_path) -> None:
        from tests.conformance.preset_check import load_fixture

        setup = Setup(("chat",), {"chat": "m"}, reasoning_tokens=64)
        path = tmp_path / "s.json"
        doc = fixture_json("s", "1", setup, {}, _empty_tape())
        path.write_text(json.dumps(doc), encoding="utf-8")
        assert load_fixture(path)[0].reasoning_tokens == 64
        base = ["--server", "s", "--version", "1", "--url", "u", "--roles", "chat"]
        rest = ["--models", "chat=m", "--out", "o.json", "--reasoning-tokens", "64"]
        assert build_parser().parse_args([*base, *rest]).reasoning_tokens == 64
        no_flag = build_parser().parse_args([*base, *rest[:-2]])
        assert no_flag.reasoning_tokens == 0
        assert setup_from(build_parser().parse_args([*base, *rest])).reasoning_tokens == 64
