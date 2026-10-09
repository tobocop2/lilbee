"""Replay: every answer comes from a record, as recorded, and what is not recorded is counted."""

from __future__ import annotations

import time
from http.client import HTTPConnection, HTTPResponse
from urllib.parse import urlsplit

import pytest
from tools.qa.crawl_parity.corpus import Alternate, AlternateWhen, Corpus, Record, Response
from tools.qa.crawl_parity.replay import Replay

HTML = (("Content-Type", "text/html; charset=shift_jis"), ("X-Recorded", "yes"))
BODY = "日本語".encode("shift_jis")


def corpus(*records: Record) -> Corpus:
    return Corpus("test", {record.path: record for record in records})


def get(url: str, cookie: str = "", method: str = "GET") -> HTTPResponse:
    parts = urlsplit(url)
    connection = HTTPConnection(parts.hostname or "", parts.port, timeout=10)
    target = parts.path + (f"?{parts.query}" if parts.query else "")
    connection.request(method, target, headers={"Cookie": cookie} if cookie else {})
    return connection.getresponse()


def test_status_headers_and_body_bytes_are_served_as_recorded() -> None:
    with Replay(corpus(Record("/page", Response(203, HTML, BODY)))) as replay:
        response = get(replay.url("/page"))
        assert response.status == 203
        assert response.headers["Content-Type"] == "text/html; charset=shift_jis"
        assert response.headers["X-Recorded"] == "yes"
        assert response.read() == BODY


def test_a_redirect_is_served_and_not_followed_by_the_server() -> None:
    record = Record("/old", Response(301, (("Location", "{{ORIGIN}}/new"),)))
    with Replay(corpus(record)) as replay:
        response = get(replay.url("/old"))
        assert response.status == 301
        assert response.headers["Location"] == f"{replay.origin}/new"


def test_another_host_is_served_under_the_other_origin() -> None:
    record = Record("/away", Response(302, (("Location", "{{OTHER_ORIGIN}}/x"),)))
    with Replay(corpus(record)) as replay:
        assert get(replay.url("/away")).headers["Location"] == f"http://localhost:{replay.port}/x"
        assert replay.origin == f"http://127.0.0.1:{replay.port}"


def test_the_origin_is_put_into_a_body_only_when_the_record_asks() -> None:
    body = b'<a href="{{ORIGIN}}/x">x</a>'
    with Replay(
        corpus(
            Record("/with", Response(200, HTML, body), substitute_origin=True),
            Record("/without", Response(200, HTML, body)),
        )
    ) as replay:
        assert replay.origin.encode() in get(replay.url("/with")).read()
        assert get(replay.url("/without")).read() == body


def test_an_address_that_is_not_recorded_is_refused_and_counted() -> None:
    with Replay(corpus(Record("/page", Response(200, HTML, BODY)))) as replay:
        assert get(replay.url("/page")).status == 200
        assert replay.unrecorded() == []
        # 599 is no status a recorded site gives, so a crawler cannot take it for a page.
        assert get(replay.url("/nowhere")).status == 599
        assert replay.unrecorded() == ["/nowhere"]
        replay.reset()
        assert replay.unrecorded() == [] and replay.requests() == []


def test_a_query_selects_its_own_record_before_the_bare_path() -> None:
    with Replay(
        corpus(
            Record("/q", Response(200, HTML, b"bare")),
            Record("/q?a=1", Response(200, HTML, b"one")),
        )
    ) as replay:
        assert get(replay.url("/q?a=1")).read() == b"one"
        assert get(replay.url("/q?b=2")).read() == b"bare"


def test_a_record_answers_without_its_cookie_from_its_alternate() -> None:
    alternate = Alternate(AlternateWhen.NO_COOKIE, Response(403, HTML, b"no"), cookie="session=ok")
    with Replay(
        corpus(Record("/need", Response(200, HTML, b"yes"), alternate=alternate))
    ) as replay:
        assert get(replay.url("/need")).status == 403
        assert get(replay.url("/need"), cookie="session=ok").read() == b"yes"
        assert [request.cookie for request in replay.requests()] == ["", "session=ok"]


def test_the_first_requests_get_the_alternate_and_a_reset_starts_again() -> None:
    alternate = Alternate(AlternateWhen.FIRST_REQUESTS, Response(429, HTML, b"wait"), count=2)
    with Replay(
        corpus(Record("/flaky", Response(200, HTML, b"ok"), alternate=alternate))
    ) as replay:
        assert [get(replay.url("/flaky")).status for _ in range(3)] == [429, 429, 200]
        replay.reset()
        assert get(replay.url("/flaky")).status == 429


def test_head_has_the_headers_and_no_body() -> None:
    with Replay(corpus(Record("/page", Response(200, HTML, BODY)))) as replay:
        response = get(replay.url("/page"), method="HEAD")
        assert response.headers["Content-Length"] == str(len(BODY))
        assert response.read() == b""


def test_a_status_with_no_body_sends_none() -> None:
    with Replay(corpus(Record("/empty", Response(204, ())))) as replay:
        response = get(replay.url("/empty"))
        assert response.status == 204
        assert response.headers["Content-Length"] is None


@pytest.mark.parametrize(("record_delay", "every_delay"), [(150, 0.0), (0, 150.0)])
def test_a_delay_is_waited_before_the_answer(record_delay: int, every_delay: float) -> None:
    with Replay(
        corpus(Record("/slow", Response(200, HTML, b"x"), delay_ms=record_delay))
    ) as replay:
        replay.delay_every_response(every_delay)
        started = time.monotonic()
        assert get(replay.url("/slow")).read() == b"x"
        assert time.monotonic() - started >= 0.15
