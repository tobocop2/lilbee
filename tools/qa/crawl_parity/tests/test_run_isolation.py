"""No run reads what an earlier run left, and a signature does not change with the replay port."""

from __future__ import annotations

import sys
from http.client import HTTPConnection
from pathlib import Path

import pytest
from tools.qa.crawl_parity import leftover, parity, pipeline, retrieval
from tools.qa.crawl_parity.corpus import Alternate, AlternateWhen, Corpus, Record, Response, Seed
from tools.qa.crawl_parity.model import Difference, Kind, Layer, Mode, Page, Side
from tools.qa.crawl_parity.replay import Replay
from tools.qa.crawl_parity.sides import (
    NEUTRAL_ORIGIN,
    CrawlRequest,
    SideConfig,
    neutral_origin,
    origin_of,
    run_crawl,
)
from tools.qa.crawl_parity.thresholds import load_thresholds
from tools.qa.crawl_parity.truth import BrowserName

DRIVERS = Path(__file__).parent / "drivers"
HTML = (("Content-Type", "text/html; charset=utf-8"),)


def site(body: str) -> Corpus:
    page = Response(200, HTML, f'<a href="/s/one">one</a><p>{body}</p>'.encode())
    one = Response(200, HTML, b"<p>see {{ORIGIN}}/s/ in plain words</p>")
    records = {"/s/": Record("/s/", page), "/s/one": Record("/s/one", one, substitute_origin=True)}
    return Corpus("site", records, (Seed("f", "/s/", (Mode.HTTP,)),))


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(leftover, "GRACE_SECONDS", 0.1)


def crawl(replay: Replay, work: Path, driver: str = "fake_good.py") -> dict[str, Page]:
    config = SideConfig(
        Side.CANDIDATE, "x", {Layer.CRAWLER: Path(sys.executable)}, str(DRIVERS / driver), ""
    )
    request = CrawlRequest(replay.url("/s/"), Mode.HTTP)
    return run_crawl(config, Layer.CRAWLER, request, work).pages


def test_a_second_crawl_in_the_same_work_directory_does_not_read_the_first(tmp_path: Path) -> None:
    with Replay(site("first")) as replay:
        first = crawl(replay, tmp_path / "work")
    with Replay(site("second")) as replay:
        # The lossy stand-in writes no markdown for a page named two; here it loses nothing.
        second = crawl(replay, tmp_path / "work", "fake_lossy.py")
    assert "first" in (first["/s/"].markdown or "")
    assert "second" in (second["/s/"].markdown or "") and "first" not in (
        second["/s/"].markdown or ""
    )
    assert len(list(tmp_path.glob("work.*"))) == 2


def test_the_replay_origin_is_written_as_one_name_in_every_page(tmp_path: Path) -> None:
    with Replay(site("body")) as replay:
        pages = crawl(replay, tmp_path / "work")
        origin = replay.origin
    markdown = pages["/s/one"].markdown or ""
    assert origin not in markdown
    assert f"see {NEUTRAL_ORIGIN}/s/ in plain" in markdown
    assert markdown.count("words") == 1
    assert origin_of(f"{origin}/s/one?x=1") == origin


def test_neutral_origin_changes_only_the_origin() -> None:
    first = neutral_origin("see http://127.0.0.1:5001/a for more", "http://127.0.0.1:5001")
    second = neutral_origin("see http://127.0.0.1:5002/a for more", "http://127.0.0.1:5002")
    assert first == second == f"see {NEUTRAL_ORIGIN}/a for more"
    assert neutral_origin("see http://127.0.0.1:5002/a", "http://127.0.0.1:5001") != first
    assert neutral_origin(None, "http://127.0.0.1:5001") is None


def test_an_index_is_built_in_a_directory_no_earlier_run_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / retrieval.INDEX_DRIVER
    script.write_text(
        "import json, pathlib, sys\n"
        "data = pathlib.Path(sys.argv[sys.argv.index('--data') + 1])\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "docs = [f.name for f in (data / 'documents').iterdir()]\n"
        "out.write_text(json.dumps({'added': len(docs), 'failed': [], 'skipped': [],\n"
        "    'index_seconds': 0.0, 'search_seconds': 0.0, 'answers': {}}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval, "DRIVERS_DIR", tmp_path)
    config = retrieval.RetrievalConfig(Path(sys.executable), "model")
    two = {"/a": Page("u/a", "alpha"), "/b": Page("u/b", "beta")}
    work = tmp_path / "index"
    assert retrieval.measure(two, [], config, top_k=5, work=work).indexed_pages == 2
    assert retrieval.measure({"/a": two["/a"]}, [], config, top_k=5, work=work).indexed_pages == 1


def get_status(replay: Replay, path: str) -> int:
    connection = HTTPConnection("127.0.0.1", replay.port, timeout=10)
    connection.request("GET", path)
    return connection.getresponse().status


def test_a_patient_reader_gets_the_page_that_first_requests_do_not() -> None:
    flaky = Alternate(AlternateWhen.FIRST_REQUESTS, Response(429, HTML, b"wait"), count=2)
    need = Alternate(AlternateWhen.NO_COOKIE, Response(403, HTML, b"no"), cookie="s=ok")
    corpus = Corpus(
        "x",
        {
            "/flaky": Record("/flaky", Response(200, HTML, b"ok"), alternate=flaky),
            "/need": Record("/need", Response(200, HTML, b"ok"), alternate=need),
        },
    )
    with Replay(corpus) as replay:
        replay.answer_as_to_a_patient_reader(True)
        assert get_status(replay, "/flaky") == 200
        assert get_status(replay, "/need") == 403
        replay.answer_as_to_a_patient_reader(False)
        assert get_status(replay, "/flaky") == 429


def test_truth_is_taken_seeds_first_as_a_patient_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    flaky = Alternate(AlternateWhen.FIRST_REQUESTS, Response(429, HTML, b"wait"), count=1)
    corpus = Corpus(
        "x",
        {
            "/a": Record("/a", Response(200, HTML, b"a"), alternate=flaky),
            "/seed/": Record("/seed/", Response(200, HTML, b"seed")),
        },
        (Seed("f", "/seed/", (Mode.HTTP,)),),
    )

    def capture(targets: dict[str, object], *_rest: object) -> dict[str, object]:
        seen["order"] = list(targets)
        seen["status"] = get_status(replay, "/a")
        return {}

    monkeypatch.setattr(pipeline, "capture", capture)
    with Replay(corpus) as replay:
        plan = pipeline.Plan(
            corpus, {}, load_thresholds(), [], tmp_path, (Mode.HTTP,), frozenset(), ("f",)
        )
        state = pipeline._Run(plan, replay)
        state.capture_truth()
        assert seen == {"order": ["/seed/", "/a"], "status": 200}
        assert get_status(replay, "/a") == 429
    assert plan.browser is BrowserName.CHROMIUM


def _lost(layer: Layer, reason: str, kind: Kind = Kind.PAGE_LOST) -> Difference:
    return Difference(kind, Mode.HTTP, layer, reason, "/p")


def test_a_lost_page_moves_to_the_lowest_layer_and_takes_that_layers_reason() -> None:
    by_layer = {
        Layer.CONVERTER: [_lost(Layer.CONVERTER, "panicerror-begin-end")],
        Layer.CRAWLER: [_lost(Layer.CRAWLER, "no-content-extracted")],
        Layer.LILBEE: [_lost(Layer.LILBEE, "page-produced-empty-markdown")],
    }
    (found,) = parity.attribute_to_lowest_layer(by_layer, Layer.LILBEE)
    assert (found.layer, found.feature) == (Layer.CONVERTER, "panicerror-begin-end")


def test_a_lost_page_and_an_extra_page_are_not_one_finding() -> None:
    by_layer = {
        Layer.CRAWLER: [_lost(Layer.CRAWLER, "x", Kind.PAGE_EXTRA)],
        Layer.LILBEE: [_lost(Layer.LILBEE, "y")],
    }
    (found,) = parity.attribute_to_lowest_layer(by_layer, Layer.LILBEE)
    assert (found.layer, found.feature) == (Layer.LILBEE, "y")
