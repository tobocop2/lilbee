"""Truth records, the question sample, and what a driver's output becomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from tools.qa.crawl_parity import leftover, retrieval, sides
from tools.qa.crawl_parity.model import Kind, Layer, Mode, Page, Side
from tools.qa.crawl_parity.sides import (
    DRIVERS_DIR,
    CrawlRequest,
    SideConfig,
    load_sides,
    path_of,
    run_crawl,
)
from tools.qa.crawl_parity.tests._support import truth
from tools.qa.crawl_parity.thresholds import RetrievalLimits
from tools.qa.crawl_parity.truth import NO_CONTEXT, Raw, Truth, TruthCache, truth_of

FAKE = Path(__file__).parent / "drivers" / "fake_good.py"
LIMITS = RetrievalLimits(top_k=5, sample_size=10, recall_drop_max=0.02)


def test_truth_separates_visible_hidden_and_attribute_words_and_places_each() -> None:
    record = truth(
        "/p",
        [
            ("svg>text", True, "chart label"),
            ("div>span", False, "secret"),
            ("p", True, "label again"),
        ],
        attributes=["tooltip"],
    )
    assert record.visible.words == ("chart", "label", "label", "again")
    assert dict(record.hidden_words) == {"secret": 1}
    assert dict(record.attribute_words) == {"tooltip": 1}
    assert record.context_of("chart") == "svg>text"
    assert record.context_of("secret") == "hidden:div>span"
    assert record.context_of("tooltip") == "attribute"
    assert record.context_of("absent") == NO_CONTEXT
    assert record.sentences == ("chart label", "label again")
    assert record.scorable() and record.usable()


def test_a_page_a_reader_does_not_get_is_not_scorable() -> None:
    assert not truth("/p", [("p", True, "not found")], status=404).scorable()
    assert not truth("/p", [("p", False, "nothing shown")]).scorable()
    failed = truth_of("/p", Raw(0, False, error="timeout"))
    assert (failed.usable(), failed.scorable(), failed.error) == (False, False, "timeout")
    assert not truth_of("/image", Raw(200, False)).usable()


def test_the_truth_cache_returns_what_was_put_after_a_reload(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "truth.json"
    cache = TruthCache(path)
    assert cache.get("k") is None
    raw = Raw(200, True, {"text": "a", "nodes": [["p", True, "a"]], "attributes": []})
    cache.put("k", raw)
    cache.save()
    reloaded = TruthCache(path).get("k")
    assert reloaded == raw
    assert truth_of("/p", raw).visible.words == ("a",)


def _truths() -> dict[str, Truth]:
    shared = "the same footer line everywhere"
    return {
        f"/p{index}": truth(
            f"/p{index}",
            [
                ("p", True, f"unique sentence number {index} here"),
                ("p", True, shared),
                ("p", True, "ab"),
            ],
        )
        for index in range(8)
    }


def test_questions_are_seeded_unique_to_one_page_and_long_enough() -> None:
    first = retrieval.questions(_truths(), 5, seed=3)
    assert first == retrieval.questions(_truths(), 5, seed=3)
    assert first != retrieval.questions(_truths(), 5, seed=4)
    assert len(first) == 5
    assert all(question.text.startswith("unique sentence number") for question in first)
    assert all(question.text in " ".join(_truths()[question.page].sentences) for question in first)
    assert len(retrieval.questions(_truths(), 100, seed=0)) == 8


def test_a_page_has_one_document_name_on_both_sides() -> None:
    assert retrieval.document_name("/a") == retrieval.document_name("/a")
    assert retrieval.document_name("/a") != retrieval.document_name("/b")
    assert retrieval.document_name("/a").endswith(".md")


def _result(answered: int) -> retrieval.RetrievalResult:
    return retrieval.RetrievalResult(10, tuple(f"q{i}" for i in range(answered)), 8, 0, 1.0, 1.0)


def test_recall_below_the_oracle_by_more_than_the_limit_is_one_difference() -> None:
    asked = [retrieval.Question(f"q{i}", f"sentence {i}", f"/p{i}") for i in range(10)]
    assert retrieval.compare(_result(10), _result(10), asked, LIMITS, Layer.LILBEE, Mode.HTTP) == []
    assert retrieval.compare(_result(9), _result(10), asked, LIMITS, Layer.LILBEE, Mode.HTTP) == []
    (difference,) = retrieval.compare(
        _result(10), _result(8), asked, LIMITS, Layer.LILBEE, Mode.HTTP
    )
    assert difference.kind is Kind.RECALL
    assert round(difference.amount, 2) == 0.20
    assert "/p8: sentence 8" in difference.detail and "/p9" in difference.detail
    assert _result(0).recall == 0.0


def test_measure_writes_each_saved_page_and_reads_the_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stand-in index driver answers from file names, so the wiring is checked without lilbee."""
    script = tmp_path / "index.py"
    script.write_text(
        "import argparse, json, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "for name in ('--data', '--questions', '--embedding-model', '--top-k', '--out'):\n"
        "    p.add_argument(name)\n"
        "a = p.parse_args()\n"
        "docs = sorted(f.name for f in (pathlib.Path(a.data) / 'documents').iterdir())\n"
        "qs = [json.loads(line) for line in open(a.questions, encoding='utf-8')]\n"
        "json.dump({'added': len(docs), 'failed': [], 'skipped': ['x'], 'index_seconds': 1.5,\n"
        "           'search_seconds': 0.5,\n"
        "           'answers': {q['id']: docs[:int(a.top_k)] for q in qs}},\n"
        "          open(a.out, 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    pages = {"/a": Page("u/a", "alpha"), "/b": Page("u/b", "beta"), "/c": Page("u/c", None, "lost")}
    asked = [retrieval.Question("q0", "alpha", "/a"), retrieval.Question("q1", "gone", "/c")]
    config = retrieval.RetrievalConfig(Path(sys.executable), "model")
    monkeypatch.setattr(retrieval, "DRIVERS_DIR", tmp_path)
    script.rename(tmp_path / retrieval.INDEX_DRIVER)
    result = retrieval.measure(pages, asked, config, top_k=5, work=tmp_path / "work")
    assert (result.asked, result.answered, result.indexed_pages, result.failed_pages) == (
        2,
        ("q0",),
        2,
        1,
    )
    assert result.recall == 0.5
    assert sorted(path.name for path in tmp_path.glob("work.*/data/documents/*")) == sorted(
        [retrieval.document_name("/a"), retrieval.document_name("/b")]
    )


def test_sides_file_names_interpreters_for_each_layer(tmp_path: Path) -> None:
    path = tmp_path / "sides.toml"
    path.write_text(
        '[oracle]\nlabel = "old"\ncrawler = "/o/python"\ncrawler_driver = "crawl4ai_crawl.py"\n'
        '[candidate]\nlilbee = "/c/python"\ncrawler = "/c/python"\nconverter = "/h/python"\n'
        'crawler_driver = "crawlberg_crawl.py"\nconverter_driver = "h2m_convert.py"\n'
        'chrome = "/chrome"\n',
        encoding="utf-8",
    )
    sides = load_sides(path)
    oracle, candidate = sides[Side.ORACLE], sides[Side.CANDIDATE]
    assert (oracle.label, oracle.has(Layer.LILBEE), oracle.has(Layer.CRAWLER)) == (
        "old",
        False,
        True,
    )
    assert candidate.label == "candidate" and candidate.chrome == "/chrome"
    assert candidate.crawl_driver(Layer.LILBEE) == DRIVERS_DIR / "lilbee_crawl.py"
    assert candidate.crawl_driver(Layer.CRAWLER) == DRIVERS_DIR / "crawlberg_crawl.py"
    path.write_text('[candidate]\ncrawler = "/c/python"\n', encoding="utf-8")
    assert set(load_sides(path)) == {Side.CANDIDATE}


def test_every_driver_a_side_can_name_exists() -> None:
    names = ["lilbee_crawl.py", "crawl4ai_crawl.py", "crawlberg_crawl.py", "crawl4ai_convert.py",
             "h2m_convert.py", "lilbee_index.py", "selftest_leaky.py", "_driver_io.py"]  # fmt: skip
    assert [name for name in names if not (DRIVERS_DIR / name).is_file()] == []


def test_a_url_becomes_its_corpus_path_with_its_query() -> None:
    assert path_of("http://127.0.0.1:5000/a/b?x=1#frag") == "/a/b?x=1"
    assert path_of("http://127.0.0.1:5000") == "/"


@pytest.fixture
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(leftover, "GRACE_SECONDS", 0.1)


def _config(driver: Path) -> SideConfig:
    return SideConfig(Side.CANDIDATE, "x", {Layer.CRAWLER: Path(sys.executable)}, str(driver), "")


def test_a_driver_that_cannot_reach_its_seed_gives_a_failed_page_not_an_empty_success(
    tmp_path: Path, quick: None
) -> None:
    request = CrawlRequest("http://127.0.0.1:9/none", Mode.HTTP)
    result = run_crawl(_config(FAKE), Layer.CRAWLER, request, tmp_path / "work")
    assert result.return_code != 0
    assert result.pages == {} and result.crawl_seconds is None and result.threads_started is None
    assert "Connection refused" in result.stderr_tail or "URLError" in result.stderr_tail


def test_a_driver_past_its_time_limit_is_killed_and_reported_as_killed(
    tmp_path: Path, quick: None
) -> None:
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    request = CrawlRequest("http://127.0.0.1:9/none", Mode.HTTP, run_timeout=1.0)
    result = run_crawl(_config(sleeper), Layer.CRAWLER, request, tmp_path / "work")
    assert result.return_code == -9
    assert result.pages == {} and result.wall_seconds < 30


def test_lilbee_failure_lines_become_pages_with_their_reason(
    tmp_path: Path, quick: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stand-in lilbee driver saves one page and logs one failure the way lilbee logs it."""
    driver = tmp_path / "lilbee_like.py"
    driver.write_text(
        "import json, pathlib, sys, time\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "out.mkdir(parents=True)\n"
        "page = {'url': 'http://h/ok?q=1', 'markdown': 'text', 'error': None,\n"
        "        'saved_at': time.time()}\n"
        "(out / 'pages.jsonl').write_text(json.dumps(page) + '\\n', encoding='utf-8')\n"
        "run = {'crawl_started': time.time() - 1, 'crawl_ended': time.time(),\n"
        "       'versions': {'v': '1'},\n"
        "       'threads_before': 1, 'threads_after': 3}\n"
        "(out / 'run.json').write_text(json.dumps(run), encoding='utf-8')\n"
        "print('WARNING lilbee.crawler.events: Crawled page yields no content: http://h/bad: "
        "browser_timeout: timed out after 30s', file=sys.stderr)\n",
        encoding="utf-8",
    )
    config = SideConfig(Side.CANDIDATE, "x", {Layer.LILBEE: Path(sys.executable)}, "", "")
    monkeypatch.setattr(sides, "LILBEE_DRIVER", str(driver))
    result = run_crawl(
        config, Layer.LILBEE, CrawlRequest("http://h/", Mode.BROWSER), tmp_path / "work"
    )
    assert set(result.pages) == {"/ok?q=1", "/bad"}
    assert result.pages["/bad"].error == "browser_timeout: timed out after 30s"
    assert list(result.saved()) == ["/ok?q=1"]
    assert (result.threads_started, result.versions, result.return_code) == (2, {"v": "1"}, 0)
    assert result.first_page_seconds is not None and 0.5 < result.first_page_seconds < 5
    assert json.dumps(result.left.temp_entries) == "[]"
