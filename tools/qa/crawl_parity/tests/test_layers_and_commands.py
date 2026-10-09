"""Layer attribution through a whole run, the left-behind path of a real driver, and exit codes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tools.qa.crawl_parity import cli, leftover, pipeline, selftest
from tools.qa.crawl_parity.converter import PROCESS_EXITED, Converter
from tools.qa.crawl_parity.model import Kind, Layer, Mode, Side
from tools.qa.crawl_parity.pipeline import left_behind_differences
from tools.qa.crawl_parity.sides import DRIVERS_DIR, CrawlRequest, SideConfig, run_crawl
from tools.qa.crawl_parity.tests.test_pipeline import CORPUS, DRIVERS, fake_capture, plan
from tools.qa.crawl_parity.thresholds import LeftBehindLimits

PYTHON = Path(sys.executable)


@pytest.fixture(autouse=True)
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "capture", fake_capture(CORPUS))
    monkeypatch.setattr(leftover, "GRACE_SECONDS", 0.1)


def layered(which: Side, crawler: str, converter: str) -> SideConfig:
    return SideConfig(
        which,
        f"stand-in {crawler} over {converter}",
        {Layer.CONVERTER: PYTHON, Layer.CRAWLER: PYTHON},
        str(DRIVERS / crawler),
        str(DRIVERS / converter),
    )


def layers_found(tmp_path: Path, converter: str) -> set[tuple[Kind, Layer]]:
    sides = {
        Side.ORACLE: layered(Side.ORACLE, "fake_good.py", "fake_convert_good.py"),
        Side.CANDIDATE: layered(Side.CANDIDATE, "fake_lossy.py", converter),
    }
    outcome = pipeline.run(plan(tmp_path, "fake_lossy.py", sides=sides, work=tmp_path / converter))
    assert [c.layer for c in outcome.comparisons] == [Layer.CONVERTER, Layer.CRAWLER]
    return {(d.kind, d.layer) for d in outcome.differences}


def test_a_loss_the_converter_alone_shows_is_the_converters(tmp_path: Path) -> None:
    assert layers_found(tmp_path, "fake_convert_lossy.py") == {
        (Kind.TEXT_LOST, Layer.CONVERTER),
        (Kind.PAGE_LOST, Layer.CRAWLER),
    }


def test_a_loss_the_converter_alone_does_not_show_is_the_crawlers(tmp_path: Path) -> None:
    assert layers_found(tmp_path, "fake_convert_good.py") == {
        (Kind.TEXT_LOST, Layer.CRAWLER),
        (Kind.PAGE_LOST, Layer.CRAWLER),
    }


def test_a_converter_that_exits_gives_a_failed_page_and_is_started_again(tmp_path: Path) -> None:
    dying = tmp_path / "dying.py"
    dying.write_text("import sys\nsys.stdin.readline()\nsys.exit(3)\n", encoding="utf-8")
    config = SideConfig(Side.CANDIDATE, "x", {Layer.CONVERTER: PYTHON}, "", str(dying))
    with Converter(config, tmp_path / "stderr.txt") as converter:
        first = converter.convert("http://h/a", "<p>a</p>")
        second = converter.convert("http://h/b", "<p>b</p>")
    assert (first.markdown, first.error) == (None, PROCESS_EXITED)
    assert (second.markdown, second.error) == (None, PROCESS_EXITED)


def test_a_converter_failure_is_the_pages_error(tmp_path: Path) -> None:
    raising = tmp_path / "raising.py"
    raising.write_text(
        f"import sys\nsys.path.insert(0, {str(DRIVERS_DIR)!r})\n"
        "from _driver_io import serve_conversions\n"
        "def convert(html, base_url):\n"
        "    if 'bad' in html:\n"
        "        raise BaseException('panic: begin > end')\n"
        "    return '' if 'empty' in html else 'ok ' + base_url\n"
        "serve_conversions(convert)\n",
        encoding="utf-8",
    )
    config = SideConfig(Side.CANDIDATE, "x", {Layer.CONVERTER: PYTHON}, "", str(raising))
    with Converter(config, tmp_path / "stderr.txt") as converter:
        bad = converter.convert("http://h/a", "<p>bad</p>")
        good = converter.convert("http://h/b", "<p>fine</p>")
        empty = converter.convert("http://h/c", "<p>empty</p>")
    assert bad.error == "BaseException: panic: begin > end"
    assert good.markdown == "ok http://h/b"
    assert (empty.markdown, empty.error) == (None, "No content extracted")


def test_a_real_driver_that_leaves_things_is_caught_through_run_crawl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SideConfig(Side.CANDIDATE, "leaky", {Layer.CRAWLER: PYTHON}, "selftest_leaky.py", "")
    request = CrawlRequest("http://127.0.0.1:9/x", Mode.BROWSER)
    # The planted child needs a moment to open its listening socket.
    monkeypatch.setattr(leftover, "GRACE_SECONDS", 1.5)
    result = run_crawl(config, Layer.CRAWLER, request, tmp_path / "work")
    assert len(result.left.processes) == 1
    assert [name[:16] for name in result.left.temp_entries] == ["planted-profile-"]
    assert result.threads_started == 1
    limits = LeftBehindLimits(processes_max=0, temp_entries_max=0, threads_max=0)
    features = {d.feature.split(":")[0] for d in left_behind_differences(result, None, limits)}
    assert features == {"process", "listening-socket", "temp", "threads"}


def test_what_a_candidate_leaves_is_a_finding_of_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaky = SideConfig(Side.CANDIDATE, "leaky", {Layer.CRAWLER: PYTHON}, "selftest_leaky.py", "")
    good = plan(tmp_path, "fake_good.py").sides[Side.ORACLE]
    # The planted child needs a moment to open its listening socket.
    monkeypatch.setattr(leftover, "GRACE_SECONDS", 1.5)
    sides = {Side.ORACLE: good, Side.CANDIDATE: leaky}
    outcome = pipeline.run(plan(tmp_path, "fake_good.py", sides=sides))
    left = {d.feature.split(":")[0] for d in outcome.differences if d.kind is Kind.LEFT_BEHIND}
    assert left == {"process", "listening-socket", "temp", "threads"}
    assert outcome.verdict is not None and not outcome.verdict.passed


def _selftest_command(tmp_path: Path) -> list[str]:
    sides = tmp_path / "sides.toml"
    sides.write_text(f'[candidate]\ncrawler = "{PYTHON.as_posix()}"\n', encoding="utf-8")
    return ["selftest", "--sides", str(sides), "--work", str(tmp_path / "work")]


@pytest.mark.parametrize(
    ("results", "code", "summary"),
    [
        (
            [selftest.Result.PASS, selftest.Result.SKIPPED],
            0,
            "SELFTEST PASS: 1 passed, 0 failed, 1 skipped",
        ),
        (
            [selftest.Result.PASS, selftest.Result.FAIL],
            1,
            "SELFTEST FAIL: 1 passed, 1 failed, 0 skipped",
        ),
    ],
)
def test_the_selftest_command_fails_when_one_selftest_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    results: list[selftest.Result],
    code: int,
    summary: str,
) -> None:
    canned = [
        selftest.SelfTest(f"check {index}", result, "detail")
        for index, result in enumerate(results)
    ]
    monkeypatch.setattr(selftest, "run", lambda *_args: canned)
    assert cli.main(_selftest_command(tmp_path)) == code
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == summary
    assert lines[0] == f"{results[0]:<7} check 0: detail"
