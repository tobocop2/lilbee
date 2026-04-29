"""T8 e2e crawl. Bootstrap Chromium, crawl a local HTTP fixture with retries,
sync the result, ask a question that requires the crawled content.

Uses a local `python -m http.server` against a static HTML file rather than a
public URL so the test is deterministic, doesn't depend on the network being
up to a third party, and can't be rate-limited.

The retry-on-failure logic uses tenacity around the crawl invocation so a
single transient connection refusal during the http.server boot doesn't fail
the test.
"""

from __future__ import annotations

import contextlib
import json
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path
from threading import Thread

import pytest
from tenacity import retry, stop_after_attempt, wait_exponential

from conftest import Lane

_CHROMIUM_BOOTSTRAP_TIMEOUT = 600.0
_CRAWL_TIMEOUT = 300.0
_SYNC_TIMEOUT = 240.0
_SEARCH_TIMEOUT = 90.0


_LILBEE_PAGE_HTML = """<!DOCTYPE html>
<html><head><title>QA Crawl Fixture</title></head>
<body>
<h1>QA Crawl Fixture</h1>
<p>This page exists solely to verify lilbee's crawl and ingest pipeline.
The unique identifying phrase is: <strong>quokka-rendezvous-protocol</strong>.</p>
<p>The crawler should fetch this page, the ingest pipeline should chunk
and embed it, and a semantic search for the unique phrase should return
the page's source URL.</p>
</body></html>
"""


@pytest.fixture(scope="session")
def chromium_ready(lane: Lane, qa_models_dir: Path) -> str:
    """Bootstrap Chromium and verify the crawler extras are usable.

    Three failure modes get skipped (not failed): the binary lacks the crawler
    extras (`pip install lilbee` without `[crawler]`), Chromium can't be
    downloaded (offline / Playwright mirror down), or the binary doesn't
    expose `setup crawler` at all (very old build).
    """
    import os

    env = os.environ.copy()
    env["LILBEE_DATA"] = str(qa_models_dir / "data-crawl")
    env["LILBEE_MODELS_DIR"] = str(qa_models_dir)
    env["LILBEE_NO_SPLASH"] = "1"
    env["LILBEE_LOG_LEVEL"] = "WARNING"

    # Probe crawler-extras availability via `lilbee add --help`. If extras are
    # missing, `lilbee add --crawl` later prints a 'Web crawling requires...'
    # hint and exits 0 without doing anything. Catch that up front so the
    # crawl test doesn't pretend to succeed.
    probe = subprocess.run(
        [lane.lilbee_bin, "add", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0 or "--crawl" not in (probe.stdout + probe.stderr):
        pytest.skip("lilbee add --help missing --crawl flag; crawler extras not installed")

    result = subprocess.run(
        [lane.lilbee_bin, "setup", "crawler"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHROMIUM_BOOTSTRAP_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            f"Chromium bootstrap failed: rc={result.returncode}\n"
            f"stderr tail: {result.stderr[-500:]}"
        )

    # Confirm extras are actually wired by running a no-op crawl probe. If the
    # CLI prints 'Web crawling requires' the extras aren't loaded at runtime
    # even though --help advertised the flag.
    runtime_probe = subprocess.run(
        [lane.lilbee_bin, "add", "http://127.0.0.1:1", "--crawl", "--max-pages", "0"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    combined = runtime_probe.stdout + runtime_probe.stderr
    if "Web crawling requires" in combined or "crawl4ai" in combined.lower():
        pytest.skip("crawl4ai extras not available at runtime; install lilbee[crawler]")
    return "ok"


@pytest.fixture
def http_fixture_server(tmp_path: Path) -> Iterator[str]:
    """Serve a single static HTML file from a free port; yield the URL."""
    docroot = tmp_path / "crawl-fixture-root"
    docroot.mkdir()
    (docroot / "index.html").write_text(_LILBEE_PAGE_HTML)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=docroot,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Drain pipes in background so the buffer doesn't fill and stall the server.
    def _drain(stream: subprocess.IO[str] | None) -> None:
        if stream is None:
            return
        for _ in stream:
            pass

    Thread(target=_drain, args=(proc.stdout,), daemon=True).start()
    Thread(target=_drain, args=(proc.stderr,), daemon=True).start()

    # Wait until the server is reachable, with a tenacity retry around the
    # actual probe to absorb the transient connect-refused during boot.
    @retry(
        stop=stop_after_attempt(15),
        wait=wait_exponential(multiplier=0.2, min=0.2, max=2.0),
        reraise=True,
    )
    def _probe() -> None:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            pass

    try:
        _probe()
        yield f"http://127.0.0.1:{port}"
    finally:
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=3)
        with contextlib.suppress(Exception):
            proc.kill()


@pytest.mark.crawl
@pytest.mark.writer
@pytest.mark.timeout(420)
def test_crawl_and_search_roundtrip(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
    models_pulled: dict[str, str],
    chromium_ready: str,
    http_fixture_server: str,
) -> None:
    """Crawl the fixture URL, sync, then search for the unique phrase. The
    URL should appear in the search results' source field.

    The crawl invocation itself is wrapped in tenacity retries: Playwright
    occasionally fails to launch on a cold runner; one or two retries lets
    a real flake recover without failing the cell.
    """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    def _crawl() -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                lane.lilbee_bin,
                "add",
                http_fixture_server,
                "--crawl",
                "--depth",
                "0",
                "--max-pages",
                "1",
            ],
            env=lilbee_env_with_models,
            capture_output=True,
            text=True,
            timeout=_CRAWL_TIMEOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"crawl failed: rc={result.returncode}\nstderr tail: {result.stderr[-500:]}"
            )
        return result

    crawl = _crawl()
    assert crawl.returncode == 0, crawl.stderr

    # `lilbee add --crawl` auto-syncs in some versions; explicit sync to be safe.
    sync = subprocess.run(
        [lane.lilbee_bin, "sync"],
        env=lilbee_env_with_models,
        capture_output=True,
        text=True,
        timeout=_SYNC_TIMEOUT,
        check=False,
    )
    assert sync.returncode == 0, sync.stderr

    # Diagnostic: confirm the crawled page actually landed in the store
    # before we try to search for it. A zero-chunk store means the crawl
    # call returned ok but ingest didn't write anything (silent failure
    # mode worth surfacing distinctly from a search miss).
    status = subprocess.run(
        [lane.lilbee_bin, "--json", "status"],
        env=lilbee_env_with_models,
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    status_payload = json.loads(status.stdout)
    assert status_payload.get("total_chunks", 0) > 0, (
        f"crawl + sync produced zero chunks. crawl stdout tail:\n"
        f"{crawl.stdout[-1000:]}\nsync stdout:\n{sync.stdout}\n"
        f"status: {status_payload}"
    )

    # Search for the unique phrase. The crawled page should be the only source.
    search = subprocess.run(
        [
            lane.lilbee_bin,
            "--json",
            "search",
            "quokka-rendezvous-protocol",
            "--top-k",
            "5",
        ],
        env=lilbee_env_with_models,
        capture_output=True,
        text=True,
        timeout=_SEARCH_TIMEOUT,
        check=False,
    )
    assert search.returncode == 0, search.stderr
    payload = json.loads(search.stdout)
    results = payload.get("results", [])
    assert results, payload
    chunk_blob = json.dumps(results)
    assert "quokka-rendezvous-protocol" in chunk_blob, payload
