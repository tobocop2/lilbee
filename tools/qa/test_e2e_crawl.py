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
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from threading import Thread

import pytest
from tenacity import retry, stop_after_attempt, wait_exponential

from conftest import Lane, LaneName, current_lane_name

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
    """Bootstrap Chromium and verify the crawler stack is usable.

    Behavior depends on the lane:
      * pypi lane installs plain `lilbee` (no [crawler] extra) so the
        crawler stack is genuinely absent. Skip; there's nothing to test.
      * binary lane bundles crawler+playwright into the release artifact
        (the user-facing contract). If the bundled binary can't load the
        crawler stack that's a release defect we want surfaced, not
        skipped. Raise so the test xfail decorator on binary captures it.
    """
    env = os.environ.copy()
    env["LILBEE_DATA"] = str(qa_models_dir / "data-crawl")
    env["LILBEE_MODELS_DIR"] = str(qa_models_dir)
    env["LILBEE_NO_SPLASH"] = "1"
    env["LILBEE_LOG_LEVEL"] = "WARNING"

    def _gate(message: str) -> str:
        # On pypi (no extras) skip is the right call. On binary, where
        # the artifact bundles the crawler stack, raising surfaces the
        # release defect so the xfail decorator on the test method gets
        # to record it.
        if lane.is_binary:
            raise RuntimeError(message)
        pytest.skip(message)

    probe = subprocess.run(
        [lane.lilbee_bin, "add", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0 or "--crawl" not in (probe.stdout + probe.stderr):
        _gate("lilbee add --help missing --crawl flag; crawler not exposed in this artifact")

    result = subprocess.run(
        [lane.lilbee_bin, "setup", "crawler"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHROMIUM_BOOTSTRAP_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        _gate(
            f"Chromium bootstrap failed: rc={result.returncode}\n"
            f"stderr tail: {result.stderr[-500:]}"
        )

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
        _gate("crawl4ai not available at runtime in this artifact")
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

    # Use sys.executable rather than "python3": Windows ships python.exe
    # (no python3 alias), so hardcoding "python3" silently fails to start
    # the fixture server and lilbee reports "Crawled 0 page(s)" with no
    # error of its own.
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
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
@pytest.mark.xfail(
    sys.platform == "win32" and current_lane_name() is LaneName.L1_PYPI,
    reason="bb-l7t4: Windows pypi lane crawl returns 0 pages from local http.server fixture",
    strict=False,
)
@pytest.mark.xfail(
    current_lane_name() is LaneName.L2_BINARY,
    reason=(
        "bb-sxsz: bundled binary's crawler stack is broken in b455 - "
        "lilbee add --help omits --crawl and lilbee setup crawler exits 2. "
        "PR #195 fixes the spawn path; xfail until a release ships with that fix."
    ),
    strict=False,
)
def test_crawl_and_search_roundtrip(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
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
