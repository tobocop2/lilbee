"""T5 e2e streaming. Chat round-trip via SSE and TUI to catch reasoning softlocks.

The non-streaming `lilbee ask` call won't catch the failure mode where the
model emits a `<think>` block and then hangs without ever producing
visible answer tokens. The streaming surfaces (HTTP /api/ask/stream and
the TUI chat screen) DO surface that bug as a stuck event sequence or a
spinner that never advances.

These tests assert:
  - TOKEN events fire after the reasoning phase
  - DONE event arrives within a generous timeout
  - The TUI advances past the 'thinking' spinner to actual response text
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from drivers.tui import TuiSession
from httpx_sse import EventSource

from conftest import Lane

_FIXTURES = Path(__file__).parent / "fixtures" / "notes"
_SYNC_TIMEOUT = 240.0
_STREAM_TIMEOUT = 240.0
_TUI_BOOT_TIMEOUT = 60.0
_TUI_RESPONSE_TIMEOUT = 360.0


def _seed_corpus(lilbee_data: Path) -> Path:
    documents = lilbee_data / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    for path in _FIXTURES.glob("*.md"):
        shutil.copy(path, documents / path.name)
    return documents


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_health(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == httpx.codes.OK:
                return
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ):
            pass
        time.sleep(0.3)
    raise TimeoutError(f"server at {url} not ready within {timeout}s")


def _fetch_token(lane: Lane, env: dict[str, str]) -> str:
    """`lilbee token` prints the bearer for a running server. Empty string if
    the binary doesn't expose the command (very old build) or it errors."""
    result = subprocess.run(
        [lane.lilbee_bin, "token"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        return ""
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


@pytest.fixture
def served_lilbee(
    lane: Lane, lilbee_env_with_models: dict[str, str]
) -> Iterator[tuple[str, dict[str, str], dict[str, str]]]:
    """Spawn `lilbee serve` and yield (url, env, headers).

    Headers carry the bearer token that protected POST endpoints require.
    """
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [lane.lilbee_bin, "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=lilbee_env_with_models,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_health(f"{base_url}/api/health", timeout=60.0)
        token = _fetch_token(lane, lilbee_env_with_models)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        yield base_url, lilbee_env_with_models, headers
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(420)
def test_ask_stream_completes_with_token_events(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
    served_lilbee: tuple[str, dict[str, str], dict[str, str]],
) -> None:
    """`POST /api/ask/stream` emits TOKEN events and a DONE event within timeout.

    A model that softlocks on the `<think>` phase would produce no TOKEN
    events past the reasoning marker and never DONE. This asserts both:
    (a) the stream advanced beyond reasoning, (b) it terminated cleanly.
    """
    _seed_corpus(lilbee_data)
    sync = subprocess.run(
        [lane.lilbee_bin, "sync"],
        env=lilbee_env_with_models,
        capture_output=True,
        text=True,
        timeout=_SYNC_TIMEOUT,
        check=False,
    )
    assert sync.returncode == 0, sync.stderr

    base_url, _env, headers = served_lilbee
    events: list[tuple[str, str]] = []
    saw_done = False
    saw_error = False

    deadline = time.monotonic() + _STREAM_TIMEOUT
    with (
        httpx.Client(timeout=_STREAM_TIMEOUT, headers=headers) as client,
        client.stream(
            "POST",
            f"{base_url}/api/ask/stream",
            json={"question": "Answer in one short sentence: which document covers EV batteries?"},
        ) as response,
    ):
        response.raise_for_status()
        for sse in EventSource(response).iter_sse():
            evt = sse.event.lower()
            events.append((evt, sse.data[:120]))
            if evt == "done":
                saw_done = True
                break
            if evt == "error":
                saw_error = True
                break
            if time.monotonic() > deadline:
                break

    summary = {
        "events_seen": len(events),
        "first_5": events[:5],
        "last_5": events[-5:],
        "saw_done": saw_done,
        "saw_error": saw_error,
    }
    assert saw_done, f"stream never reached DONE within {_STREAM_TIMEOUT}s; {summary}"
    assert not saw_error, f"stream ended with ERROR; {summary}"
    token_events = [e for e, _ in events if e == "token"]
    assert token_events, f"no token events emitted; {summary}"


@pytest.mark.tui
@pytest.mark.writer
@pytest.mark.timeout(420)
@pytest.mark.xfail(
    sys.platform in {"darwin", "win32"},
    reason=(
        "bb-9c67: TUI chat softlocks on 'thinking...' on macOS/Windows; "
        "Linux pypi/binary lanes pass"
    ),
    strict=False,
)
def test_tui_chat_advances_past_thinking_spinner(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
    models_pulled: dict[str, str],
) -> None:
    """The TUI chat screen produces a response after the thinking spinner.

    This is the failure mode users have been reporting: the spinner /
    'thinking...' indicator hangs forever, no actual response text renders.
    Asserting that we eventually see content from the corpus (a phrase from
    ev-notes that the model is highly likely to surface for a battery
    question) catches the softlock. The unique seed phrase makes the
    assertion deterministic without requiring exact token comparison.
    """
    _seed_corpus(lilbee_data)
    sync = subprocess.run(
        [lane.lilbee_bin, "sync"],
        env=lilbee_env_with_models,
        capture_output=True,
        text=True,
        timeout=_SYNC_TIMEOUT,
        check=False,
    )
    assert sync.returncode == 0, sync.stderr

    session = TuiSession([lane.lilbee_bin], env=lilbee_env_with_models)
    try:
        session.wait_for("lilbee", timeout=_TUI_BOOT_TIMEOUT)
        session.send("Answer in one short sentence: which document covers EV batteries?\r")
        # The contract this test checks is "the TUI streams a response and
        # doesn't softlock on 'thinking...'", not "the model answers the
        # question correctly" (covered separately by CLI/HTTP cite-the-source
        # assertions against the structured sources array). A 135M model
        # routinely picks the wrong chunk and dumps it verbatim — that's a
        # model-quality issue, not a TUI bug. Assert the response section
        # rendered SOMETHING from either source: any phrase that appears in
        # one of the seed corpus files. If none appear within the timeout,
        # the spinner softlocked.
        corpus_markers = (
            # ev-notes phrases
            "lithium",
            "battery",
            "Wh/kg",
            # coffee-notes phrases
            "French press",
            "extraction",
            "grind",
        )
        deadline = time.monotonic() + _TUI_RESPONSE_TIMEOUT
        while time.monotonic() < deadline:
            visible = session.text().lower()
            if any(marker.lower() in visible for marker in corpus_markers):
                return
            time.sleep(1.0)
        screenshot = lilbee_data / "tui-softlock.txt"
        session.screenshot(screenshot)
        raise AssertionError(
            f"TUI never rendered a response derived from the corpus within "
            f"{_TUI_RESPONSE_TIMEOUT}s. Suggests softlock on 'thinking...'.\n"
            f"Last visible screen:\n{session.text()}"
        )
    finally:
        session.close()
