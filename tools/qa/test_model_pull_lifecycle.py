"""T6 model pull lifecycle.

Validates `lilbee model pull` persists the model registry across server
restarts (catalog state lives on disk, not in-memory) and that pulling a
non-existent HF ref returns a clear error rather than a stack trace.

Today /api/models/pull and `lilbee model pull` are smoke-tested only against
an empty registry; a regression that broke disk persistence (e.g. only
recording the model in a per-process cache) would not surface until a user
restarted lilbee and discovered their model gone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import httpx
import pytest

from conftest import Lane, lilbee_env, serve_lilbee_with

_PULL_TIMEOUT = 60.0


def _serve_once_and_query_installed(lane: Lane, env: dict[str, str]) -> list[dict[str, object]]:
    """Boot lilbee serve on a free port, hit /api/models/installed, tear down."""
    with serve_lilbee_with(lane, env) as base_url:
        response = httpx.get(f"{base_url}/api/models/installed", timeout=30.0)
        assert response.status_code == httpx.codes.OK, response.text
        payload = response.json()
    # Endpoint shape varies across releases: bare list OR {"models": [...]}.
    if isinstance(payload, list):
        return payload
    return list(payload.get("models", []))


@pytest.mark.catalog
@pytest.mark.writer
@pytest.mark.timeout(360)
def test_pulled_models_survive_server_restart(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
    models_pulled: dict[str, str],
) -> None:
    """Models pulled by the session fixture appear in /api/models/installed
    on a fresh ``lilbee serve`` and remain on a second cold-start with the
    same data dir.

    The pull happens in the session-scoped ``models_pulled`` fixture, not
    in this test body. What this test gates is the post-pull invariant:
    once a model is pulled, the on-disk registry serves it identically
    across server restarts. A regression that kept the registered set in
    a per-process cache (instead of on disk) would make the two cold-start
    queries return different sets.
    """
    first = _serve_once_and_query_installed(lane, lilbee_env_with_models)
    second = _serve_once_and_query_installed(lane, lilbee_env_with_models)

    def _names(rows: list[dict[str, object]]) -> set[str]:
        out: set[str] = set()
        for row in rows:
            for key in ("name", "hf_repo", "ref"):
                value = row.get(key)
                if isinstance(value, str) and value:
                    out.add(value)
                    break
        return out

    first_names = _names(first)
    second_names = _names(second)
    assert first_names, f"first /api/models/installed returned empty: {first}"
    assert first_names == second_names, (
        "installed models diverged across server restart "
        f"(first={first_names!r}, second={second_names!r})"
    )

    chat_repo = "/".join(models_pulled["chat"].split("/")[:2])
    assert any(chat_repo in name for name in second_names), (
        f"pulled chat model {chat_repo!r} missing after restart: {second_names!r}"
    )


@pytest.mark.catalog
@pytest.mark.timeout(120)
def test_pull_unknown_model_returns_clear_error(lane: Lane, lilbee_data: Path) -> None:
    """Pulling a non-existent HF ref fails with a user-facing error message,
    not a bare Python traceback, and not a silent zero exit.

    The keyword list is deliberately specific: it admits the messages a CLI
    user would recognize (`not found`, `does not exist`, `404`, etc.) and
    rejects the generic word `error` because that would also match every
    `RuntimeError`/`OSError` traceback line and let a bare trace pass.
    """
    env = lilbee_env(lilbee_data)
    result = subprocess.run(
        [
            lane.lilbee_bin,
            "model",
            "pull",
            "this-org-does-not-exist-1234/this-repo-does-not-exist-5678-GGUF",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=_PULL_TIMEOUT,
        check=False,
    )
    assert result.returncode != 0, (
        f"pull of non-existent model should fail; got rc=0\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    combined = (result.stdout + result.stderr).lower()
    user_facing_phrases = (
        "not found",
        "does not exist",
        "404",
        "no such",
        "could not",
        "unable to",
    )
    assert any(phrase in combined for phrase in user_facing_phrases), (
        f"expected a user-facing error phrase, got something else:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "traceback" not in combined, (
        f"unknown-model error leaked a Python traceback to the user:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
