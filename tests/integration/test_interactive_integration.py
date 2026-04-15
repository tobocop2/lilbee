"""Real integration tests with actual models -- no mocked embeddings or LLM.

Exercises the full CLI pipeline with real llama-cpp inference:
- nomic-embed-text for embeddings
- Qwen3 0.6B for chat
- crawl4ai against pytest-httpserver

Models are downloaded once per module (~800 MB first run) and cached
in ~/.lilbee/models/ for subsequent runs.

Marked @pytest.mark.slow -- excluded from default `make check`.
"""

import asyncio
import ipaddress
import json
import os
from unittest import mock

import pytest
from typer.testing import CliRunner

from lilbee.catalog import FEATURED_CHAT, FEATURED_EMBEDDING, download_model
from lilbee.cli.app import app
from lilbee.config import cfg
from lilbee.platform import canonical_models_dir
from lilbee.services import reset_services as reset_provider

pytestmark = pytest.mark.slow

runner = CliRunner()


def _parse_json_output(output: str) -> dict:
    """Extract JSON from CLI output that may have llama.cpp noise lines."""
    for line in output.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return json.loads(output)


SPECS_MD = """\
# Thunderbolt X500 Specifications

Engine: 3.5L V6 TurboForce
Oil capacity: 6.5 quarts
Top speed: 155 mph
Tire pressure: 35 PSI front, 33 PSI rear
"""

AUTH_PART1_MD = """\
# Authentication: OAuth 2.0

OAuth 2.0 is the industry-standard protocol for authorization.
It works by delegating user authentication to the service that
hosts the user account. Tokens are issued by the authorization
server after the resource owner grants access.
"""

AUTH_PART2_MD = """\
# Authentication: JWT Tokens

JSON Web Tokens (JWT) are compact, URL-safe tokens used for
transmitting claims between parties. A JWT has three parts:
header, payload, and signature. They are commonly used for
stateless session management.
"""

AUTH_PART3_MD = """\
# Authentication: Session Management

Session-based authentication stores state on the server side.
When a user logs in, the server creates a session and returns
a session ID cookie. The cookie is sent with every subsequent
request to maintain the authenticated state.
"""

DEPLOY_MD = """\
# Kubernetes Deployment Guide

Use kubectl to deploy your application to the cluster:

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deployment/myapp
kubectl get pods -l app=myapp
```

Monitor with `kubectl logs` and scale with `kubectl scale`.
"""

FIBONACCI_PY = '''\
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''

DB_PERF_MD = """\
# Database Performance Tuning

Connection pooling is essential for database performance.
A pool maintains a set of reusable connections, eliminating
the overhead of creating new connections for each request.
Configure pool size based on your workload: too few connections
cause queuing, too many waste memory.
"""

API_PERF_MD = """\
# API Performance Optimization

Connection pooling at the HTTP client layer reduces latency.
Reuse TCP connections across requests to avoid the handshake
overhead. Most HTTP client libraries support connection pooling
out of the box. Set appropriate timeouts and max connections.
"""


def _chat_model_entry():
    """Return the chat model catalog entry based on env var or default."""
    from tests.integration.conftest import _CI_CHAT_MODEL

    name, tag = _CI_CHAT_MODEL.split(":")
    return next(m for m in FEATURED_CHAT if m.name == name and m.tag == tag)


def _embedding_model_entry():
    """Return the nomic-embed-text catalog entry."""
    return FEATURED_EMBEDDING[0]


@pytest.fixture(scope="module")
def real_models():
    """Download real models once per module. Cached in ~/.lilbee/models/."""
    cfg.models_dir = canonical_models_dir()
    chat_entry = _chat_model_entry()
    embed_entry = _embedding_model_entry()

    chat_path = download_model(chat_entry)
    embed_path = download_model(embed_entry)

    return chat_path, embed_path


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, real_models):
    """Redirect config to temp dirs, configure real llama-cpp models."""
    snapshot = {name: getattr(cfg, name) for name in type(cfg).model_fields}

    _chat_path, _embed_path = real_models

    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir()
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir()
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.data_root = tmp_path

    cfg.llm_provider = "llama-cpp"
    cfg.models_dir = canonical_models_dir()
    cfg.chat_model = _chat_model_entry().ref
    cfg.embedding_model = _embedding_model_entry().ref
    cfg.embedding_dim = 768

    cfg.concept_graph = False
    cfg.query_expansion_count = 0
    cfg.hyde = False
    cfg.max_tokens = 512
    cfg.chunk_size = 128
    cfg.chunk_overlap = 20
    cfg.max_embed_chars = 500

    reset_provider()

    # Point LILBEE_DATA at the test root so CLI apply_overrides() doesn't
    # redirect cfg paths to the CI temp directory mid-command.
    old_lilbee_data = os.environ.get("LILBEE_DATA")
    os.environ["LILBEE_DATA"] = str(tmp_path)

    # Serialize async ingestion to avoid concurrent llama.cpp calls (not thread-safe)
    _max_concurrent_patch = mock.patch("lilbee.ingest._MAX_CONCURRENT", 1)
    _max_concurrent_patch.start()

    yield tmp_path

    _max_concurrent_patch.stop()
    # Reset provider to free llama.cpp model handles before next test
    reset_provider()
    for name, val in snapshot.items():
        setattr(cfg, name, val)
    if old_lilbee_data is None:
        os.environ.pop("LILBEE_DATA", None)
    else:
        os.environ["LILBEE_DATA"] = old_lilbee_data


def _write_all_docs():
    """Write all test documents into documents_dir."""
    docs = {
        "specs.md": SPECS_MD,
        "auth-part1.md": AUTH_PART1_MD,
        "auth-part2.md": AUTH_PART2_MD,
        "auth-part3.md": AUTH_PART3_MD,
        "deploy.md": DEPLOY_MD,
        "fibonacci.py": FIBONACCI_PY,
        "db-perf.md": DB_PERF_MD,
        "api-perf.md": API_PERF_MD,
    }
    for name, content in docs.items():
        (cfg.documents_dir / name).write_text(content)


def _sync():
    """Run real sync."""
    from lilbee.ingest import sync

    return asyncio.run(sync(quiet=True))


def _sync_with_docs():
    """Write all docs and sync."""
    _write_all_docs()
    return _sync()


class TestSearch:
    def test_search_finds_exact_keyword(self, isolated_env):
        _sync_with_docs()
        result = runner.invoke(app, ["--json", "search", "Thunderbolt X500"])
        assert result.exit_code == 0, result.output
        data = _parse_json_output(result.output)
        results = data.get("results", [])
        sources = [r["source"] for r in results]
        assert any("specs.md" in s for s in sources), f"Expected specs.md in {sources}"

    def test_search_finds_semantic(self, isolated_env):
        _sync_with_docs()
        result = runner.invoke(app, ["--json", "search", "engine specifications"])
        assert result.exit_code == 0, result.output
        data = _parse_json_output(result.output)
        results = data.get("results", [])
        sources = [r["source"] for r in results]
        assert any("specs.md" in s for s in sources), f"Expected specs.md in {sources}"


class TestAsk:
    def test_ask_known_fact(self, isolated_env):
        _sync_with_docs()
        result = runner.invoke(app, ["--json", "ask", "What is the oil capacity?"])
        assert result.exit_code == 0, result.output
        data = _parse_json_output(result.output)
        answer = data.get("answer", "").lower()
        assert "6.5" in answer or "quart" in answer, f"Expected oil capacity in: {answer}"

    def test_ask_includes_citations(self, isolated_env):
        _sync_with_docs()
        result = runner.invoke(app, ["--json", "ask", "What is the oil capacity?"])
        assert result.exit_code == 0, result.output
        data = _parse_json_output(result.output)
        sources = data.get("sources", [])
        source_names = [s.get("source", "") for s in sources]
        assert any("specs.md" in s for s in source_names), (
            f"Expected specs.md citation in {source_names}"
        )


class TestDiversity:
    def test_mmr_diverse_sources(self, isolated_env):
        _sync_with_docs()
        result = runner.invoke(app, ["--json", "search", "authentication", "-k", "10"])
        assert result.exit_code == 0, result.output
        data = _parse_json_output(result.output)
        results = data.get("results", [])
        auth_sources = {r["source"] for r in results if "auth" in r["source"]}
        assert len(auth_sources) >= 2, f"Expected diverse auth sources, got {auth_sources}"


class TestAddAndDelete:
    def test_add_file_then_search(self, isolated_env):
        _sync_with_docs()
        external = isolated_env / "external.md"
        external.write_text(
            "# Zymurgy Reference\n\n"
            "Zymurgy is the study of fermentation in brewing and winemaking.\n"
        )
        result = runner.invoke(app, ["add", str(external)])
        assert result.exit_code == 0, result.output

        search_result = runner.invoke(app, ["--json", "search", "zymurgy"])
        assert search_result.exit_code == 0, search_result.output
        data = _parse_json_output(search_result.output)
        results = data.get("results", [])
        assert len(results) > 0, "Expected to find added document"

    def test_delete_removes(self, isolated_env):
        _sync_with_docs()
        result = runner.invoke(app, ["remove", "specs.md"])
        assert result.exit_code == 0, result.output

        search_result = runner.invoke(app, ["--json", "search", "Thunderbolt X500"])
        assert search_result.exit_code == 0
        data = _parse_json_output(search_result.output)
        results = data.get("results", [])
        sources = [r.get("source", "") for r in results]
        assert not any("specs.md" in s for s in sources), (
            f"specs.md should be removed but found in {sources}"
        )


class TestSync:
    def test_sync_idempotent(self, isolated_env):
        _write_all_docs()
        r1 = _sync()
        assert len(r1.added) > 0

        r2 = _sync()
        assert r2.added == [], f"Second sync should add nothing, got {r2.added}"
        assert r2.unchanged > 0


class TestRebuild:
    def test_rebuild_works(self, isolated_env):
        _sync_with_docs()
        result = runner.invoke(app, ["rebuild"])
        assert result.exit_code == 0, result.output

        search_result = runner.invoke(app, ["--json", "search", "Thunderbolt"])
        assert search_result.exit_code == 0
        data = _parse_json_output(search_result.output)
        results = data.get("results", [])
        assert len(results) > 0, "Rebuild should preserve searchability"


class TestCodeSearch:
    def test_code_search(self, isolated_env):
        _sync_with_docs()
        result = runner.invoke(app, ["--json", "search", "fibonacci"])
        assert result.exit_code == 0, result.output
        data = _parse_json_output(result.output)
        results = data.get("results", [])
        sources = [r["source"] for r in results]
        assert any("fibonacci.py" in s for s in sources), f"Expected fibonacci.py in {sources}"
        py_results = [r for r in results if "fibonacci.py" in r["source"]]
        if py_results:
            r = py_results[0]
            assert r.get("line_start", 0) > 0 or r.get("line_end", 0) > 0


class TestCrawl:
    def test_crawl_then_search(self, isolated_env):
        pytest.importorskip("crawl4ai")
        pytest.importorskip("pytest_httpserver")

        from pytest_httpserver import HTTPServer

        from lilbee import crawler as crawler_mod

        page_html = (
            "<html><head><title>Test</title></head>"
            "<body><h1>Quantum Entanglement</h1>"
            "<p>Quantum entanglement is a phenomenon where two particles "
            "become interconnected and instantaneously affect each other "
            "regardless of distance. Einstein called it spooky action.</p>"
            "</body></html>"
        )

        server = HTTPServer(host="127.0.0.1")
        server.expect_request("/quantum").respond_with_data(page_html, content_type="text/html")
        server.start()

        loopback_v4 = ipaddress.ip_network("127.0.0.0/8")
        loopback_v6 = ipaddress.ip_network("::1/128")
        filtered = tuple(
            net
            for net in crawler_mod.get_blocked_networks()
            if net not in (loopback_v4, loopback_v6)
        )
        original_fn = crawler_mod.get_blocked_networks
        crawler_mod.get_blocked_networks = lambda: filtered  # type: ignore[assignment]

        try:
            url = f"http://127.0.0.1:{server.port}/quantum"
            result = runner.invoke(app, ["add", url])
            assert result.exit_code == 0, result.output

            search_result = runner.invoke(app, ["--json", "search", "quantum entanglement"])
            assert search_result.exit_code == 0, search_result.output
            data = _parse_json_output(search_result.output)
            results = data.get("results", [])
            assert len(results) > 0, "Expected to find crawled content"
        finally:
            crawler_mod.get_blocked_networks = original_fn  # type: ignore[assignment]
            server.clear()
            if server.is_running():
                server.stop()


class TestStatusJson:
    def test_status_json(self, isolated_env):
        result = runner.invoke(app, ["--json", "status"])
        assert result.exit_code == 0, result.output
        data = _parse_json_output(result.output)
        assert "config" in data
        assert "chat_model" in data["config"]


class TestVersion:
    def test_version(self, isolated_env):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "lilbee" in result.output.lower() or "." in result.output


class TestSearchEmpty:
    def test_search_empty(self, isolated_env):
        result = runner.invoke(app, ["--json", "search", "anything at all"])
        assert result.exit_code == 0
        data = _parse_json_output(result.output)
        results = data.get("results", [])
        assert results == [] or isinstance(results, list)


class TestInit:
    def test_init(self, isolated_env, monkeypatch):
        target = isolated_env / "myproject"
        target.mkdir()
        monkeypatch.chdir(target)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert (target / ".lilbee").exists()
        assert (target / ".lilbee" / "documents").exists()
        assert (target / ".lilbee" / "data").exists()
