"""Integration test configuration — override unit test isolation for models_dir.

The root conftest._isolate_cfg sets models_dir to a temp dir for every test,
which breaks integration tests that need real models from the canonical location.
This fixture restores models_dir after the root fixture changes it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from lilbee.config import cfg
from lilbee.platform import canonical_models_dir

SPECS_MD = """\
# Thunderbolt X500

Engine: 3.5L V6 TurboForce
Oil capacity: 6.5 quarts
Top speed: 155 mph
Horsepower: 365 hp
Transmission: 8-speed automatic
"""

AUTH_PART1_MD = """\
# Authentication: OAuth Setup

Configure OAuth 2.0 with client ID and secret. Register your application
in the developer portal to obtain credentials. The authorization endpoint
handles the initial redirect flow with PKCE challenge.
"""

AUTH_PART2_MD = """\
# Authentication: JWT Tokens

JWT tokens are signed with RS256 algorithm using asymmetric keys.
Access tokens expire after 15 minutes. Refresh tokens are stored
securely and rotated on each use for replay attack prevention.
"""

AUTH_PART3_MD = """\
# Authentication: Session Management

Sessions stored in Redis with 24h TTL. Session IDs are generated
using cryptographically secure random bytes. Inactive sessions
are garbage collected every 6 hours by the cleanup worker.
"""

DEPLOY_MD = """\
# Deployment Guide

Use kubectl apply to deploy containers to the Kubernetes cluster.
The CI/CD pipeline builds Docker images tagged with the git SHA.
Rolling updates ensure zero downtime during releases. Configure
resource limits and health checks in the deployment manifest.
"""

DB_PERF_MD = """\
# Database Performance

Index your queries. Use connection pooling with PgBouncer to reduce
connection overhead. Monitor slow queries with pg_stat_statements.
Vacuum and analyze tables regularly. Partition large tables by date
for faster range scans.
"""

API_PERF_MD = """\
# API Performance

Cache responses with Redis. Use connection pooling for upstream services.
Rate limit with token bucket algorithm to prevent abuse. Enable gzip
compression for large payloads. Use async I/O for non-blocking requests.
"""

FIBONACCI_PY = '''\
"""Fibonacci sequence calculator."""


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number iteratively."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''

NOTES_MD = """\
# Meeting Notes March 2026

Decided to migrate to PostgreSQL 16 for improved JSON support.
Timeline: Q2 2026. Migration lead: Sarah Chen. Budget approved
for dedicated DBA contractor. Rollback plan documented in wiki.
"""


def setup_rag_pipeline(tmp: Path, *, wiki: bool = False) -> dict:
    """Common pipeline setup: write docs, configure cfg, download models, sync.

    Returns data dict for use in test fixtures.
    """
    from lilbee.catalog import FEATURED_CHAT, FEATURED_EMBEDDING, download_model
    from lilbee.ingest import sync
    from lilbee.model_manager import reset_model_manager
    from lilbee.services import reset_services

    docs_dir = tmp / "documents"
    data_dir = tmp / "data"
    lancedb_dir = data_dir / "lancedb"
    docs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    test_docs = {
        "specs.md": SPECS_MD,
        "auth-part1.md": AUTH_PART1_MD,
        "auth-part2.md": AUTH_PART2_MD,
        "auth-part3.md": AUTH_PART3_MD,
        "deploy.md": DEPLOY_MD,
        "db-perf.md": DB_PERF_MD,
        "api-perf.md": API_PERF_MD,
        "fibonacci.py": FIBONACCI_PY,
        "notes.md": NOTES_MD,
    }
    for name, content in test_docs.items():
        (docs_dir / name).write_text(content)

    cfg.llm_provider = "llama-cpp"
    cfg.documents_dir = docs_dir
    cfg.data_dir = data_dir
    cfg.data_root = tmp
    cfg.lancedb_dir = lancedb_dir
    cfg.query_expansion_count = 0
    cfg.concept_graph = False
    cfg.hyde = False

    if wiki:
        cfg.wiki = True
        cfg.wiki_dir = "wiki"
        wiki_root = tmp / "wiki"
        wiki_root.mkdir(parents=True, exist_ok=True)

    reset_services()
    reset_model_manager()

    embed_entry = FEATURED_EMBEDDING[0]
    download_model(embed_entry)
    cfg.embedding_model = embed_entry.ref

    chat_entry = FEATURED_CHAT[0]
    download_model(chat_entry)
    cfg.chat_model = chat_entry.ref

    result = asyncio.run(sync(quiet=True))

    return {
        "tmp": tmp,
        "docs_dir": docs_dir,
        "data_dir": data_dir,
        "lancedb_dir": lancedb_dir,
        "sync_result": result,
        "test_docs": test_docs,
    }


@pytest.fixture(autouse=True)
def _preserve_models_dir():
    """Ensure models_dir stays at canonical location for integration tests."""
    cfg.models_dir = canonical_models_dir()
    yield
