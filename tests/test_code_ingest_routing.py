"""Source files sync through the code chunker and land in the store as content type code.

The test syncs real source files of several languages into a real store in a tmp dir,
then reads the stored rows back. Only the embedder is a stand-in.
"""

from __future__ import annotations

from unittest import mock
from unittest.mock import MagicMock

import pytest

from lilbee.app.services import get_services, set_services
from lilbee.core.config import cfg
from lilbee.data.extract import code_chunker
from lilbee.data.ingest import sync
from lilbee.data.store import Store
from tests._profile_fixtures import (
    gate_releases_at_once,  # noqa: F401 -- autouse fixture, applied by import
    isolated_cfg,  # noqa: F401 -- autouse fixture, applied by import
)
from tests.conftest import make_mock_services

SOURCES = {
    "alpha.py": "def alpha():\n    return 1\n",
    "beta.js": "function beta() {\n  return 2;\n}\n",
    "gamma.go": "package gamma\n\nfunc Gamma() int {\n\treturn 3\n}\n",
    "delta.rs": "fn delta() -> i32 {\n    4\n}\n",
}


@pytest.fixture(autouse=True)
def real_store():
    embedder = MagicMock()
    embedder.validate_model.return_value = True
    embedder.embedding_available.return_value = True
    embedder.embed_batch.side_effect = lambda texts, **_kw: [
        [0.1] * cfg.embedding_dim for _ in texts
    ]
    set_services(
        make_mock_services(
            store=Store(cfg),
            embedder=embedder,
            worker_pool=MagicMock(),
            pool_runtime=MagicMock(),
            pool_health_ticker=MagicMock(),
        )
    )
    yield
    set_services(None)


@pytest.mark.usefixtures("code_parsers_ready")
async def test_code_files_sync_through_the_code_chunker_as_content_type_code():
    cfg.documents_dir.mkdir(parents=True, exist_ok=True)
    for name, text in SOURCES.items():
        (cfg.documents_dir / name).write_text(text, encoding="utf-8")

    with mock.patch("lilbee.data.ingest.code.chunk_code", wraps=code_chunker.chunk_code) as chunker:
        result = await sync(quiet=True)

    assert sorted(result.added) == sorted(SOURCES)
    assert sorted(call.kwargs["source_name"] for call in chunker.call_args_list) == sorted(SOURCES)

    store = get_services().store
    for name, text in SOURCES.items():
        rows = store.get_chunks_by_source(name)
        assert rows, f"{name} stored no chunks"
        assert {row.content_type for row in rows} == {"code"}, name
        # The chunker prepends a header naming the source; the text splitter does not.
        assert all(row.chunk.startswith(f"# File: {name}") for row in rows), name
        assert any(text.splitlines()[0] in row.chunk for row in rows), name
