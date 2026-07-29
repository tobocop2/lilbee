from pathlib import Path

import pytest
from litestar.testing import AsyncTestClient

from lilbee.core.config import cfg
from tests.server.test_wiki_routes import _create_app, _h


@pytest.fixture(autouse=True)
def env(tmp_path: Path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.wiki = True
    cfg.wiki_dir = "wiki"
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.mark.asyncio
async def test_status():
    async with AsyncTestClient(_create_app()) as client:
        resp = await client.post("/api/wiki/build", params={"dry_run": True}, headers=_h())
    print("STATUS", resp.status_code, resp.headers["content-type"])
