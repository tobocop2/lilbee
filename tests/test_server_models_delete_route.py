"""DELETE /api/models/{model} takes a model ref that holds slashes."""

from unittest import mock

import pytest
from litestar.testing import TestClient

from lilbee.app.models import RemoveResult
from lilbee.core.config import cfg

SLASH_REF = "bartowski/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    """Redirect config paths so the app never reads the real data folder."""
    snapshot = cfg.model_copy()
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.data_root = tmp_path
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    yield tmp_path
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture()
def client():
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    yield TestClient(create_app())
    auth_mod.session_manager.cleanup()


@pytest.fixture()
def installed_model():
    """An installed model double: the shared remove path reports the ref it was given."""

    def remove(ref, source=None):
        return RemoveResult(model=ref, deleted=True, freed_gb=4.5)

    with mock.patch("lilbee.app.models.remove_model_data", side_effect=remove) as patched:
        yield patched


class TestDeleteWithSlashRef:
    def test_a_ref_with_slashes_reaches_the_remove_path(self, client, installed_model):
        resp = client.delete(f"/api/models/{SLASH_REF}")
        assert resp.status_code == 200
        assert resp.json() == {"deleted": True, "model": SLASH_REF, "freed_gb": 4.5}
        assert installed_model.call_args.args[0] == SLASH_REF

    def test_the_ref_arrives_without_a_leading_slash(self, client, installed_model):
        client.delete(f"/api/models/{SLASH_REF}")
        assert not installed_model.call_args.args[0].startswith("/")

    def test_a_percent_encoded_ref_arrives_decoded(self, client, installed_model):
        resp = client.delete("/api/models/bartowski%2FQwen3-8B-GGUF%2FQwen3-8B-Q4_K_M.gguf")
        assert resp.status_code == 200
        assert installed_model.call_args.args[0] == SLASH_REF

    def test_a_ref_with_a_tag_arrives_whole(self, client, installed_model):
        resp = client.delete("/api/models/ollama/llama3:latest?source=ollama")
        assert resp.status_code == 200
        assert installed_model.call_args.args[0] == "ollama/llama3:latest"
        assert installed_model.call_args.args[1].value == "ollama"

    def test_a_ref_with_one_segment_still_matches(self, client, installed_model):
        resp = client.delete("/api/models/test")
        assert resp.status_code == 200
        assert installed_model.call_args.args[0] == "test"


class TestFixedModelPathsStayReachable:
    """The wider delete match must not capture the fixed sub-paths of /api/models/."""

    @pytest.mark.parametrize(
        "path", ["catalog", "installed", "external", "show", "pull", "chat", "reranker"]
    )
    def test_a_fixed_sub_path_still_answers_its_own_method(self, client, installed_model, path):
        resp = client.delete(f"/api/models/{path}")
        assert resp.status_code == 405
        assert installed_model.call_count == 0

    def test_get_installed_still_reaches_its_route(self, client):
        resp = client.get("/api/models/installed")
        assert resp.status_code == 200
        assert "models" in resp.json()
