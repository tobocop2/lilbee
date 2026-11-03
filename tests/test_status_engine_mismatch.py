"""Status names an embedding engine change and leaves the index otherwise as it is."""

from unittest.mock import MagicMock

import pytest

from lilbee.app import services as svc_mod
from lilbee.app.status import _index_status
from lilbee.data.store import EmbeddingEngineChangedError, EmbeddingModelMismatchError
from tests.conftest import make_mock_services

META = {"embedding_model": "org/embed-GGUF/embed.gguf", "embedding_dim": 768}


@pytest.fixture()
def store():
    store = MagicMock()
    store.get_meta.return_value = META
    store.index_mismatch.return_value = None
    store.canary_mismatch.return_value = None
    svc_mod.set_services(make_mock_services(store=store))
    yield store
    svc_mod.set_services(None)


def test_a_clean_index_has_no_mismatch(store):
    assert _index_status().engine_mismatch is None


def test_an_engine_change_is_named(store):
    store.canary_mismatch.return_value = EmbeddingEngineChangedError(
        model="m", dim=768, distance=0.2, verified=True
    )

    status = _index_status()

    assert "different vectors" in status.engine_mismatch
    assert status.embedding_model == META["embedding_model"]


def test_another_embedder_name_is_reported_as_before_and_the_canary_is_not_asked(store):
    store.index_mismatch.return_value = EmbeddingModelMismatchError(
        persisted_model="a", persisted_dim=8, current_model="b", current_dim=8
    )

    assert _index_status().engine_mismatch is None
    store.canary_mismatch.assert_not_called()


def test_an_index_with_no_meta_row_has_no_status(store):
    store.get_meta.return_value = None

    assert _index_status() is None
