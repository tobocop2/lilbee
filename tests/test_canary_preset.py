"""The canary trigger of a preset embedder: the preset URL stops a search, a key does not."""

import json
import socket
from unittest import mock

import pytest

from lilbee.app import presets
from lilbee.core import blocked_networks, settings
from lilbee.core.config import Config, cfg
from lilbee.data.store import EmbeddingEngineChangedError, Store
from lilbee.data.store.canary import CANARY_MAX_DISTANCE, CanaryProbe
from lilbee.providers.server_registry import route_for
from lilbee.retrieval import canary_probe as probe_module
from lilbee.retrieval.canary_probe import canary_probe, embedder_trigger
from tests.test_canary_probe import _config
from tests.test_store import _make_records
from tests.test_store_canary import Engine

URL_A = "http://10.0.0.7:9000/v1"
URL_B = "http://10.0.0.8:9000/v1"
KEY = "SENTINEL-key-77aa"
MODEL = "srv/embed-m"
T = CANARY_MAX_DISTANCE


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path):
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.server_presets = None
    yield
    for name in Config.model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture(autouse=True)
def _no_dns(monkeypatch):
    def getaddrinfo(host, _port):
        raise socket.gaierror("no answer")

    monkeypatch.setattr(blocked_networks.socket, "getaddrinfo", getaddrinfo)


@pytest.fixture()
def registry():
    return mock.MagicMock()


def _trigger(registry, model=MODEL):
    return embedder_trigger(_config(model), registry)


class TestPresetTrigger:
    def test_the_trigger_names_the_preset_its_url_and_the_model(self, registry):
        presets.add_preset("srv", URL_A, roles=["embed"])

        trigger = _trigger(registry)

        assert trigger is not None
        assert trigger.startswith("process:")
        assert trigger.endswith(f":srv@{URL_A}:{MODEL}")

    def test_the_trigger_is_stable_in_one_process(self, registry):
        presets.add_preset("srv", URL_A, roles=["embed"])

        assert _trigger(registry) == _trigger(registry)

    def test_a_url_edit_under_the_same_name_changes_the_trigger(self, registry):
        presets.add_preset("srv", URL_A, roles=["embed"])
        before = _trigger(registry)

        presets.edit_preset("srv", url=URL_B)

        assert _trigger(registry) != before

    def test_a_model_change_changes_the_trigger(self, registry):
        presets.add_preset("srv", URL_A, roles=["embed"])

        assert _trigger(registry, "srv/other") != _trigger(registry)

    def test_two_presets_at_two_urls_differ(self, registry):
        presets.add_preset("srv", URL_A, roles=["embed"])
        presets.add_preset("srw", URL_B, roles=["embed"])

        assert _trigger(registry, "srv/m") != _trigger(registry, "srw/m")

    def test_a_key_change_keeps_the_trigger(self, registry):
        presets.add_preset("srv", URL_A, roles=["embed"], api_key=KEY)
        before = _trigger(registry)

        presets.edit_preset("srv", api_key="another-key-91")

        assert _trigger(registry) == before

    def test_no_key_enters_the_trigger(self, registry):
        presets.add_preset("srv", URL_A, roles=["embed"], api_key=KEY)

        trigger = _trigger(registry)

        assert trigger is not None
        assert KEY not in trigger

    def test_a_login_and_a_query_in_the_url_stay_out_of_the_trigger(self, registry):
        entry = {
            "name": "srv",
            "url": "http://user:hunter2@10.0.0.7:9000/v1?token=abc#frag",
            "roles": ["embed"],
        }
        settings.set_value(cfg.data_root, "server_presets", json.dumps([entry]))

        trigger = _trigger(registry)

        assert trigger is not None
        assert trigger.endswith(f":srv@{URL_A}:{MODEL}")
        assert "hunter2" not in trigger
        assert "abc" not in trigger

    def test_a_preset_removed_between_the_parse_and_the_read_names_no_embedder(self, registry):
        presets.add_preset("srv", URL_A, roles=["embed"])
        with mock.patch.object(probe_module, "route_for", return_value=None):
            assert _trigger(registry) is None

    def test_a_built_in_remote_stays_process_scoped(self, registry):
        trigger = _trigger(registry, "openai/text-embedding-3-small")

        assert trigger is not None
        assert trigger.endswith(":openai/text-embedding-3-small")
        assert "@" not in trigger


class ServerEmbedder:
    """An embedder that answers from the server at the URL the preset holds now."""

    def __init__(self, spaces):
        self.spaces = spaces
        self.calls = 0

    def embed_batch(self, texts):
        self.calls += 1
        route = route_for("srv")
        return self.spaces[route.base_url].embed(texts)


@pytest.fixture()
def spaces():
    return {URL_A: Engine(), URL_B: Engine(distance=2 * T)}


@pytest.fixture()
def indexed(tmp_path, registry, spaces):
    """A library embedded through the preset at URL_A, and a way to open it again."""
    presets.add_preset("srv", URL_A, roles=["embed"], api_key=KEY)
    config = cfg.model_copy(
        update={"lancedb_dir": tmp_path / "lancedb_preset", "embedding_model": MODEL}
    )
    embedder = ServerEmbedder(spaces)

    def reopen(trigger=None):
        probe = canary_probe(config, embedder, registry)
        if trigger is not None:
            probe = CanaryProbe(embed=probe.embed, trigger=trigger)
        return Store(config, canary_probe=probe)

    store = reopen()
    store.add_chunks(_make_records())
    return store, reopen, embedder


class TestStopThroughTheStore:
    def test_a_url_edit_to_another_server_stops_search_and_asks_for_a_rebuild(self, indexed):
        store, reopen, _ = indexed
        presets.edit_preset("srv", url=URL_B, api_key=KEY)

        with pytest.raises(EmbeddingEngineChangedError, match="different vectors"):
            reopen().search([0.1] * cfg.embedding_dim)

    def test_a_url_edit_to_the_same_vectors_does_not_stop(self, indexed, spaces):
        store, reopen, _ = indexed
        spaces[URL_B] = Engine()
        presets.edit_preset("srv", url=URL_B, api_key=KEY)

        assert reopen().canary_mismatch() is None

    def test_a_key_edit_neither_stops_nor_embeds_a_canary(self, indexed):
        store, reopen, embedder = indexed
        presets.edit_preset("srv", api_key="another-key-91")
        embedder.calls = 0

        reopened = reopen()

        assert reopened.canary_mismatch() is None
        assert embedder.calls == 0

    def test_the_stop_ends_when_the_url_returns(self, indexed):
        store, reopen, _ = indexed
        presets.edit_preset("srv", url=URL_B, api_key=KEY)
        assert reopen().canary_mismatch() is not None

        presets.edit_preset("srv", url=URL_A, api_key=KEY)

        assert reopen().canary_mismatch() is None


class TestIndexWithAnOldTrigger:
    """An index written by an earlier release holds ``process:<token>:<model>``."""

    def test_an_old_trigger_with_the_same_server_serves_and_is_rewritten(self, indexed):
        store, reopen, embedder = indexed
        state = store.canary_state()
        assert state is not None
        old = f"process:oldtoken:{MODEL}"
        meta = store.get_meta()
        store._store_canary(meta, type(state)(state.vectors, old, state.verified))

        fresh = reopen()

        assert fresh.canary_mismatch() is None
        rewritten = fresh.canary_state()
        assert rewritten is not None
        assert rewritten.trigger.endswith(f":srv@{URL_A}:{MODEL}")
        assert rewritten.vectors == state.vectors
        assert embedder.calls >= 1

    def test_an_old_trigger_with_another_server_stops(self, indexed):
        store, reopen, _ = indexed
        state = store.canary_state()
        old = f"process:oldtoken:{MODEL}"
        store._store_canary(store.get_meta(), type(state)(state.vectors, old, state.verified))
        presets.edit_preset("srv", url=URL_B, api_key=KEY)

        assert reopen().canary_mismatch() is not None
