"""Tests for the services container."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lilbee.app.engine_choice import EngineBook
from lilbee.core.config import cfg
from lilbee.providers.roles import WorkerRole


@pytest.fixture(autouse=True)
def isolated_cfg():
    snapshot = cfg.model_copy()
    yield
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture()
def parts(monkeypatch):
    """Stand in for every part ``build_services`` and ``reset_store`` make, to see each call."""
    import lilbee.data.extract.backends as backends_mod
    import lilbee.modelhub.role_validator as validator_mod
    import lilbee.retrieval.canary_probe as canary_mod
    import lilbee.retrieval.clustering as clustering_mod
    import lilbee.retrieval.concepts as concepts_mod
    import lilbee.retrieval.embedder as embedder_mod
    import lilbee.retrieval.query as query_mod
    import lilbee.retrieval.reranker as reranker_mod
    from lilbee.app import services as services_mod

    made = SimpleNamespace(
        store=MagicMock(name="new store"),
        canary_probe=MagicMock(name="canary_probe"),
        embedder=MagicMock(name="Embedder"),
        reranker=MagicMock(name="Reranker"),
        concepts=MagicMock(name="ConceptGraph"),
        clusterer=MagicMock(name="Clusterer"),
        searcher=MagicMock(name="Searcher"),
    )
    made.new_store = MagicMock(name="_new_store", return_value=made.store)
    monkeypatch.setattr(services_mod, "_new_store", made.new_store)
    monkeypatch.setattr(canary_mod, "canary_probe", made.canary_probe)
    monkeypatch.setattr(embedder_mod, "Embedder", made.embedder)
    monkeypatch.setattr(reranker_mod, "Reranker", made.reranker)
    monkeypatch.setattr(concepts_mod, "ConceptGraph", made.concepts)
    monkeypatch.setattr(clustering_mod, "Clusterer", made.clusterer)
    monkeypatch.setattr(query_mod, "Searcher", made.searcher)
    monkeypatch.setattr(backends_mod, "sync_xberg_backends", MagicMock())
    monkeypatch.setattr(validator_mod, "warn_unregistered_role_refs", MagicMock())
    return made


class TestSyncVisionOcrBackend:
    def _patch_xberg(self, monkeypatch, *, listed):
        reg = MagicMock()
        unreg = MagicMock()
        monkeypatch.setattr("xberg.list_ocr_backends", lambda: listed)
        monkeypatch.setattr("xberg.register_ocr_backend", reg)
        monkeypatch.setattr("xberg.unregister_ocr_backend", unreg)
        return reg, unreg

    def test_registers_when_model_set_and_absent(self, monkeypatch):
        from lilbee.data.extract.backends import BackendKind, sync_xberg_backend

        monkeypatch.setattr(cfg, "vision_model", "vendor/glm-ocr")
        reg, unreg = self._patch_xberg(monkeypatch, listed=["tesseract"])
        sync_xberg_backend(BackendKind.OCR, MagicMock())
        reg.assert_called_once()
        unreg.assert_not_called()

    def test_rebinds_to_current_provider_when_already_registered(self, monkeypatch):
        """A rebuilt provider must replace the stale binding: unregister then re-register.

        ``reset_services`` shuts the old provider down; if sync left the prior
        registration in place, xberg would keep routing OCR to the dead provider.
        """
        from lilbee.data.extract.backends import BackendKind, sync_xberg_backend

        monkeypatch.setattr(cfg, "vision_model", "vendor/glm-ocr")
        reg, unreg = self._patch_xberg(monkeypatch, listed=["lilbee-vision"])
        sync_xberg_backend(BackendKind.OCR, MagicMock())
        unreg.assert_called_once_with("lilbee-vision")
        reg.assert_called_once()

    def test_bind_backend_registers_once_per_provider(self, monkeypatch):
        """A second bind of the provider that is already registered changes nothing."""
        from lilbee.data.extract.backends import BackendKind, bind_backend

        listed: list[str] = []
        reg, unreg = self._patch_xberg(monkeypatch, listed=listed)
        reg.side_effect = lambda backend: listed.append(backend.name())
        provider = MagicMock()
        bind_backend(BackendKind.OCR, provider)
        bind_backend(BackendKind.OCR, provider)
        reg.assert_called_once()
        unreg.assert_not_called()

    def test_unregisters_when_model_cleared(self, monkeypatch):
        from lilbee.data.extract.backends import BackendKind, sync_xberg_backend

        monkeypatch.setattr(cfg, "vision_model", "")
        reg, unreg = self._patch_xberg(monkeypatch, listed=["lilbee-vision"])
        sync_xberg_backend(BackendKind.OCR, MagicMock())
        unreg.assert_called_once_with("lilbee-vision")
        reg.assert_not_called()

    def test_noop_when_no_model_and_absent(self, monkeypatch):
        from lilbee.data.extract.backends import BackendKind, sync_xberg_backend

        monkeypatch.setattr(cfg, "vision_model", "")
        reg, unreg = self._patch_xberg(monkeypatch, listed=["tesseract"])
        sync_xberg_backend(BackendKind.OCR, MagicMock())
        reg.assert_not_called()
        unreg.assert_not_called()

    def test_settings_role_reload_syncs_vision_backend(self, monkeypatch):
        """A vision_model change via any settings path (REST/MCP/TUI/CLI) registers it."""
        from lilbee.app.services import set_services
        from lilbee.app.settings import _reload_changed_roles
        from tests.conftest import make_mock_services

        set_services(make_mock_services())
        try:
            monkeypatch.setattr(cfg, "vision_model", "org/V-GGUF/v-Q4_K_M.gguf")
            reg, _unreg = self._patch_xberg(monkeypatch, listed=["tesseract"])
            _reload_changed_roles({"vision_model"})
            reg.assert_called_once()
        finally:
            set_services(None)


class TestServicesDataclass:
    def test_fields_are_immutable(self):
        from lilbee.app.services import CrawlerSyncState, Services

        services = Services(
            provider=MagicMock(),
            store=MagicMock(),
            embedder=MagicMock(),
            reranker=MagicMock(),
            concepts=MagicMock(),
            clusterer=MagicMock(),
            searcher=MagicMock(),
            registry=MagicMock(),
            hf_client=MagicMock(),
            ingest_lock_registry=MagicMock(),
            model_manager=MagicMock(),
            crawler_semaphore=None,
            crawler_sync_state=CrawlerSyncState(),
            known_models=MagicMock(),
            engines=EngineBook(cfg),
        )
        with pytest.raises(AttributeError):
            services.clusterer = MagicMock()  # type: ignore[misc]


class TestCancelInference:
    def test_delegates_to_provider(self):
        """``cancel_inference`` forwards to the provider's cancel hook."""
        from tests.conftest import make_mock_services

        provider = MagicMock()
        services = make_mock_services(provider=provider)
        services.cancel_inference()
        provider.cancel_inference.assert_called_once_with()


class TestReloadRole:
    def test_delegates_to_provider_with_role(self):
        """``reload_role`` forwards the requested role (and wait flag) to the provider."""
        from tests.conftest import make_mock_services

        provider = MagicMock()
        services = make_mock_services(provider=provider)
        services.reload_role(WorkerRole.EMBED)
        provider.reload_role.assert_called_once_with(WorkerRole.EMBED, wait=False)

    def test_forwards_wait_flag_to_provider(self):
        """``wait=True`` (the chat-swap path) is forwarded to the provider."""
        from tests.conftest import make_mock_services

        provider = MagicMock()
        services = make_mock_services(provider=provider)
        services.reload_role(WorkerRole.CHAT, wait=True)
        provider.reload_role.assert_called_once_with(WorkerRole.CHAT, wait=True)


class TestAddPoolListener:
    def test_forwards_callbacks_to_provider(self):
        """``add_pool_listener`` forwards both callbacks to the provider."""
        from tests.conftest import make_mock_services

        provider = MagicMock()
        services = make_mock_services(provider=provider)

        def on_spawning(_role: WorkerRole) -> None: ...

        def on_spawned(_role: WorkerRole) -> None: ...

        services.add_pool_listener(on_spawning=on_spawning, on_spawned=on_spawned)
        provider.add_spawn_listener.assert_called_once_with(
            on_spawning=on_spawning, on_spawned=on_spawned
        )


class TestWatchPool:
    """A pool watcher outlives the container it was first given, so a reset loses no hint."""

    @pytest.fixture()
    def builds(self, monkeypatch):
        """Make ``get_services`` build a container on a fresh recording provider each time."""
        from lilbee.app import services as services_mod
        from tests.conftest import make_mock_services

        providers: list[MagicMock] = []

        def build(
            config,
            provider=None,
            registry=None,
            interactive=False,
            config_refresh=None,
            held_writes=None,
            followed_keys=None,
        ):
            providers.append(MagicMock())
            return make_mock_services(provider=providers[-1])

        monkeypatch.setattr(services_mod, "build_services", build)
        monkeypatch.setattr("lilbee.app.settings.reconcile_embedding_dim", lambda registry: None)
        monkeypatch.setattr(cfg, "worker_pool_eager_start", False)
        services_mod.set_services(None)
        return providers

    def test_a_watcher_hears_the_container_built_after_a_reset(self, builds):
        from lilbee.app.services import get_services, reset_services, watch_pool

        def on_spawning(_role: WorkerRole) -> None: ...

        def on_spawned(_role: WorkerRole) -> None: ...

        get_services()
        watch_pool(on_spawning=on_spawning, on_spawned=on_spawned)
        reset_services()
        get_services()

        builds[1].add_spawn_listener.assert_called_once_with(
            on_spawning=on_spawning, on_spawned=on_spawned
        )

    def test_a_watcher_set_before_the_first_build_hears_it(self, builds):
        from lilbee.app.services import get_services, watch_pool

        def on_spawned(_role: WorkerRole) -> None: ...

        watch_pool(on_spawned=on_spawned)
        get_services()

        builds[0].add_spawn_listener.assert_called_once_with(
            on_spawning=None, on_spawned=on_spawned
        )

    def test_the_same_watcher_twice_is_heard_once(self, builds):
        from lilbee.app.services import get_services, watch_pool

        def on_spawned(_role: WorkerRole) -> None: ...

        get_services()
        watch_pool(on_spawned=on_spawned)
        watch_pool(on_spawned=on_spawned)

        builds[0].add_spawn_listener.assert_called_once_with(
            on_spawning=None, on_spawned=on_spawned
        )

    def test_the_watchers_are_attached_before_the_container_is_visible(self, builds, monkeypatch):
        from lilbee.app import services as services_mod
        from lilbee.app.services import get_services, watch_pool

        def on_spawned(_role: WorkerRole) -> None: ...

        visible_at_attach: list[bool] = []

        def record(**_kwargs: object) -> None:
            visible_at_attach.append(services_mod._state.singleton is not None)

        original_build = services_mod.build_services

        def build(*args, **kwargs):
            services = original_build(*args, **kwargs)
            services.provider.add_spawn_listener.side_effect = record
            return services

        monkeypatch.setattr(services_mod, "build_services", build)
        watch_pool(on_spawned=on_spawned)
        get_services()

        assert visible_at_attach == [False]


class TestEagerStartBranch:
    """``get_services`` warms the fleet when ``cfg.worker_pool_eager_start`` is set."""

    def test_eager_start_warms_provider_when_flag_set(self, monkeypatch):
        cfg.worker_pool_eager_start = True

        from lilbee.app import services as services_mod

        services_mod.set_services(None)
        provider = MagicMock()
        monkeypatch.setattr(
            "lilbee.providers.factory.create_provider", lambda _cfg, **_kw: provider
        )
        try:
            services_mod.get_services()
        finally:
            services_mod.set_services(None)
        provider.warm_up_pool.assert_called_once_with()

    def test_eager_start_swallows_warm_up_failure(self, monkeypatch):
        """suppress(Exception) keeps get_services() resilient if warm-up raises."""
        cfg.worker_pool_eager_start = True

        from lilbee.app import services as services_mod

        services_mod.set_services(None)
        provider = MagicMock()
        provider.warm_up_pool.side_effect = RuntimeError("simulated warm-up failure")
        monkeypatch.setattr(
            "lilbee.providers.factory.create_provider", lambda _cfg, **_kw: provider
        )
        try:
            svc = services_mod.get_services()
            assert svc is not None
        finally:
            services_mod.set_services(None)

    def test_no_warm_up_when_flag_clear(self, monkeypatch):
        cfg.worker_pool_eager_start = False

        from lilbee.app import services as services_mod

        services_mod.set_services(None)
        provider = MagicMock()
        monkeypatch.setattr(
            "lilbee.providers.factory.create_provider", lambda _cfg, **_kw: provider
        )
        try:
            services_mod.get_services()
        finally:
            services_mod.set_services(None)
        provider.warm_up_pool.assert_not_called()


class TestFirstBuildUnderTokenSizing:
    def test_first_build_finishes_when_counting_resolves_through_the_container(self, monkeypatch):
        """The fleet's count resolves its model through get_services(); counting while
        the container is first built re-entered its creation lock and never returned."""
        import threading

        from lilbee.app import services as services_mod
        from lilbee.providers.engine_params import resolve_model_path

        cfg.worker_pool_eager_start = False
        cfg.embedding_model = "org/embed-GGUF/embed.gguf"
        provider = MagicMock()

        def count_like_the_fleet(_text: str) -> int:
            resolve_model_path(cfg.embedding_model)
            return 7

        provider.count_tokens.side_effect = count_like_the_fleet
        monkeypatch.setattr(
            "lilbee.providers.factory.create_provider", lambda _cfg, **_kw: provider
        )
        services_mod.set_services(None)
        built: list[object] = []
        worker = threading.Thread(
            target=lambda: built.append(services_mod.get_services()), daemon=True
        )
        try:
            worker.start()
            worker.join(timeout=10)
            assert not worker.is_alive(), "get_services() deadlocked on its own build"
            assert built[0].provider is provider
        finally:
            services_mod.set_services(None)


class TestBuildServices:
    """``build_services`` constructs a full container bound to an arbitrary Config."""

    def test_builds_full_container_with_supplied_provider(self, tmp_path):
        from lilbee.app.services import Services, build_services

        cfg.data_root = tmp_path
        cfg.documents_dir = tmp_path / "documents"
        cfg.data_dir = tmp_path / "data"
        cfg.lancedb_dir = tmp_path / "data" / "lancedb"
        cfg.documents_dir.mkdir(parents=True, exist_ok=True)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)

        provider = MagicMock()
        services = build_services(cfg, provider=provider)
        try:
            assert isinstance(services, Services)
            # The supplied provider is threaded through, not rebuilt via the factory.
            assert services.provider is provider
            assert services.store is not None
            assert services.searcher is not None
        finally:
            services.store.close()

    def test_builds_each_part_from_the_config_and_hands_it_to_the_services(self, parts):
        from lilbee.app.services import build_services

        config = cfg.model_copy()
        provider = MagicMock(name="provider")
        registry = MagicMock(name="registry")

        services = build_services(config, provider=provider, registry=registry)

        embedder = parts.embedder.return_value
        reranker = parts.reranker.return_value
        concepts = parts.concepts.return_value
        parts.embedder.assert_called_once_with(config, provider)
        parts.canary_probe.assert_called_once_with(config, embedder, registry)
        parts.new_store.assert_called_once_with(config, parts.canary_probe.return_value)
        parts.reranker.assert_called_once_with(config)
        parts.concepts.assert_called_once_with(config, parts.store)
        parts.clusterer.assert_called_once_with(config, parts.store)
        parts.searcher.assert_called_once_with(
            config, provider, parts.store, embedder, reranker, concepts
        )
        assert services.provider is provider
        assert services.registry is registry
        assert services.store is parts.store
        assert services.embedder is embedder
        assert services.reranker is reranker
        assert services.concepts is concepts
        assert services.clusterer is parts.clusterer.return_value
        assert services.searcher is parts.searcher.return_value

    @pytest.mark.parametrize(("limit", "slots"), [(0, None), (1, 1), (2, 2)])
    def test_the_crawl_semaphore_follows_crawl_max_concurrent(self, parts, limit, slots):
        from lilbee.app.services import build_services

        config = cfg.model_copy()
        config.crawl_max_concurrent = limit

        semaphore = build_services(config, provider=MagicMock()).crawler_semaphore

        if slots is None:
            assert semaphore is None
            return
        assert semaphore is not None

        async def taken_while_free() -> list[bool]:
            states = []
            for _ in range(slots):
                states.append(semaphore.locked())
                await semaphore.acquire()
            states.append(semaphore.locked())
            return states

        # Free until the last of its slots is taken, then full: one slot for a limit of one.
        assert asyncio.run(taken_while_free()) == [False] * slots + [True]

    @staticmethod
    def _a_build_that_fails_after_the_store(monkeypatch, tmp_path, new_store):
        """Make ``build_services`` fail on the concept graph, after the store exists."""
        import lilbee.retrieval.canary_probe as canary_mod
        import lilbee.retrieval.concepts as concepts_mod
        from lilbee.app import services as services_mod

        cfg.data_root = tmp_path
        cfg.documents_dir = tmp_path / "documents"
        cfg.data_dir = tmp_path / "data"
        cfg.lancedb_dir = tmp_path / "data" / "lancedb"
        monkeypatch.setattr(canary_mod, "canary_probe", MagicMock())
        monkeypatch.setattr(services_mod, "_new_store", lambda *_args: new_store)
        monkeypatch.setattr(concepts_mod, "ConceptGraph", MagicMock(side_effect=OSError("boom")))

    def test_a_failed_build_closes_its_store_and_leaves_a_supplied_provider_open(
        self, monkeypatch, tmp_path
    ):
        from lilbee.app.services import build_services

        new_store = MagicMock()
        self._a_build_that_fails_after_the_store(monkeypatch, tmp_path, new_store)
        provider = MagicMock()

        with pytest.raises(OSError, match="boom"):
            build_services(cfg, provider=provider)

        new_store.close.assert_called_once_with()
        provider.shutdown.assert_not_called()

    def test_a_failed_build_shuts_down_the_provider_it_created(self, monkeypatch, tmp_path):
        import lilbee.providers.factory as factory_mod
        from lilbee.app.services import build_services

        new_store = MagicMock()
        self._a_build_that_fails_after_the_store(monkeypatch, tmp_path, new_store)
        created = MagicMock()
        monkeypatch.setattr(factory_mod, "create_provider", lambda *_a, **_k: created)

        with pytest.raises(OSError, match="boom"):
            build_services(cfg)

        new_store.close.assert_called_once_with()
        created.shutdown.assert_called_once_with()

    def test_an_interrupt_during_the_build_still_cleans_up(self, monkeypatch, tmp_path):
        import lilbee.retrieval.concepts as concepts_mod
        from lilbee.app.services import build_services

        new_store = MagicMock()
        self._a_build_that_fails_after_the_store(monkeypatch, tmp_path, new_store)
        monkeypatch.setattr(concepts_mod, "ConceptGraph", MagicMock(side_effect=KeyboardInterrupt))
        provider = MagicMock()

        with pytest.raises(KeyboardInterrupt):
            build_services(cfg, provider=provider)

        new_store.close.assert_called_once_with()

    def test_a_cleanup_error_is_logged_and_the_build_error_stays(
        self, monkeypatch, tmp_path, caplog
    ):
        from lilbee.app.services import build_services

        new_store = MagicMock()
        new_store.close.side_effect = RuntimeError("close failed")
        self._a_build_that_fails_after_the_store(monkeypatch, tmp_path, new_store)

        with caplog.at_level("WARNING"), pytest.raises(OSError, match="boom"):
            build_services(cfg, provider=MagicMock())

        assert "Cleanup after a failed services build failed" in caplog.text


class TestServicesScope:
    """``services_scope`` shadows the singleton for the block without touching it."""

    def test_scope_overrides_get_services_and_resets(self):
        from lilbee.app import services as services_mod
        from lilbee.app.services import get_services, services_scope
        from tests.conftest import make_mock_services

        override = make_mock_services()
        services_mod.set_services(None)
        with services_scope(override):
            # get_services resolves the scoped override without building the global.
            assert get_services() is override
            assert services_mod.peek_services() is None
        # The override is reset on exit and never leaked into the global singleton.
        assert services_mod._state.override.get() is None
        assert services_mod.peek_services() is None


class TestResetServicesSwapBeforeClose:
    """reset_services clears the singleton before tearing the old one down."""

    def test_reference_cleared_before_teardown(self):
        from lilbee.app import services as services_mod
        from tests.conftest import make_mock_services

        observed: list[bool] = []
        store = MagicMock()
        provider = MagicMock()
        # When teardown runs, the module singleton must already be None so a
        # concurrent get_services() never hands out a closing container.
        store.close.side_effect = lambda: observed.append(services_mod.peek_services() is None)

        services_mod.set_services(make_mock_services(store=store, provider=provider))
        try:
            services_mod.reset_services()
            assert services_mod.peek_services() is None
            provider.shutdown.assert_called_once()
            store.close.assert_called_once()
            assert observed == [True]  # cleared before close
        finally:
            services_mod.set_services(None)


class TestResetStore:
    def test_keeps_provider_and_embedder_replaces_store(self, tmp_path):
        """``reset_store`` rebuilds Store-bound services without unloading the provider."""
        from lilbee.app import services as services_mod
        from lilbee.app.services import get_services, reset_services, reset_store

        cfg.data_root = tmp_path
        cfg.documents_dir = tmp_path / "documents"
        cfg.data_dir = tmp_path / "data"
        cfg.lancedb_dir = tmp_path / "data" / "lancedb"
        cfg.documents_dir.mkdir(parents=True, exist_ok=True)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)

        try:
            reset_services()
            before = get_services()
            old_store = before.store
            old_concepts = before.concepts
            old_searcher = before.searcher
            old_provider = before.provider
            old_embedder = before.embedder
            old_reranker = before.reranker
            old_registry = before.registry
            old_model_manager = before.model_manager

            reset_store()

            after = services_mod.peek_services()
            assert after is not None
            assert after.store is not old_store
            assert after.concepts is not old_concepts
            assert after.searcher is not old_searcher
            # Heavy singletons stay loaded.
            assert after.provider is old_provider
            assert after.embedder is old_embedder
            assert after.reranker is old_reranker
            assert after.registry is old_registry
            assert after.model_manager is old_model_manager
        finally:
            reset_services()

    def test_rebuilds_the_graph_clusterer_searcher_and_services_from_the_config_and_new_store(
        self, parts
    ):
        """The rebuilt parts take the config and the new store, and the services hold them."""
        from lilbee.app import services as services_mod
        from tests.conftest import make_mock_services

        old = make_mock_services(
            provider=MagicMock(name="provider"),
            embedder=MagicMock(name="embedder"),
            reranker=MagicMock(name="reranker"),
            registry=MagicMock(name="registry"),
            searcher=MagicMock(name="old searcher"),
        )
        services_mod.set_services(old)
        concepts = parts.concepts.return_value
        try:
            services_mod.reset_store()
            after = services_mod.peek_services()
            parts.canary_probe.assert_called_once_with(cfg, old.embedder, old.registry)
            parts.new_store.assert_called_once_with(cfg, parts.canary_probe.return_value)
            parts.concepts.assert_called_once_with(cfg, parts.store)
            parts.clusterer.assert_called_once_with(cfg, parts.store)
            parts.searcher.assert_called_once_with(
                cfg, old.provider, parts.store, old.embedder, old.reranker, concepts
            )
            assert after.store is parts.store
            assert after.concepts is concepts
            assert after.clusterer is parts.clusterer.return_value
            assert after.searcher is parts.searcher.return_value
            old.store.close.assert_called_once_with()
        finally:
            services_mod.set_services(None)

    def test_no_op_when_services_uncached(self):
        """``reset_store`` is a no-op if Services has not been built yet."""
        from lilbee.app import services as services_mod
        from lilbee.app.services import reset_services, reset_store

        reset_services()
        assert services_mod.peek_services() is None
        reset_store()
        assert services_mod.peek_services() is None


class TestWaitForOpens:
    """``wait_for_opens`` covers every store this process built, also one /reset made."""

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        """Start from no recorded stores, and put back the ones that were there."""
        from lilbee.app.services import _state

        with _state.stores_lock:
            saved = list(_state.stores)
            _state.stores.clear()
        yield
        with _state.stores_lock:
            _state.stores.clear()
            _state.stores.update(saved)

    @pytest.fixture()
    def held_open(self, monkeypatch):
        """Make every Store open block until *release* is set; *started* marks an open under way."""
        import threading

        from lilbee.data.store import Store

        started = threading.Event()
        release = threading.Event()

        def blocked_open(self):
            started.set()
            release.wait(timeout=30)
            self._db = MagicMock()
            return self._db

        monkeypatch.setattr(Store, "_open_held", blocked_open)
        yield started, release
        release.set()

    def test_waits_for_an_open_on_the_store_reset_made(self, held_open, tmp_path):
        import threading

        from lilbee.app import services as services_mod
        from tests.conftest import make_mock_services

        started, release = held_open
        services_mod.set_services(make_mock_services(store=services_mod._new_store(cfg)))
        try:
            services_mod.reset_store()
            new_store = services_mod.peek_services().store
            opener = threading.Thread(target=new_store.get_db, daemon=True)
            opener.start()
            assert started.wait(timeout=10)
            waiter_result: list[bool] = []
            waiter = threading.Thread(
                target=lambda: waiter_result.append(services_mod.wait_for_opens(10.0)),
                daemon=True,
            )
            waiter.start()
            waiter.join(timeout=0.3)
            assert waiter.is_alive()  # still waiting while the open runs
            release.set()
            waiter.join(timeout=10)
            opener.join(timeout=10)
            assert waiter_result == [True]
            assert new_store.is_open
        finally:
            release.set()
            services_mod.set_services(None)

    def test_waits_for_the_first_store_open(self, held_open):
        import threading

        from lilbee.app import services as services_mod

        started, release = held_open
        store = services_mod._new_store(cfg)
        store.open_in_background()
        assert started.wait(timeout=10)
        result: list[bool] = []
        waiter = threading.Thread(
            target=lambda: result.append(services_mod.wait_for_opens(10.0)), daemon=True
        )
        waiter.start()
        waiter.join(timeout=0.3)
        assert waiter.is_alive()
        release.set()
        waiter.join(timeout=10)
        assert result == [True]

    def test_the_bound_is_shared_by_all_opens(self, monkeypatch):
        from lilbee.app import services as services_mod
        from lilbee.data.store import Store

        clock = [100.0]
        waits: list[float] = []

        def hanging_wait(self, timeout):
            waits.append(timeout)
            clock[0] += timeout  # the open outlasts whatever it is given
            return False

        monkeypatch.setattr(services_mod.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(Store, "wait_for_open", hanging_wait)
        first = services_mod._new_store(cfg)
        second = services_mod._new_store(cfg)
        assert services_mod.wait_for_opens(5.0) is False
        assert sorted(waits) == [0.0, 5.0]  # the second gets what the first left
        assert first is not second

    def test_returns_at_once_when_no_open_runs(self):
        import time

        from lilbee.app import services as services_mod

        services_mod._new_store(cfg)
        begin = time.monotonic()
        assert services_mod.wait_for_opens(5.0) is True
        assert time.monotonic() - begin < 1.0


class TestGetServicesThreadSafety:
    def test_concurrent_first_touch_builds_one_container(self, monkeypatch):
        """Concurrent first ``get_services()`` calls must build exactly one container.

        A duplicate concurrent build re-registers xberg's process-global backends
        mid-flight, and the losing thread raises 'Embedding backend already
        registered' -- surfacing as failed download tasks when several workers
        first-touch services at once."""
        import threading
        import time

        from lilbee.app import services as services_mod
        from tests.conftest import make_mock_services

        builds: list[int] = []

        def slow_build(
            config,
            provider=None,
            registry=None,
            interactive=False,
            config_refresh=None,
            held_writes=None,
            followed_keys=None,
        ):
            builds.append(threading.get_ident())
            time.sleep(0.05)
            return make_mock_services()

        monkeypatch.setattr(services_mod, "build_services", slow_build)
        monkeypatch.setattr("lilbee.app.settings.reconcile_embedding_dim", lambda registry: None)
        monkeypatch.setattr(cfg, "worker_pool_eager_start", False)
        services_mod._state.singleton = None

        results: list[object] = []
        errors: list[BaseException] = []

        def touch() -> None:
            try:
                results.append(services_mod.get_services())
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=touch) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(builds) == 1
        assert len({id(r) for r in results}) == 1


class TestCanaryWiring:
    """The stores of the container embed their canaries with the embedder of the container."""

    @pytest.fixture()
    def counted_embeds(self, monkeypatch, tmp_path):
        from lilbee.retrieval.embedder import Embedder

        cfg.data_root = tmp_path
        cfg.documents_dir = tmp_path / "documents"
        cfg.data_dir = tmp_path / "data"
        cfg.lancedb_dir = tmp_path / "data" / "lancedb"
        cfg.documents_dir.mkdir(parents=True, exist_ok=True)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(cfg, "embedding_model", "ollama/canary-test")
        calls: list[list[str]] = []

        def _embed(self, texts, **kwargs):
            calls.append(list(texts))
            return [[1.0, float(i)] for i, _ in enumerate(texts)]

        monkeypatch.setattr(Embedder, "embed_batch", _embed)
        return calls

    @staticmethod
    def _record(dim):
        return {
            "source": "a.md",
            "content_type": "text",
            "chunk_type": "raw",
            "page_start": 0,
            "page_end": 0,
            "line_start": 0,
            "line_end": 0,
            "chunk": "text",
            "chunk_index": 0,
            "vector": [0.5] * dim,
        }

    def test_a_built_container_stores_a_baseline_at_the_first_write(self, counted_embeds):
        from lilbee.app.services import build_services
        from lilbee.data.store.canary import CANARY_TEXTS

        services = build_services(cfg, provider=MagicMock())
        try:
            services.store.add_chunks([self._record(cfg.embedding_dim)])

            assert services.store.canary_state() is not None
            assert counted_embeds == [list(CANARY_TEXTS)]
        finally:
            services.store.close()

    def test_a_store_rebuilt_in_place_keeps_the_check(self, counted_embeds):
        from lilbee.app.services import get_services, reset_services, reset_store

        try:
            reset_services()
            get_services()
            reset_store()

            get_services().store.add_chunks([self._record(cfg.embedding_dim)])

            assert get_services().store.canary_state() is not None
            assert len(counted_embeds) == 1
        finally:
            reset_services()
