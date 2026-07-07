import pytest

from lilbee.app import placement as app_placement
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.placement import InstancePlan
from lilbee.providers.fleet.placement_spec import PlacementSpec, RolePlacement
from lilbee.providers.fleet.planning import ResolvedPlacement
from lilbee.providers.roles import WorkerRole

GIB = 1024**3


def _resolved():
    return ResolvedPlacement(
        devices=(
            FleetDevice("CUDA", 0, "NVIDIA A100", 80 * GIB, 72 * GIB),
            FleetDevice("CUDA", 1, "NVIDIA A100", 80 * GIB, 80 * GIB),
        ),
        instances=(InstancePlan(role=WorkerRole.CHAT, devices=(0, 1), tensor_split=(1, 1)),),
        unplaceable_roles=(),
        model_refs={WorkerRole.CHAT: "org/chat.gguf"},
    )


def _resolved_with_skipped():
    from lilbee.providers.fleet.planning import ResolvedPlacement

    return ResolvedPlacement(
        devices=(FleetDevice("CUDA", 0, "NVIDIA A100", 80 * GIB, 72 * GIB),),
        instances=(InstancePlan(role=WorkerRole.EMBED, devices=(0,), tensor_split=None),),
        unplaceable_roles=(),
        model_refs={WorkerRole.EMBED: "org/embed.gguf"},
        skipped_not_installed={WorkerRole.CHAT: "org/Qwen3-4B.gguf"},
    )


def test_view_surfaces_skipped_not_installed(monkeypatch):
    from lilbee.app.placement import SkippedRole

    monkeypatch.setattr(
        app_placement, "resolve_placement_plan", lambda spec: _resolved_with_skipped()
    )
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    view = app_placement.get_placement()
    assert view.skipped_not_installed == (
        SkippedRole(role=WorkerRole.CHAT, model="org/Qwen3-4B.gguf"),
    )
    # The skipped role is absent from the placed roles (that is the bug it explains).
    assert all(r.role is not WorkerRole.CHAT for r in view.roles)


def test_view_has_no_skipped_when_all_installed(monkeypatch):
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    assert app_placement.get_placement().skipped_not_installed == ()


def _provider_with(role_ready, warm_phase):
    import unittest.mock as m

    from lilbee.providers.warm_progress import WarmProgress

    provider = m.MagicMock()
    provider.role_ready.return_value = role_ready
    provider.warm_progress.return_value = (
        WarmProgress(phase=warm_phase) if warm_phase is not None else None
    )
    services = m.MagicMock()
    services.provider = provider
    return services


def test_warm_progress_none_without_services(monkeypatch):
    monkeypatch.setattr(app_placement, "peek_services", lambda: None)
    assert app_placement.active_chat_warm_progress() is None


def test_warm_progress_none_when_role_ready(monkeypatch):
    monkeypatch.setattr(app_placement, "peek_services", lambda: _provider_with(True, None))
    assert app_placement.active_chat_warm_progress() is None


def test_warm_progress_returns_snapshot_during_active_phase(monkeypatch):
    from lilbee.providers.warm_progress import WarmPhase

    monkeypatch.setattr(
        app_placement, "peek_services", lambda: _provider_with(False, WarmPhase.READING_WEIGHTS)
    )
    snapshot = app_placement.active_chat_warm_progress()
    assert snapshot is not None
    assert snapshot.phase is WarmPhase.READING_WEIGHTS


def test_warm_progress_none_when_not_warming(monkeypatch):
    monkeypatch.setattr(app_placement, "peek_services", lambda: _provider_with(False, None))
    assert app_placement.active_chat_warm_progress() is None


def test_warm_progress_none_on_warm_error(monkeypatch):
    from lilbee.providers.warm_progress import WarmPhase

    monkeypatch.setattr(
        app_placement, "peek_services", lambda: _provider_with(False, WarmPhase.ERROR)
    )
    assert app_placement.active_chat_warm_progress() is None


def test_active_spec_reads_cfg(monkeypatch):
    monkeypatch.setattr(app_placement.cfg, "placement", None)
    assert app_placement._active_spec() is None


def test_active_spec_parses_stored_json(monkeypatch):
    spec = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0, 1))})
    monkeypatch.setattr(app_placement.cfg, "placement", spec.to_json())
    assert app_placement._active_spec() == spec


def test_get_placement_renders_view(monkeypatch):
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    view = app_placement.get_placement()
    assert view.manual is False
    assert view.gpus[0].label == "CUDA0"
    assert view.gpus[0].free_bytes == 72 * GIB
    assert view.roles[0].role is WorkerRole.CHAT
    assert view.roles[0].devices == (0, 1)


def test_preview_passes_candidate_spec(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        app_placement,
        "resolve_placement_plan",
        lambda spec: seen.update({"spec": spec}) or _resolved(),
    )
    cand = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))})
    app_placement.preview_placement(cand)
    assert seen["spec"] is cand


def test_preview_has_no_side_effects(monkeypatch):
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    peeked = {"called": False}
    monkeypatch.setattr(app_placement, "peek_services", lambda: peeked.__setitem__("called", True))
    app_placement.preview_placement(None)
    assert peeked["called"] is False  # no reload, no services touch


class _FakeProviderServices:
    """A peeked services container whose provider records reload_placement calls."""

    class _Provider:
        def __init__(self) -> None:
            self.reloads: list[bool] = []

        def reload_placement(self, *, wait: bool = False) -> None:
            self.reloads.append(wait)

    def __init__(self) -> None:
        self.provider = self._Provider()


def test_set_persists_and_reloads_live_fleet(monkeypatch):
    writes = {}
    services = _FakeProviderServices()
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    monkeypatch.setattr(app_placement.settings, "update_values", lambda root, d: writes.update(d))
    monkeypatch.setattr(app_placement, "peek_services", lambda: services)
    prior = app_placement.cfg.placement
    spec = PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0, 1), tensor_split=(1, 1))})
    try:
        app_placement.set_placement(spec)
        assert writes["placement"] == spec.to_json()
        # The live fleet applied the change surgically and synchronously.
        assert services.provider.reloads == [True]
        assert app_placement.cfg.placement == spec.to_json()
    finally:
        app_placement.cfg.placement = prior


def test_set_clears_read_device_cache_when_nothing_runs(monkeypatch):
    """With no services built, the next boot should probe devices fresh."""
    cleared = {"called": False}
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    monkeypatch.setattr(app_placement.settings, "update_values", lambda root, d: None)
    monkeypatch.setattr(app_placement, "peek_services", lambda: None)
    monkeypatch.setattr(
        app_placement, "clear_read_device_cache", lambda: cleared.__setitem__("called", True)
    )
    prior = app_placement.cfg.placement
    try:
        app_placement.set_placement(PlacementSpec({WorkerRole.EMBED: RolePlacement(devices=(0,))}))
        assert cleared["called"] is True
    finally:
        app_placement.cfg.placement = prior


def test_set_keeps_device_probe_on_the_live_path(monkeypatch):
    """The surgical reload diffs plans against the same clean-box probe (bb-a8f):
    re-probing under a loaded fleet would poison the chat context sizing."""
    cleared = {"called": False}
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    monkeypatch.setattr(app_placement.settings, "update_values", lambda root, d: None)
    monkeypatch.setattr(app_placement, "peek_services", _FakeProviderServices)
    monkeypatch.setattr(
        app_placement, "clear_read_device_cache", lambda: cleared.__setitem__("called", True)
    )
    prior = app_placement.cfg.placement
    try:
        app_placement.set_placement(PlacementSpec({WorkerRole.EMBED: RolePlacement(devices=(0,))}))
        assert cleared["called"] is False
    finally:
        app_placement.cfg.placement = prior


def test_set_none_clears(monkeypatch):
    deletes = {}
    monkeypatch.setattr(app_placement, "resolve_placement_plan", lambda spec: _resolved())
    monkeypatch.setattr(app_placement, "_active_spec", lambda: None)
    monkeypatch.setattr(
        app_placement.settings, "delete_values", lambda root, keys: deletes.setdefault("keys", keys)
    )
    monkeypatch.setattr(app_placement, "peek_services", lambda: None)
    prior = app_placement.cfg.placement
    try:
        app_placement.set_placement(None)
        assert deletes["keys"] == ["placement"]
        assert app_placement.cfg.placement is None
    finally:
        app_placement.cfg.placement = prior


def test_set_validates_before_persist(monkeypatch):
    from lilbee.providers.fleet.placement_spec import PlacementError

    def boom(spec):
        raise PlacementError("chat needs 70 GiB but device 0 has 40 GiB free")

    monkeypatch.setattr(app_placement, "resolve_placement_plan", boom)
    wrote = {"any": False}
    monkeypatch.setattr(
        app_placement.settings, "update_values", lambda root, d: wrote.__setitem__("any", True)
    )
    with pytest.raises(PlacementError):
        app_placement.set_placement(PlacementSpec({WorkerRole.CHAT: RolePlacement(devices=(0,))}))
    assert wrote["any"] is False


def test_view_multi_replica_unions_devices():
    resolved = ResolvedPlacement(
        devices=(
            FleetDevice("CUDA", 0, "NVIDIA A100", 80 * GIB, 72 * GIB),
            FleetDevice("CUDA", 1, "NVIDIA A100", 80 * GIB, 80 * GIB),
        ),
        instances=(
            InstancePlan(role=WorkerRole.EMBED, devices=(0,), tensor_split=None),
            InstancePlan(role=WorkerRole.EMBED, devices=(1,), tensor_split=None),
        ),
        unplaceable_roles=(),
        model_refs={WorkerRole.EMBED: "org/embed.gguf"},
    )
    view = app_placement._view(resolved, manual=False, spec_json=None)
    embed_views = [r for r in view.roles if r.role is WorkerRole.EMBED]
    assert len(embed_views) == 1
    role_view = embed_views[0]
    assert role_view.replicas == 2
    assert role_view.devices == (0, 1)


class _WaitProvider:
    """Provider stub scripting role_ready / warm_progress across polls."""

    def __init__(self, ready_after: int, snapshots: list) -> None:
        self._ready_after = ready_after
        self._snapshots = snapshots
        self.polls = 0

    def role_ready(self, role: WorkerRole) -> bool:
        self.polls += 1
        return self.polls > self._ready_after

    def warm_progress(self):
        idx = min(self.polls - 1, len(self._snapshots) - 1)
        return self._snapshots[idx] if self._snapshots else None


class _WaitServices:
    def __init__(self, provider: _WaitProvider) -> None:
        self.provider = provider


def test_wait_chat_ready_true_when_role_already_serves(monkeypatch):
    provider = _WaitProvider(ready_after=0, snapshots=[])
    monkeypatch.setattr(app_placement, "peek_services", lambda: _WaitServices(provider))
    assert app_placement.wait_chat_ready(timeout_s=5) is True


def test_wait_chat_ready_waits_out_an_active_warm(monkeypatch):
    from lilbee.providers.warm_progress import WarmPhase, WarmProgress

    loading = WarmProgress(phase=WarmPhase.LOADING_ENGINE)
    provider = _WaitProvider(ready_after=2, snapshots=[loading, loading])
    monkeypatch.setattr(app_placement, "peek_services", lambda: _WaitServices(provider))
    monkeypatch.setattr(app_placement, "_CHAT_READY_POLL_S", 0.01)
    assert app_placement.wait_chat_ready(timeout_s=5) is True
    assert provider.polls == 3  # two loading polls, then ready


def test_wait_chat_ready_stops_when_no_warm_in_flight(monkeypatch):
    # Not ready, nothing warming (stale finished snapshot): release after the
    # grace instead of holding the caller until the timeout.
    from lilbee.providers.warm_progress import WarmPhase, WarmProgress

    done = WarmProgress(phase=WarmPhase.READY)
    provider = _WaitProvider(ready_after=10_000, snapshots=[done])
    monkeypatch.setattr(app_placement, "peek_services", lambda: _WaitServices(provider))
    monkeypatch.setattr(app_placement, "_CHAT_READY_POLL_S", 0.01)
    monkeypatch.setattr(app_placement, "_CHAT_READY_GRACE_S", 0.0)
    assert app_placement.wait_chat_ready(timeout_s=5) is False


def test_wait_chat_ready_stops_on_warm_error(monkeypatch):
    from lilbee.providers.warm_progress import WarmPhase, WarmProgress

    failed = WarmProgress(phase=WarmPhase.ERROR, error="boom")
    provider = _WaitProvider(ready_after=10_000, snapshots=[failed])
    monkeypatch.setattr(app_placement, "peek_services", lambda: _WaitServices(provider))
    monkeypatch.setattr(app_placement, "_CHAT_READY_POLL_S", 0.01)
    monkeypatch.setattr(app_placement, "_CHAT_READY_GRACE_S", 0.0)
    assert app_placement.wait_chat_ready(timeout_s=5) is False


def test_wait_chat_ready_false_without_services(monkeypatch):
    monkeypatch.setattr(app_placement, "peek_services", lambda: None)
    assert app_placement.wait_chat_ready(timeout_s=5) is False


def test_wait_chat_ready_times_out_while_warm_never_finishes(monkeypatch):
    from lilbee.providers.warm_progress import WarmPhase, WarmProgress

    stuck = WarmProgress(phase=WarmPhase.LOADING_ENGINE)
    provider = _WaitProvider(ready_after=10_000, snapshots=[stuck])
    monkeypatch.setattr(app_placement, "peek_services", lambda: _WaitServices(provider))
    monkeypatch.setattr(app_placement, "_CHAT_READY_POLL_S", 0.005)
    assert app_placement.wait_chat_ready(timeout_s=0.02) is False
