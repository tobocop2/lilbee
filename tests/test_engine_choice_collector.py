"""The collector that gives each role its engine, and the installed mapping behind it."""

from __future__ import annotations

import hashlib
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from lilbee.app import engine_choice
from lilbee.app.engine_choice import engine_for_format, host_facts, resolve_engines
from lilbee.catalog.types import ModelFormat
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet.engine_store import EngineStore, PinLock, engines_root
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.base import EngineName, LaunchRequest
from lilbee.providers.fleet.engines.choice import (
    ADAPTERS,
    EngineAssignment,
    HostFacts,
    NoEngineError,
    choose_engine,
    installed_engines,
)
from lilbee.providers.roles import ROLE_REGISTRY, WorkerRole
from tests.conformance.fake_adapter import BundledFakeAdapter, FakeAdapter


def test_installed_mapping_holds_the_formats_each_adapter_loads() -> None:
    installed = installed_engines()
    assert installed
    bundled = {name: adapter for name, adapter in ADAPTERS.items() if adapter.bundled}
    assert bundled
    assert set(bundled) < set(ADAPTERS)
    assert set(installed) == set(bundled)
    for name, adapter in bundled.items():
        assert installed[name] == adapter.formats
    assert installed[EngineName.LLAMA] == frozenset({ModelFormat.GGUF})


def test_every_role_runs_on_the_default_engine_with_no_setting() -> None:
    assignment = resolve_engines(Config())
    assert set(assignment.choices) == set(ROLE_REGISTRY)
    for role in ROLE_REGISTRY:
        engine = assignment.for_role(role)
        assert engine.name is EngineName.LLAMA
        assert engine.adapter is LLAMA
        assert engine.notice is None


def test_launch_through_the_collector_equals_the_adapter_launch() -> None:
    request = LaunchRequest(
        binary=Path("/b/llama-server"),
        model_path=Path("/m/x.gguf"),
        devices=(0,),
        n_gpu_layers=99,
        slots=2,
        ctx_per_slot=4096,
    )
    chosen = resolve_engines(Config()).for_role(WorkerRole.CHAT).adapter
    spec = chosen.role_spec(WorkerRole.CHAT, None, None)
    assert chosen.launch_argv(spec, request) == LLAMA.launch_argv(
        LLAMA.role_spec(WorkerRole.CHAT, None, None), request
    )


def test_two_roles_hold_two_engines() -> None:
    assignment = EngineAssignment.for_test({"chat": "llama.cpp", "embed": "mlx"})
    assert assignment.for_role(WorkerRole.CHAT).name is EngineName.LLAMA
    assert assignment.for_role(WorkerRole.EMBED).name is EngineName.MLX


def test_a_role_with_no_engine_raises_and_the_others_still_answer() -> None:
    config = Config()
    config.embedding_model = "org/embedder#mlx"
    assignment = resolve_engines(config)
    with pytest.raises(NoEngineError, match="For embed: No installed engine loads the mlx format"):
        assignment.for_role(WorkerRole.EMBED)
    assert assignment.for_role(WorkerRole.CHAT).name is EngineName.LLAMA


def test_the_notice_names_the_role_as_an_input_of_the_text() -> None:
    kwargs = dict(
        fmt=ModelFormat.GGUF,
        host=HostFacts(system="linux", machine="x86_64"),
        installed={EngineName.LLAMA: frozenset({ModelFormat.GGUF})},
        preferred="mlx",
        override=None,
    )
    with_role = choose_engine(**kwargs, role=WorkerRole.CHAT)
    assert with_role.notice == (
        "Engine mlx for chat is not installed. Using llama.cpp. "
        "Run lilbee engine list to see the installed engines."
    )
    assert choose_engine(**kwargs).notice == (
        "Engine mlx is not installed. Using llama.cpp. "
        "Run lilbee engine list to see the installed engines."
    )
    none = choose_engine(**{**kwargs, "fmt": ModelFormat.MLX, "preferred": "auto"})
    assert none.notice == (
        "No installed engine loads the mlx format. "
        "Run lilbee engine list to see the installed engines."
    )
    with_model = choose_engine(
        **{**kwargs, "fmt": ModelFormat.MLX, "preferred": "auto"}, model="org/embedder#mlx"
    )
    assert with_model.notice == (
        "No installed engine loads the mlx format of org/embedder#mlx. "
        "Run lilbee engine list to see the installed engines."
    )
    named = choose_engine(
        **{**kwargs, "fmt": ModelFormat.MLX, "preferred": "auto"}, role=WorkerRole.EMBED
    )
    assert named.notice == f"For embed: {none.notice}"


def test_host_facts_are_lower_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "ARM64")
    facts = host_facts()
    assert (facts.system, facts.machine) == ("darwin", "arm64")


def test_placement_plans_with_the_adapter_the_chat_role_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.app import placement
    from lilbee.providers.fleet.engines import choice
    from lilbee.providers.fleet.placement_spec import PlacementError

    class _Marked(BundledFakeAdapter):
        name = "llama.cpp"
        formats = frozenset({"gguf"})

    marked = _Marked()
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, marked)  # type: ignore[arg-type]
    seen: list[object] = []

    def _resolve(spec: object, *, engines: dict[WorkerRole, object], **kwargs: object) -> None:
        seen.append(engines[WorkerRole.CHAT])
        raise PlacementError("stop")

    monkeypatch.setattr(placement, "resolve_placement_plan", _resolve)
    with pytest.raises(PlacementError, match="stop"):
        placement.preview_placement()
    assert seen == [marked]


class _StoreEngine(FakeAdapter):
    """A registered adapter whose build lives in the engines folder, as mistral.rs will."""

    name = "mistral.rs"
    formats = frozenset({"gguf"})
    sha = ""

    def pin_record(self) -> Mapping[str, Any]:
        return {**super().pin_record(), "sha256": self.sha}


def _install_build(models_dir: Path, version: str = "1.2.3") -> str:
    wheel = models_dir.parent / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        tool = zipfile.ZipInfo("pkg/bin/server")
        tool.external_attr = (stat.S_IFREG | 0o755) << 16
        archive.writestr(tool, "#!/bin/sh\n")
        archive.writestr("pkg-1.dist-info/METADATA", "License-Expression: MIT\n")
    sha = hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock = PinLock("mistral.rs", version, "cpu", {}, ("MIT",), sha)
    EngineStore(engines_root(models_dir)).install_from_wheel(lock, wheel)
    return sha


@pytest.fixture()
def store_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(cfg, "models_dir", models_dir)
    monkeypatch.setitem(ADAPTERS, EngineName.MISTRAL, _StoreEngine())  # type: ignore[arg-type]
    return models_dir


def _install_pinned(models_dir: Path) -> None:
    """Install a build and pin the registered adapter to its wheel."""
    ADAPTERS[EngineName.MISTRAL].sha = _install_build(models_dir)  # type: ignore[attr-defined]


def test_a_registered_adapter_with_no_build_is_not_installed(store_engine: Path) -> None:
    installed = installed_engines()
    assert EngineName.MISTRAL not in installed
    assert EngineName.LLAMA in installed


def test_a_build_folder_with_no_record_is_not_installed(store_engine: Path) -> None:
    half = engines_root(store_engine) / "mistral.rs" / "1.2.3-cpu"
    half.mkdir(parents=True)
    (half / "server").write_text("half built", encoding="utf-8")
    assert EngineName.MISTRAL not in installed_engines()


def test_a_build_with_a_complete_record_is_installed(store_engine: Path) -> None:
    _install_pinned(store_engine)
    installed = installed_engines()
    assert installed[EngineName.MISTRAL] == frozenset({"gguf"})
    assert EngineName.LLAMA in installed


def test_a_record_of_another_version_does_not_install_the_pinned_build(
    store_engine: Path,
) -> None:
    _install_build(store_engine, version="0.9.0")
    assert EngineName.MISTRAL not in installed_engines()


def test_the_preferred_engine_description_names_every_engine_setting() -> None:
    description = Config.model_fields["preferred_engine"].description
    assert description
    members = list(EngineSetting)
    assert EngineSetting.MISTRAL in members
    assert all(member.value in description for member in members)


def test_every_engine_name_is_a_setting_value() -> None:
    assert {name.value for name in EngineName} == {s.value for s in EngineSetting} - {"auto"}


def test_a_record_of_another_wheel_is_not_installed_and_the_log_names_the_engine(
    store_engine: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _install_build(store_engine)
    ADAPTERS[EngineName.MISTRAL].sha = "0" * 64  # type: ignore[attr-defined]
    with caplog.at_level("WARNING"):
        installed = installed_engines()
    assert EngineName.MISTRAL not in installed
    assert EngineName.LLAMA in installed
    assert "mistral.rs" in caplog.text
    assert "different wheel" in caplog.text


def test_an_unreadable_engines_folder_leaves_llama_chosen(
    store_engine: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refuse(self: EngineStore, lock: PinLock) -> None:
        raise PermissionError("engines folder is locked")

    _install_pinned(store_engine)
    monkeypatch.setattr(EngineStore, "complete_build", refuse)
    with caplog.at_level("WARNING"):
        installed = installed_engines()
    assert EngineName.MISTRAL not in installed
    assert EngineName.LLAMA in installed
    assert "mistral.rs" in caplog.text
    assert "engines folder is locked" in caplog.text


def test_a_build_of_an_engine_with_no_published_pack_is_not_installed(
    store_engine: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _install_build(store_engine)
    ADAPTERS[EngineName.MISTRAL].sha = None  # type: ignore[attr-defined]
    with caplog.at_level("WARNING"):
        installed = installed_engines()
    assert EngineName.MISTRAL not in installed
    assert EngineName.LLAMA in installed
    assert caplog.text == ""


def test_choice_still_picks_llama_when_a_store_read_fails(
    store_engine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: EngineStore, lock: PinLock) -> None:
        raise PermissionError("engines folder is locked")

    monkeypatch.setattr(EngineStore, "complete_build", refuse)
    assert engine_for_format(ModelFormat.GGUF, cfg).name == EngineName.LLAMA


class _BadPinEngine(_StoreEngine):
    """A registered adapter whose pin record is wrong in one stated way."""

    def __init__(self, change: Mapping[str, Any], drop: str | None = None) -> None:
        self._change = change
        self._drop = drop

    def pin_record(self) -> Mapping[str, Any]:
        record = {**super().pin_record(), "sha256": "ab", **self._change}
        if self._drop:
            del record[self._drop]
        return record


@pytest.mark.parametrize(
    ("change", "drop", "fault"),
    [
        ({}, "sha256", "KeyError"),
        ({"backend": "a/b"}, None, "ValueError"),
        ({"backend": None}, None, "TypeError"),
    ],
    ids=["no-sha256", "slash-in-backend", "backend-none"],
)
def test_a_pin_that_cannot_be_parsed_leaves_only_that_engine_not_installed(
    store_engine: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    change: Mapping[str, Any],
    drop: str | None,
    fault: str,
) -> None:
    _install_build(store_engine)
    monkeypatch.setitem(ADAPTERS, EngineName.MISTRAL, _BadPinEngine(change, drop))  # type: ignore[arg-type]
    with caplog.at_level("WARNING"):
        installed = installed_engines()
    assert EngineName.MISTRAL not in installed
    assert EngineName.LLAMA in installed
    assert "mistral.rs counts as not installed" in caplog.text
    assert fault in caplog.text


def test_an_adapter_that_never_states_bundled_is_reported_not_hidden(
    store_engine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Forgot:
        name = "mistral.rs"
        formats = frozenset({"gguf"})

    monkeypatch.setitem(ADAPTERS, EngineName.MISTRAL, _Forgot())  # type: ignore[arg-type]
    with pytest.raises(AttributeError, match="bundled"):
        installed_engines()


GGUF = ModelFormat.GGUF
BOTH = {EngineName.LLAMA: frozenset({GGUF}), EngineName.MISTRAL: frozenset({GGUF})}
LINUX = HostFacts(system="linux", machine="x86_64")


def _rule(role: WorkerRole, preferred: str = "auto", override: str | None = None) -> Any:
    return choose_engine(
        fmt=GGUF,
        host=LINUX,
        installed=BOTH,
        preferred=preferred,
        override=override,
        role=role,
        model="org/m",
    )


def test_a_preferred_engine_that_does_not_serve_rerank_leaves_rerank_on_the_next_engine() -> None:
    rerank = _rule(WorkerRole.RERANK, preferred="mistral.rs")
    assert (rerank.name, rerank.notice_kind, rerank.notice) == (EngineName.LLAMA, None, None)
    for role in (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.VISION):
        assert _rule(role, preferred="mistral.rs").name is EngineName.MISTRAL


def test_a_stored_override_for_an_unserved_role_falls_back_with_a_notice() -> None:
    rerank = _rule(WorkerRole.RERANK, override="mistral.rs")
    assert rerank.name is EngineName.LLAMA
    assert rerank.notice == (
        "Engine mistral.rs for rerank does not serve this role. Using llama.cpp."
    )
    assert _rule(WorkerRole.CHAT, override="mistral.rs").name is EngineName.MISTRAL


class _NoChat(FakeAdapter):
    served_roles = frozenset({WorkerRole.RERANK, WorkerRole.EMBED})


def test_a_stand_in_with_no_chat_is_never_chosen_for_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(ADAPTERS, EngineName.TEI, _NoChat())  # type: ignore[arg-type]
    installed = {**BOTH, EngineName.TEI: frozenset({GGUF})}
    kwargs: dict[str, Any] = dict(fmt=GGUF, host=LINUX, installed=installed, override=None)
    chat = choose_engine(**kwargs, preferred="tei", role=WorkerRole.CHAT)
    embed = choose_engine(**kwargs, preferred="tei", role=WorkerRole.EMBED)
    assert (chat.name, embed.name) == (EngineName.LLAMA, EngineName.TEI)


def test_the_default_skips_an_engine_that_does_not_serve_the_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(ADAPTERS, EngineName.LLAMA, _NoChat())  # type: ignore[arg-type]
    chat = _rule(WorkerRole.CHAT)
    assert chat.name is None
    assert chat.notice == "For chat: No installed engine loads the gguf format of org/m. " + (
        "Run lilbee engine list to see the installed engines."
    )
    assert _rule(WorkerRole.EMBED).name is EngineName.LLAMA


def test_the_launch_plan_puts_rerank_on_the_next_engine_after_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine_choice, "installed_engines", lambda: BOTH)
    monkeypatch.setattr(engine_choice, "host_facts", lambda: LINUX)
    config = Config()
    config.preferred_engine = EngineSetting.MISTRAL
    config.reranker_model = "org/rerank/r.gguf"
    config.engine_choices = '{"org/rerank/r.gguf": "mistral.rs"}'
    assignment = resolve_engines(config)
    rerank = assignment.for_role(WorkerRole.RERANK)
    assert (rerank.adapter, rerank.notice is not None) == (LLAMA, True)
    assert assignment.for_role(WorkerRole.CHAT).name is EngineName.MISTRAL
    config.engine_choices = None
    assert resolve_engines(config).for_role(WorkerRole.RERANK).adapter is LLAMA
