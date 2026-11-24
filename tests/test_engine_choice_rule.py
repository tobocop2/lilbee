"""The pure engine rule: every cell of host x format x installed x preferred x override."""

from __future__ import annotations

import itertools
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lilbee.app.engine_choice import _choice_for
from lilbee.catalog.runnable import _read_host
from lilbee.catalog.types import ModelFormat
from lilbee.providers.base import ProviderErrorKind
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import (
    ADAPTERS,
    LIST_HINT,
    EngineChoice,
    NoEngineError,
    NoticeKind,
    choose_engine,
)
from lilbee.providers.fleet.engines.host_gate import HostGate
from lilbee.providers.fleet.host_caps import HostArch, HostCaps, HostOS, collect
from lilbee.providers.roles import WorkerRole

L, OV, M, IK = EngineName.LLAMA, EngineName.OPENVINO, EngineName.MLX, EngineName.IK_LLAMA
G, MX, OVF, ST = (
    ModelFormat.GGUF,
    ModelFormat.MLX,
    ModelFormat.OPENVINO,
    ModelFormat.SAFETENSORS,
)


def _caps(system: str, machine: str, flags: list[str] | None = None, nvidia=None) -> HostCaps:
    cpu_info = {} if flags is None else {"flags": flags, "brand_raw": "Xeon"}
    return collect(system=system, machine=machine, cpu_info=cpu_info, nvidia=nvidia)


MAC = _caps("Darwin", "arm64")
NVIDIA = _caps("Linux", "x86_64", ["avx2"])
INTEL_MAC = _caps("Darwin", "x86_64", ["avx2"])
HOSTS = (MAC, NVIDIA, INTEL_MAC)
FORMATS = (G, MX, OVF, ST)
# Formats each engine loads, as an adapter reports them; a safetensors engine is invented
# so the cell where an installed engine does load a fourth format is crossed too.
LOADS = {L: {G}, OV: {G, OVF}, M: {MX}, IK: {G}}
INSTALLED = (
    {L: LOADS[L]},
    {L: LOADS[L], OV: LOADS[OV]},
    {L: LOADS[L], M: LOADS[M]},
    {L: LOADS[L], IK: LOADS[IK]},
    {OV: LOADS[OV]},
    {},
)
PREFERRED = ("auto", L, IK, M)
OVERRIDE = (None, IK, M, OV)


def _default(host, fmt, installed):
    order = []
    if host.os is HostOS.MACOS and host.arch is HostArch.ARM64:
        order.append(M)
    order.append(L)
    return next((e for e in order if e in installed and fmt in installed[e]), None)


@pytest.fixture(autouse=True)
def _mlx_without_its_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rule is crossed with every host, so the real gates of mlx and openvino stay out of it.

    The gates have their own rows in test_mlx_adapter and test_engine_openvino.
    """
    for name in (M, OV):
        monkeypatch.setattr(ADAPTERS[name], "host_gate", lambda _host: HostGate(True))


def _oracle(host, fmt, installed, preferred, override):
    requested = override or (None if preferred == "auto" else preferred)
    kind = "override-unusable" if override else "preferred-unusable"
    if requested is None:
        default = _default(host, fmt, installed)
        return default, (None if default else "none")
    if requested in installed:
        if fmt in installed[requested]:
            return requested, None
        if override:
            return None, kind
    default = _default(host, fmt, installed)
    return default, kind


CASES = list(itertools.product(HOSTS, FORMATS, INSTALLED, PREFERRED, OVERRIDE))


def test_cases_are_not_empty() -> None:
    assert len(CASES) == 3 * 4 * 6 * 4 * 4


@pytest.mark.parametrize(("host", "fmt", "installed", "preferred", "override"), CASES)
def test_choice_equals_the_table(host, fmt, installed, preferred, override) -> None:
    got = choose_engine(
        fmt=fmt, host=lambda: host, installed=installed, preferred=preferred, override=override
    )
    assert (got.name, got.notice_kind) == _oracle(host, fmt, installed, preferred, override)
    if got.name is not None:
        assert fmt in installed[got.name]


def _ask(**kw) -> EngineChoice:
    base = dict(
        fmt=G, host=lambda: NVIDIA, installed={L: LOADS[L]}, preferred="auto", override=None
    )
    return choose_engine(**{**base, **kw})


def test_real_adapter_formats_feed_the_rule() -> None:
    assert _ask(installed={L: LLAMA.formats}).name is L
    assert _ask(installed={L: LLAMA.formats}, fmt=ST).name is None


def test_override_wins_over_preference() -> None:
    both = {L: LOADS[L], IK: LOADS[IK]}
    assert _ask(installed=both, preferred=L, override=IK).name is IK


def test_preferred_applies_when_override_absent() -> None:
    assert _ask(installed={L: LOADS[L], IK: LOADS[IK]}, preferred=IK).name is IK


def test_auto_never_picks_the_opt_in_engine() -> None:
    assert _ask(installed={IK: LOADS[IK], L: LOADS[L]}).name is L
    assert _ask(installed={IK: LOADS[IK]}).name is None


def test_not_installed_override_falls_back_with_a_notice() -> None:
    got = _ask(override=M)
    assert got.notice_kind is NoticeKind.OVERRIDE_UNUSABLE
    assert got.notice == f"Engine mlx is not installed. Using llama.cpp. {LIST_HINT}"


def test_override_that_cannot_load_the_format_gives_no_engine() -> None:
    got = _ask(installed={L: LOADS[L], M: LOADS[M]}, override=M)
    assert got.name is None
    assert got.notice_kind is NoticeKind.OVERRIDE_UNUSABLE
    assert got.notice == "Engine mlx cannot load the gguf format. Choose an engine that loads gguf."


def test_preference_that_cannot_load_the_format_falls_back_with_a_notice() -> None:
    got = _ask(host=lambda: MAC, installed={L: LOADS[L], M: LOADS[M]}, preferred=M)
    assert got.name is L
    assert got.notice_kind is NoticeKind.PREFERRED_UNUSABLE
    assert got.notice == "Engine mlx cannot load the gguf format. Using llama.cpp."


def test_preference_that_cannot_load_with_no_default_gives_no_engine() -> None:
    got = _ask(fmt=ST, installed={L: LOADS[L], M: LOADS[M]}, preferred=M)
    assert got.name is None
    assert got.notice == (
        "Engine mlx cannot load the safetensors format. "
        "No installed engine loads the safetensors format."
    )


def test_unknown_engine_name_is_not_installed() -> None:
    got = _ask(preferred="nonesuch")
    assert got.name is L
    assert got.notice_kind is NoticeKind.PREFERRED_UNUSABLE
    assert "Engine nonesuch is not installed." in got.notice


def test_not_installed_with_no_fallback_names_the_format() -> None:
    got = _ask(fmt=ST, override=M)
    assert got.name is None
    assert got.notice == (
        "Engine mlx is not installed. No installed engine loads the safetensors format."
    )


def test_auto_with_no_engine_returns_text_and_require_raises() -> None:
    got = _ask(fmt=ST)
    assert got.name is None
    assert got.notice_kind is NoticeKind.NONE
    assert got.notice == (f"No installed engine loads the safetensors format. {LIST_HINT}")
    with pytest.raises(NoEngineError, match="No installed engine loads the safetensors") as caught:
        got.require()
    assert caught.value.kind is ProviderErrorKind.BAD_REQUEST


def test_auto_names_the_installed_engine_that_loads_a_format_the_default_skips() -> None:
    got = _ask(fmt=ST, installed={L: LOADS[L], M: {ST}})
    assert got.name is None
    assert got.notice == (
        "No engine is chosen by default for the safetensors format. "
        "Choose mlx with lilbee engine prefer. "
        "It applies to every model without its own choice."
    )


def test_a_choice_for_no_role_names_prefer_and_never_use() -> None:
    both = {IK: {ST}, M: {ST}}
    got = _ask(fmt=ST, installed=both, model="org/m#safetensors")
    assert got.name is None
    assert got.notice is not None
    assert f"Choose {IK} or {M} with lilbee engine prefer." in got.notice
    assert "engine use" not in got.notice


def test_a_choice_for_a_role_names_use_and_prefer() -> None:
    got = _ask(fmt=ST, installed={M: {ST}}, role=WorkerRole.EMBED, model="org/m#safetensors")
    assert got.notice == (
        "For embed: No engine is chosen by default "
        "for the safetensors format of org/m#safetensors. "
        "Choose mlx with lilbee engine use for this model, "
        "or with lilbee engine prefer for every model without its own choice."
    )


def test_a_preference_that_skips_the_role_names_the_engine_that_serves_it() -> None:
    adapter = {L: LOADS[L], M: {ST}}
    got = _ask(fmt=ST, installed=adapter, preferred=L)
    assert got.name is None
    assert got.notice == (
        "Engine llama.cpp cannot load the safetensors format. "
        "No engine is chosen by default. "
        "Choose mlx with lilbee engine prefer. "
        "It applies to every model without its own choice."
    )


def test_a_preference_that_skips_a_role_names_use_and_prefer() -> None:
    got = _ask(
        fmt=ST,
        installed={L: LOADS[L], M: {ST}},
        preferred=L,
        role=WorkerRole.EMBED,
        model="org/m#safetensors",
    )
    assert got.notice is not None
    assert got.notice.endswith(
        "No engine is chosen by default. "
        "Choose mlx with lilbee engine use for this model, "
        "or with lilbee engine prefer for every model without its own choice."
    )


def test_require_returns_the_name() -> None:
    assert _ask().require() is L


@pytest.mark.parametrize("blank", ["", "  "])
def test_blank_engine_name_is_refused(blank: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _ask(override=blank)
    with pytest.raises(ValueError, match="must not be empty"):
        _ask(preferred=blank)


def test_name_that_is_a_substring_of_an_installed_engine_does_not_match() -> None:
    got = _ask(preferred="llama")
    assert got.notice_kind is NoticeKind.PREFERRED_UNUSABLE
    assert got.name is L
    assert "Engine llama is not installed." in got.notice


NO_AVX2 = _caps("Linux", "x86_64", ["sse4_2"])
UNREADABLE = collect(
    system="", machine="", cpu_info={}, nvidia={"driver": None, "cuda_driver": None, "gpus": None}
)
LLAMA_AND_IK = {L: {G}, IK: {G}}


def _no_read() -> HostCaps:
    raise AssertionError("the host was read")


def _config(override: str | None = None, preferred: str = "auto") -> SimpleNamespace:
    choices = json.dumps({"org/m": override}) if override else None
    return SimpleNamespace(engine_choices=choices, preferred_engine=preferred)


def _choose_at_run(config: SimpleNamespace, detected: HostCaps | Exception) -> EngineChoice:
    """The run's choice for org/m, with the host read through the session cache."""
    side_effect = detected if isinstance(detected, Exception) else None
    _read_host.cache_clear()
    try:
        with (
            patch("lilbee.catalog.runnable.detect", return_value=detected, side_effect=side_effect),
            patch("lilbee.app.engine_choice.model_format", return_value=G),
        ):
            return _choice_for(config, "org/m", LLAMA_AND_IK, WorkerRole.CHAT)
    finally:
        _read_host.cache_clear()


def test_engine_the_host_refuses_falls_back_with_the_reason() -> None:
    got = _choose_at_run(_config(override=IK), NO_AVX2)
    assert got.name is L
    assert got.notice_kind is NoticeKind.OVERRIDE_UNUSABLE
    assert got.notice == (
        "Engine ik_llama.cpp cannot run on this computer. Xeon lacks AVX2. "
        "ik_llama.cpp needs a CPU with AVX2 or NEON. Using llama.cpp."
    )


def test_a_refused_preference_falls_back_with_the_reason() -> None:
    got = _choose_at_run(_config(preferred=IK), NO_AVX2)
    assert got.name is L
    assert got.notice_kind is NoticeKind.PREFERRED_UNUSABLE
    assert "AVX2" in (got.notice or "")


def test_a_refused_engine_with_no_default_gives_no_engine_and_the_reason() -> None:
    got = _ask(host=lambda: NO_AVX2, installed={IK: {G}}, override=IK)
    assert got.name is None
    assert "AVX2" in (got.notice or "")
    assert "No installed engine that loads this model can run on this computer." in (got.notice or "")


def test_a_refused_engine_is_not_offered_in_the_hint() -> None:
    got = _ask(host=lambda: NO_AVX2, installed={IK: {G}})
    assert got.name is None
    assert "Choose" not in (got.notice or "")


_IK_REFUSED = (
    "Engine ik_llama.cpp loads the gguf format but cannot run on this computer. "
    "Xeon lacks AVX2. ik_llama.cpp needs a CPU with AVX2 or NEON."
)


def test_auto_names_the_installed_engine_that_the_host_refuses_and_its_reason() -> None:
    got = _ask(host=lambda: NO_AVX2, installed={IK: {G}})
    assert got.name is None
    assert got.notice_kind is NoticeKind.NONE
    assert got.notice == _IK_REFUSED
    assert "No installed engine loads" not in got.notice
    assert LIST_HINT not in got.notice


def test_auto_with_a_role_keeps_the_role_prefix_on_the_host_refusal() -> None:
    got = _ask(host=lambda: NO_AVX2, installed={IK: {G}}, role=WorkerRole.CHAT, model="org/m")
    assert got.name is None
    assert got.notice == (
        "For chat: Engine ik_llama.cpp loads the gguf format of org/m but cannot run on this computer. "
        "Xeon lacks AVX2. ik_llama.cpp needs a CPU with AVX2 or NEON."
    )


def _host_refuses(monkeypatch: pytest.MonkeyPatch, name: EngineName, reason: str) -> None:
    monkeypatch.setattr(ADAPTERS[name], "host_gate", lambda _host: HostGate(False, reason))


def test_auto_names_every_refused_engine_with_one_space_between_the_sentences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _host_refuses(monkeypatch, OV, "No Intel GPU here.")
    got = _ask(host=lambda: NO_AVX2, installed={IK: {G}, OV: {G}})
    assert got.name is None
    assert got.notice == (
        f"{_IK_REFUSED} "
        "Engine openvino loads the gguf format but cannot run on this computer. "
        "No Intel GPU here."
    )


def test_auto_leaves_out_a_refused_engine_that_does_not_serve_the_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # mlx serves embed and the host refuses it; openvino serves chat only, so its refusal
    # says nothing about the embed role.
    _host_refuses(monkeypatch, M, "No Apple silicon here.")
    _host_refuses(monkeypatch, OV, "No Intel GPU here.")
    got = _ask(
        fmt=MX,
        host=lambda: NVIDIA,
        installed={M: {MX}, OV: {MX}},
        role=WorkerRole.EMBED,
    )
    assert got.name is None
    assert got.notice == (
        "For embed: Engine mlx loads the mlx format but cannot run on this computer. "
        "No Apple silicon here."
    )
    assert "openvino" not in got.notice


@pytest.mark.parametrize(
    "request_kw", [{"override": IK}, {"preferred": IK}], ids=["override", "pref"]
)
def test_an_unreadable_host_leaves_the_request_standing(request_kw: dict[str, str]) -> None:
    got = _ask(host=lambda: UNREADABLE, installed=LLAMA_AND_IK, **request_kw)
    assert (got.name, got.notice) == (IK, None)


def test_a_host_read_that_fails_keeps_every_engine() -> None:
    assert _choose_at_run(_config(), OSError("no platform")).name is L
    assert _choose_at_run(_config(override=IK), OSError("no platform")).name is IK


@pytest.mark.parametrize(
    "request_kw", [{}, {"override": L}, {"preferred": L}], ids=["auto", "override", "pref"]
)
def test_the_bundled_engine_reads_nothing_about_the_host(request_kw: dict[str, str]) -> None:
    assert _ask(host=_no_read, **request_kw).name is L


def test_an_engine_that_is_not_installed_reads_nothing_about_the_host() -> None:
    assert _ask(host=_no_read, override=IK).name is L


def test_the_host_is_read_once_for_many_choices() -> None:
    _read_host.cache_clear()
    try:
        with (
            patch("lilbee.catalog.runnable.detect", return_value=NO_AVX2) as detect,
            patch("lilbee.app.engine_choice.model_format", return_value=G),
        ):
            for _ in range(3):
                _choice_for(_config(override=IK), "org/m", LLAMA_AND_IK, WorkerRole.CHAT)
        assert detect.call_count == 1
    finally:
        _read_host.cache_clear()


def test_a_process_on_the_bundled_engine_never_reads_the_host() -> None:
    _read_host.cache_clear()
    with (
        patch("lilbee.catalog.runnable.detect") as detect,
        patch("lilbee.app.engine_choice.model_format", return_value=G),
    ):
        _choice_for(_config(), "org/m", LLAMA_AND_IK, WorkerRole.CHAT)
    assert detect.call_count == 0


def test_the_default_skips_an_engine_with_an_unknown_host_answer() -> None:
    unknown = SimpleNamespace(
        bundled=False,
        served_roles=frozenset(WorkerRole),
        host_gate=lambda _host: HostGate(True, "unreadable", known=False),
    )
    with patch.dict("lilbee.providers.fleet.engines.choice.ADAPTERS", {M: unknown}):
        got = _ask(host=lambda: MAC, installed={L: {G}, M: {G}})
    assert got.name is L


def test_the_default_keeps_mlx_only_on_apple_silicon() -> None:
    assert _ask(host=lambda: MAC, installed={L: {G}, M: {G}}).name is M
    assert _ask(host=lambda: INTEL_MAC, installed={L: {G}, M: {G}}).name is L
