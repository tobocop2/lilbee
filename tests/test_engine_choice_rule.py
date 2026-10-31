"""The pure engine rule: every cell of host x format x installed x preferred x override."""

from __future__ import annotations

import itertools

import pytest

from lilbee.catalog.types import ModelFormat
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import (
    Accel,
    EngineChoice,
    HostFacts,
    NoEngineError,
    NoticeKind,
    choose_engine,
)

L, OV, M, IK = EngineName.LLAMA, EngineName.OPENVINO, EngineName.MLX, EngineName.IK_LLAMA
G, MX, OVF, ST = (
    ModelFormat.GGUF,
    ModelFormat.MLX,
    ModelFormat.OPENVINO,
    ModelFormat.SAFETENSORS,
)
MAC = HostFacts("darwin", "arm64", ())
NVIDIA = HostFacts("linux", "x86_64", ())
NPU = HostFacts("linux", "x86_64", (Accel.INTEL_NPU,))
MAC_NPU = HostFacts("darwin", "arm64", (Accel.INTEL_NPU,))
INTEL_MAC = HostFacts("darwin", "x86_64", ())
HOSTS = (MAC, NVIDIA, NPU, MAC_NPU, INTEL_MAC)
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
    if Accel.INTEL_NPU in host.accel:
        order.append(OV)
    if host.system == "darwin" and host.machine == "arm64":
        order.append(M)
    order.append(L)
    return next((e for e in order if e in installed and fmt in installed[e]), None)


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
    assert len(CASES) == 5 * 4 * 6 * 4 * 4


@pytest.mark.parametrize(("host", "fmt", "installed", "preferred", "override"), CASES)
def test_choice_equals_the_table(host, fmt, installed, preferred, override) -> None:
    got = choose_engine(
        fmt=fmt, host=host, installed=installed, preferred=preferred, override=override
    )
    assert (got.name, got.notice_kind) == _oracle(host, fmt, installed, preferred, override)
    if got.name is not None:
        assert fmt in installed[got.name]


def _ask(**kw) -> EngineChoice:
    base = dict(fmt=G, host=NVIDIA, installed={L: LOADS[L]}, preferred="auto", override=None)
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
    assert got.notice == "Engine mlx is not installed. Install mlx to use it. Using llama.cpp."


def test_override_that_cannot_load_the_format_gives_no_engine() -> None:
    got = _ask(installed={L: LOADS[L], M: LOADS[M]}, override=M)
    assert got.name is None
    assert got.notice_kind is NoticeKind.OVERRIDE_UNUSABLE
    assert got.notice == "Engine mlx cannot load the gguf format. Choose an engine that loads gguf."


def test_preference_that_cannot_load_the_format_falls_back_with_a_notice() -> None:
    got = _ask(host=MAC, installed={L: LOADS[L], M: LOADS[M]}, preferred=M)
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
        "Engine mlx is not installed. Install mlx to use it. "
        "No installed engine loads the safetensors format."
    )


def test_auto_with_no_engine_returns_text_and_require_raises() -> None:
    got = _ask(fmt=ST)
    assert got.name is None
    assert got.notice_kind is NoticeKind.NONE
    assert got.notice == (
        "No installed engine loads the safetensors format. "
        "Install an engine that loads safetensors."
    )
    with pytest.raises(NoEngineError, match="No installed engine loads the safetensors"):
        got.require()


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
