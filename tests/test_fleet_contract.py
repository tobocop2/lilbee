"""The bind contract: models + engine pin decide sharing; derived values do not."""

from lilbee.providers.fleet.contract import chat_ctx_covers, contract_matches
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_manager import SwapState
from lilbee.providers.roles import WorkerRole


def _launch(
    role: WorkerRole, model: str, *, ctx: int = 0, built_ctx_target: int = 0
) -> InstanceLaunch:
    return InstanceLaunch(
        role=role,
        argv=["/bin/llama-server"],
        env_overrides={},
        model=model,
        ctx=ctx,
        built_ctx_target=built_ctx_target,
    )


def _state(*launches: InstanceLaunch, pin: str = "pin-a") -> SwapState:
    return SwapState(
        pid=1,
        pgid=None,
        proxy_port=4100,
        launches=tuple(launch.to_state() for launch in launches),
        engine_pin=pin,
    )


def test_same_models_and_pin_match() -> None:
    state = _state(_launch(WorkerRole.CHAT, "m-chat"), _launch(WorkerRole.EMBED, "m-embed"))
    wanted = [(WorkerRole.CHAT, "m-chat"), (WorkerRole.EMBED, "m-embed")]
    assert contract_matches(state, wanted, "pin-a") is True


def test_pin_mismatch_refuses() -> None:
    state = _state(_launch(WorkerRole.CHAT, "m-chat"), pin="pin-b")
    assert contract_matches(state, [(WorkerRole.CHAT, "m-chat")], "pin-a") is False


def test_model_mismatch_refuses() -> None:
    state = _state(_launch(WorkerRole.CHAT, "m-other"))
    assert contract_matches(state, [(WorkerRole.CHAT, "m-chat")], "pin-a") is False


def test_wanted_role_missing_from_engine_refuses() -> None:
    state = _state(_launch(WorkerRole.CHAT, "m-chat"))
    wanted = [(WorkerRole.CHAT, "m-chat"), (WorkerRole.EMBED, "m-embed")]
    assert contract_matches(state, wanted, "pin-a") is False


def test_engine_serving_extra_roles_still_matches() -> None:
    state = _state(_launch(WorkerRole.CHAT, "m-chat"), _launch(WorkerRole.EMBED, "m-embed"))
    assert contract_matches(state, [(WorkerRole.CHAT, "m-chat")], "pin-a") is True


def test_derived_ctx_difference_does_not_refuse() -> None:
    state = _state(_launch(WorkerRole.CHAT, "m-chat", ctx=8192))
    assert contract_matches(state, [(WorkerRole.CHAT, "m-chat")], "pin-a") is True


def test_undecodable_contract_refuses() -> None:
    state = SwapState(
        pid=1,
        pgid=None,
        proxy_port=4100,
        launches=({"junk": True},),
        engine_pin="pin-a",
    )
    assert contract_matches(state, [(WorkerRole.CHAT, "m-chat")], "pin-a") is False


def test_empty_engine_contract_refuses() -> None:
    state = _state(pin="pin-a")
    assert contract_matches(state, [(WorkerRole.CHAT, "m-chat")], "pin-a") is False


def test_chat_ctx_covers_adopts_a_larger_live_window() -> None:
    launches = [_launch(WorkerRole.CHAT, "m-chat", ctx=12288)]
    assert chat_ctx_covers(launches, 8192) is True


def test_chat_ctx_covers_accepts_an_equal_window() -> None:
    launches = [_launch(WorkerRole.CHAT, "m-chat", ctx=12288)]
    assert chat_ctx_covers(launches, 12288) is True


def test_chat_ctx_covers_refuses_a_window_smaller_than_the_demand() -> None:
    launches = [_launch(WorkerRole.CHAT, "m-chat", ctx=8192)]
    assert chat_ctx_covers(launches, 12288) is False


def test_chat_ctx_covers_accepts_with_no_demand() -> None:
    launches = [_launch(WorkerRole.CHAT, "m-chat", ctx=8192)]
    assert chat_ctx_covers(launches, 0) is True


def test_chat_ctx_covers_adopts_when_builder_aimed_at_least_as_high() -> None:
    """A window below the demand still covers when the builder's target reached
    it: the same planner already aimed that high and achieved this window, so
    a replace would rebuild the identical window in a loop (model ceiling or
    tight cards)."""
    launches = [_launch(WorkerRole.CHAT, "m-chat", ctx=16384, built_ctx_target=24576)]
    assert chat_ctx_covers(launches, 24576) is True


def test_chat_ctx_covers_refuses_when_builder_aimed_lower() -> None:
    """A builder that aimed lower than this demand may have left window on the
    table (the multi-GPU deflation case); refusing rebuilds at the higher
    target exactly once, and the new record then covers."""
    launches = [_launch(WorkerRole.CHAT, "m-chat", ctx=16384, built_ctx_target=16384)]
    assert chat_ctx_covers(launches, 65536) is False


def test_chat_ctx_covers_refuses_pre_field_records_below_demand() -> None:
    """Old records without built_ctx_target keep the plain window rule: one
    rebuild after upgrade records the target and converges."""
    launches = [_launch(WorkerRole.CHAT, "m-chat", ctx=16384)]
    assert chat_ctx_covers(launches, 24576) is False


def test_chat_ctx_covers_accepts_an_unrecorded_live_window() -> None:
    # A record from before ctx was persisted carries 0; adopt, never restart.
    launches = [_launch(WorkerRole.CHAT, "m-chat", ctx=0)]
    assert chat_ctx_covers(launches, 12288) is True


def test_chat_ctx_covers_ignores_non_chat_launches() -> None:
    launches = [_launch(WorkerRole.EMBED, "m-embed", ctx=512)]
    assert chat_ctx_covers(launches, 12288) is True


def test_empty_wanted_binds_a_pin_equal_nonempty_engine() -> None:
    # provider._bind_all_in_dir calls contract_matches(state, (), pin) as "is this a
    # pin-equal, decodable, non-empty engine we could bind?". The vacuous all() over
    # empty wanted must stay gated on the pin, decodability, and non-empty-served
    # checks, so an early-return-True refactor for empty wanted would fail here.
    state = _state(_launch(WorkerRole.CHAT, "m-chat"))
    assert contract_matches(state, (), "pin-a") is True
    assert contract_matches(state, (), "pin-b") is False  # pin still gates
    assert contract_matches(_state(pin="pin-a"), (), "pin-a") is False  # empty served refused
