"""The bind contract: models + engine pin decide sharing; derived values do not."""

from lilbee.providers.fleet.contract import contract_matches
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_manager import _SwapState
from lilbee.providers.roles import WorkerRole


def _launch(role: WorkerRole, model: str, *, ctx: int = 0) -> InstanceLaunch:
    return InstanceLaunch(
        role=role, argv=["/bin/llama-server"], env_overrides={}, model=model, ctx=ctx
    )


def _state(*launches: InstanceLaunch, pin: str = "pin-a") -> _SwapState:
    return _SwapState(
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
    state = _SwapState(
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


def test_empty_wanted_binds_a_pin_equal_nonempty_engine() -> None:
    # provider._bind_all_in_dir calls contract_matches(state, (), pin) as "is this a
    # pin-equal, decodable, non-empty engine we could bind?". The vacuous all() over
    # empty wanted must stay gated on the pin, decodability, and non-empty-served
    # checks, so an early-return-True refactor for empty wanted would fail here.
    state = _state(_launch(WorkerRole.CHAT, "m-chat"))
    assert contract_matches(state, (), "pin-a") is True
    assert contract_matches(state, (), "pin-b") is False  # pin still gates
    assert contract_matches(_state(pin="pin-a"), (), "pin-a") is False  # empty served refused
