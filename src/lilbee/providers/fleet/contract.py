"""Decides whether a running engine can serve a lilbee's configuration.

The contract is the per-role models plus the engine build pin. Planner-derived
values (ctx, slots) are accepted from the running engine, never recomputed:
they vary legitimately with GPU occupancy at plan time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.providers.fleet.launch import InstanceLaunch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lilbee.providers.fleet.swap_manager import SwapState
    from lilbee.providers.roles import WorkerRole


def decoded_launches(state: SwapState) -> list[InstanceLaunch] | None:
    """The engine's recorded launches, or ``None`` when the contract is undecodable.

    The single decode site. Callers previously re-decoded launches bare, relying on
    a preceding contract_matches call to have proven decodability, so reordering or
    dropping that guard turned a non-match into an unhandled exception in the bind
    ladder. Returning ``None`` makes "undecodable" a value every caller must handle.
    """
    try:
        return [InstanceLaunch.from_state(item) for item in state.launches]
    except (KeyError, TypeError, ValueError):
        return None


def served_pairs(state: SwapState) -> set[tuple[WorkerRole, str]] | None:
    """The (role, model) pairs the engine behind *state* serves, or ``None``."""
    launches = decoded_launches(state)
    return None if launches is None else {(launch.role, launch.model) for launch in launches}


def contract_matches(state: SwapState, wanted: Iterable[tuple[WorkerRole, str]], pin: str) -> bool:
    """Whether the engine behind *state* serves every wanted (role, model) pair.

    The engine may serve more roles than asked; the engine build pin must
    equal *pin*, and an empty or undecodable served contract never matches.
    """
    if state.engine_pin != pin:
        return False
    served = served_pairs(state)
    if not served:
        return False
    return all(pair in served for pair in wanted)
