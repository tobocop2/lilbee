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

    from lilbee.providers.fleet.swap_manager import _SwapState
    from lilbee.providers.roles import WorkerRole


def contract_matches(state: _SwapState, wanted: Iterable[tuple[WorkerRole, str]], pin: str) -> bool:
    """Whether the engine behind *state* serves every wanted (role, model) pair.

    The engine may serve more roles than asked; the engine build pin must
    equal *pin*, and an empty or undecodable served contract never matches.
    """
    if state.engine_pin != pin:
        return False
    try:
        served = {
            (launch.role, launch.model)
            for launch in (InstanceLaunch.from_state(item) for item in state.launches)
        }
    except (KeyError, TypeError, ValueError):
        return False
    if not served:
        return False
    return all(pair in served for pair in wanted)
