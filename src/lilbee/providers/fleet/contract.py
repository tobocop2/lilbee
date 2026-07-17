"""Decides whether a running engine can serve a lilbee's configuration.

Sharing keys on what actually matters for correctness: the per-role models and
the engine build pin. Planner-derived values (ctx, slots) are accepted from the
running engine's contract, never recomputed for comparison, because they vary
legitimately with GPU occupancy at plan time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.providers.fleet.launch import InstanceLaunch

if TYPE_CHECKING:
    from lilbee.providers.fleet.swap_manager import _SwapState


def contract_matches(state: _SwapState, wanted: list[InstanceLaunch], pin: str) -> bool:
    """Whether the engine behind *state* serves every launch in *wanted*.

    The engine may serve more roles than asked; every wanted (role, model)
    pair must be present, and the engine build pin must equal *pin*.
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
    return all((launch.role, launch.model) in served for launch in wanted)
