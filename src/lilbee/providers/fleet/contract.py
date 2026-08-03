"""Decides whether a running engine can serve a lilbee's configuration.

The contract is the per-role models plus the engine build pin. Planner-derived
values (ctx, slots) are accepted from the running engine, never recomputed:
they vary legitimately with GPU occupancy at plan time. The one exception is
``chat_ctx_covers``: a live chat window smaller than what this process needs
cannot be adopted, since no prompt fit can grow it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.roles import WorkerRole

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lilbee.providers.fleet.swap_manager import SwapState


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


def chat_ctx_covers(launches: Iterable[InstanceLaunch], demanded_ctx: int) -> bool:
    """Whether the engine's per-slot chat window serves *demanded_ctx* tokens.

    ``launches`` are the running engine's recorded launches. A zero demand, no
    chat launch, or a record without a positive chat ctx never refuses: derived
    values are adopted from the running engine. A window below the demand still
    covers when the record's ``built_ctx_target`` reaches the demand: the same
    planner aimed at least as high and achieved this window, so replacing the
    engine would rebuild the same window in a loop. A refusal sends the ladder
    to its replace-or-overflow decision.
    """
    chat = [launch for launch in launches if launch.role is WorkerRole.CHAT and launch.ctx > 0]
    live = min((launch.ctx for launch in chat), default=0)
    if demanded_ctx <= 0 or live <= 0 or demanded_ctx <= live:
        return True
    built_target = min((launch.built_ctx_target for launch in chat), default=0)
    return built_target > 0 and demanded_ctx <= built_target


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
