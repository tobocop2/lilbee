"""Config keys whose change makes the fleet owner rebuild."""

from __future__ import annotations

from lilbee.core.config.keys import LOAD_AFFECTING_KEYS
from lilbee.providers.roles import ROLE_GATE_FIELD_TO_ROLE

FLEET_RELOAD_KEYS: frozenset[str] = LOAD_AFFECTING_KEYS | frozenset(ROLE_GATE_FIELD_TO_ROLE)
"""Settings whose change on a config reload makes the fleet owner rebuild.

A peer's plan view holds exactly these keys, so it waits only for a rebuild that comes.
This module sits above ``roles`` because ``roles`` must not import the config package,
which imports ``roles`` back through the model-reference chain.
"""
