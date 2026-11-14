"""Every registered engine adapter defines every member of the adapter protocol."""

from __future__ import annotations

import pytest

from lilbee.providers.fleet.engines import EngineAdapter
from lilbee.providers.fleet.engines.choice import ADAPTERS

PROTOCOL_MEMBERS = sorted(
    name
    for name in {*vars(EngineAdapter), *EngineAdapter.__annotations__}
    if not name.startswith("_")
)


def test_the_protocol_member_list_is_not_empty() -> None:
    assert {"target_env", "runtime_env", "launch_argv", "name", "bundled"} <= set(PROTOCOL_MEMBERS)


def test_the_adapter_registry_is_not_empty() -> None:
    assert len(ADAPTERS) > 0


@pytest.mark.parametrize("adapter", ADAPTERS.values(), ids=lambda adapter: adapter.name)
def test_an_adapter_defines_every_protocol_member(adapter: EngineAdapter) -> None:
    missing = [name for name in PROTOCOL_MEMBERS if not hasattr(adapter, name)]
    assert missing == []
