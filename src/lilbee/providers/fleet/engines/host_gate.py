"""The answer an engine gives about whether the host can run it."""

from __future__ import annotations

from dataclasses import dataclass

from lilbee.providers.fleet.host_caps import Feature


@dataclass(frozen=True)
class HostGate:
    """Whether the host can run an engine; ``known`` is false when a field could not be read.

    An unknown field neither passes nor refuses: ``ok`` stays true so an explicit
    choice and a catalog row stand, and ``auto_selectable`` is false so the engine
    is never picked for the user. ``reason`` says why in both cases.
    """

    ok: bool
    reason: str = ""
    known: bool = True

    @property
    def auto_selectable(self) -> bool:
        """True only when every field the gate reads is known and present."""
        return self.ok and self.known


def gate_on(*features: Feature) -> HostGate:
    """A refusal if a known feature is absent, unknown if one is unreadable, else a pass."""
    for feature in features:
        if feature.known and not feature.present:
            return HostGate(False, feature.reason)
    for feature in features:
        if not feature.known:
            return HostGate(True, feature.reason, known=False)
    return HostGate(True)


PASS = HostGate(True)
