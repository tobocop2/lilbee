"""The answer an engine gives about whether the host can run it."""

from __future__ import annotations

from dataclasses import dataclass

from lilbee.providers.fleet.host_caps import Feature, HostCaps, NpuVendor


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
PLATFORM_UNREAD = HostGate(
    True, "The OS or CPU architecture of this host could not be read.", known=False
)


def npu_gate(host: HostCaps, vendor: NpuVendor, generation: str | None = None) -> HostGate:
    """Whether *host* has an NPU of *vendor*, and of *generation* when one is named.

    An NPU list that could not be read is unknown, not absent. So is an NPU of the right
    vendor whose generation the host does not say. A Qualcomm accelerator on Linux is
    never a Snapdragon X NPU: the Linux class lists data-centre cards under that vendor.
    """
    if not host.npu_known:
        return HostGate(True, host.npu_reason, known=False)
    mine = [npu for npu in host.npus if npu.vendor is vendor]
    if vendor is NpuVendor.QUALCOMM and mine:
        return HostGate(False, f"{mine[0].name} found, which is not a Snapdragon X NPU.")
    if not mine:
        return HostGate(False, f"This computer has no {vendor.value} NPU.")
    if generation is None or any(npu.generation == generation for npu in mine):
        return PASS
    if any(npu.generation is None for npu in mine):
        return HostGate(
            True, f"The generation of the {mine[0].name} could not be read.", known=False
        )
    return HostGate(False, f"{mine[0].name} is not {generation}.")
