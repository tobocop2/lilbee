"""The Vulkan adapter enumeration, as a child process that prints JSON.

``vkCreateInstance`` loads every vendor ICD installed on the host. A broken or
conflicting one faults inside the loader, and a fault is a signal, not an
exception: no ``except`` clause in the calling process can catch it. Running the
enumeration in the daemon therefore means any user's bad driver can kill lilbee,
and not only on the Vulkan path, since the unified-memory classification asks
this question for CUDA, ROCm and SYCL devices too.

So it runs here instead, where dying is an answer. The parent reads the JSON on
stdout; a non-zero exit, a signal, or unparseable output all mean the same thing,
which is that the loader has no opinion and the host is treated as it was before
anything asked.

The enumeration itself is unchanged and still lives in
:mod:`lilbee.providers.fleet.gpu_select`; this module is the process boundary
around it.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from lilbee.providers.fleet.gpu_select import VulkanDevice, enumerate_in_process

# Exit code for "the loader could not be reached or failed", as distinct from a
# clean run that found no adapters, which exits 0 with an empty list.
NO_OPINION_EXIT = 2


def _as_json(device: VulkanDevice) -> dict[str, object]:
    """One adapter as JSON-safe fields; the UUID is bytes and travels as hex."""
    return {
        "index": device.index,
        "device_type": device.device_type,
        "device_name": device.device_name,
        "vendor_id": device.vendor_id,
        "vram_bytes": device.vram_bytes,
        "device_uuid": device.device_uuid.hex(),
        "storage_buffer_16bit": device.storage_buffer_16bit,
        "free_bytes": device.free_bytes,
    }


def from_json(payload: list[dict[str, Any]]) -> list[VulkanDevice]:
    """Rebuild the adapters the child printed.

    Raises on a payload that is not the shape :func:`_as_json` writes, which the
    parent reads as the loader having no opinion.
    """
    return [
        VulkanDevice(
            index=int(entry["index"]),
            device_type=int(entry["device_type"]),
            device_name=str(entry["device_name"]),
            vendor_id=int(entry["vendor_id"]),
            vram_bytes=int(entry["vram_bytes"]),
            device_uuid=bytes.fromhex(str(entry["device_uuid"])),
            storage_buffer_16bit=entry["storage_buffer_16bit"],
            free_bytes=entry["free_bytes"],
        )
        for entry in payload
    ]


def main() -> None:
    """Print the enumeration as JSON, or exit non-zero when there is no answer."""
    devices = enumerate_in_process()
    if devices is None:
        sys.exit(NO_OPINION_EXIT)
    json.dump([_as_json(device) for device in devices], sys.stdout)


if __name__ == "__main__":  # pragma: no cover - process entry glue
    main()
