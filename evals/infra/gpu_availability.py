"""Live per-datacentre GPU stock on RunPod.

A RunPod network volume is pinned to a datacentre, so the datacentre has to be
chosen before the volume exists. ``sky show-gpus`` cannot answer this: it is the
static price catalogue, and SkyPilot only learns real availability by trying to
launch.

The GraphQL endpoint sits behind Cloudflare and returns 403 (error 1010) to a
default User-Agent, which reads exactly like a bad key or a rate limit. Sending
a browser User-Agent is the whole fix.
"""

from __future__ import annotations

import json
import pathlib
import tomllib
import urllib.request

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
QUERY = '{"query":"query{ dataCenters{ id gpuAvailability{ id available stockStatus } } }"}'
# Low is included deliberately. RunPod multi-GPU flaps under contention even in
# High-stock datacentres, and the retry loop costs nothing because billing
# starts only when a box actually provisions. Excluding Low would report "no
# capacity" while capacity exists intermittently.
USABLE_STOCK = ("High", "Medium", "Low")
# Not every datacentre offers network volumes, and the ones that do are a
# different set from the ones with GPUs. Picking on GPU stock alone gets a
# region where the volume cannot be created at all, which the API only reveals
# at apply time. This list is what the API returns when the placement is
# rejected; RunPod does not expose it as a query.
VOLUME_CAPABLE = frozenset(
    {
        "AP-IN-2",
        "AP-JP-1",
        "CA-MTL-3",
        "CA-MTL-4",
        "EU-CZ-1",
        "EU-FR-1",
        "EU-NL-1",
        "EU-RO-1",
        "EUR-IS-1",
        "EUR-IS-3",
        "EUR-NO-1",
        "EUR-NO-2",
        "US-CA-2",
        "US-IL-1",
        "US-MO-2",
        "US-NC-1",
        "US-NE-1",
        "US-TX-3",
        "US-WA-1",
    }
)
# A100 is sm_80, which the prebuilt cu124 engine has kernels for. H100 is sm_90
# and it does not, so an H100 box needs the preflight to clear before use.
WANTED = ("A100", "H100")


def api_key() -> str:
    """The key from the RunPod config, under `apikey` rather than `api_key`."""
    config = tomllib.loads(pathlib.Path.home().joinpath(".runpod/config.toml").read_text())
    key = config.get("apikey") or config.get("default", {}).get("apikey")
    if not key:
        raise SystemExit("no RunPod apikey found in ~/.runpod/config.toml")
    return key


def availability() -> list[tuple[str, str, str]]:
    """(datacentre, gpu, stock) for every wanted GPU at usable stock."""
    request = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={api_key()}",
        data=QUERY.encode(),
        headers={"User-Agent": BROWSER_UA, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        centres = json.load(response)["data"]["dataCenters"]
    rows = [
        (centre["id"], gpu["id"], gpu["stockStatus"])
        for centre in centres
        for gpu in centre.get("gpuAvailability") or []
        if gpu.get("available")
        and gpu.get("stockStatus") in USABLE_STOCK
        and any(want in (gpu.get("id") or "") for want in WANTED)
        and centre["id"] in VOLUME_CAPABLE
    ]
    # A100 first, then High before Medium: the known-good engine target wins.
    return sorted(
        rows,
        key=lambda row: (
            "A100" not in row[1],
            "SXM" not in row[1],
            row[2] != "High",
            row[2] != "Medium",
            row[0],
        ),
    )


def main() -> int:
    rows = availability()
    if not rows:
        print("no A100 or H100 at usable stock in a volume-capable datacentre")
        return 1
    print(f"{'datacentre':<12} {'gpu':<30} stock")
    for centre, gpu, stock in rows:
        print(f"{centre:<12} {gpu:<30} {stock}")
    print(f"\nplace the volume in: {rows[0][0]}  (for {rows[0][1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
