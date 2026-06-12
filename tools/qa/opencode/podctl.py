#!/usr/bin/env python3
"""RunPod infrastructure-as-code for the QA/reel pods. Stdlib only.

Reads the API key from ~/.runpod/config.toml (runpodctl's config). State that
must survive pods (models, corpus, engine cache) lives on a named network
volume; pods are on-demand secure-cloud so they cannot be spot-evicted.

  podctl.py volume            ensure the QA network volume exists, print id
  podctl.py up                create the QA pod (attaches the volume), print ssh
  podctl.py ls                list pods
  podctl.py ssh <podId>       print the ssh command for a pod
  podctl.py stop <podId>      stop a pod (volume persists)
  podctl.py rm <podId>        terminate a pod
"""

from __future__ import annotations

import json
import os
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
VOLUME_NAME = "lilbee-qa"
VOLUME_GB = 250
DATACENTER = "US-KS-2"
GPU_TYPES = [
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
]
GPU_COUNT = 3
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
POD_NAME = "lilbee-qa"
CONTAINER_DISK_GB = 450


def _key() -> str:
    cfg = tomllib.load((Path.home() / ".runpod" / "config.toml").open("rb"))
    key = cfg.get("apikey") or cfg.get("default", {}).get("api_key")
    if not key:
        sys.exit("no RunPod api key in ~/.runpod/config.toml (run: runpodctl config --apiKey ...)")
    return key


def _req(method: str, path: str, body: dict | None = None) -> dict | list:
    req = urllib.request.Request(  # noqa: S310  literal https API base
        f"{API}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_key()}"},
    )
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310  literal https API base
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        sys.exit(f"{method} {path} -> HTTP {exc.code}: {exc.read().decode()[:500]}")


def ensure_volume() -> str:
    for vol in _req("GET", "/networkvolumes"):
        if vol["name"] == VOLUME_NAME:
            print(f"volume {vol['id']} ({vol['name']}, {vol['size']}GB, {vol['dataCenterId']})")
            return vol["id"]
    vol = _req(
        "POST",
        "/networkvolumes",
        {"name": VOLUME_NAME, "size": VOLUME_GB, "dataCenterId": DATACENTER},
    )
    print(f"created volume {vol['id']} ({VOLUME_GB}GB in {DATACENTER})")
    return vol["id"]


def up() -> None:
    """Create the QA pod: on-demand secure cloud, never interruptible.

    Attaches the named network volume when RUNPOD_QA_VOLUME is set; otherwise
    runs on container disk (450GB; durability comes from rolling harvests).
    Volume placement is DC-bound and the REST API exposes no availability
    query, so the volume is adopted in whatever DC pods reliably land in.
    """
    body = {
        "name": POD_NAME,
        "imageName": IMAGE,
        "gpuTypeIds": GPU_TYPES,
        "gpuCount": GPU_COUNT,
        "cloudType": "SECURE",
        "computeType": "GPU",
        "interruptible": False,
        "containerDiskInGb": CONTAINER_DISK_GB,
        "supportPublicIp": True,
        "ports": ["22/tcp"],
    }
    volume_id = os.environ.get("RUNPOD_QA_VOLUME")
    if volume_id:
        body["networkVolumeId"] = volume_id
        body["volumeMountPath"] = "/workspace"
    pod = _req("POST", "/pods", body)
    pod_id = pod["id"]
    print(f"pod {pod_id} creating; waiting for runtime...")
    for _ in range(60):
        time.sleep(10)
        info = _req("GET", f"/pods/{pod_id}")
        status = info.get("desiredStatus")
        if status == "RUNNING":
            machine = info.get("machine") or {}
            print(
                f"pod {pod_id} RUNNING in {info.get('dataCenterId') or machine.get('dataCenterId')}"
            )
            host_id = machine.get("podHostId") or f"{pod_id}-<see console>"
            print(f"ssh: ssh -tt -i ~/.ssh/runpod_qa {host_id}@ssh.runpod.io")
            return
        print(f"  status: {status}")
    sys.exit("pod did not reach RUNNING within 10 minutes")


def _acct_suffix(info: dict) -> str:
    # The ssh-proxy username is <podId>-<machine user fragment>; RunPod exposes
    # it via the pod's machine info when available, else the console shows it.
    machine = info.get("machine") or {}
    return machine.get("podHostId", "").split("-")[-1] or "<see console for ssh user>"


def ls() -> None:
    pods = _req("GET", "/pods")
    if not pods:
        print("no pods")
        return
    for pod in pods:
        host = (pod.get("machine") or {}).get("podHostId", "")
        print(
            f"{pod['id']}  {pod.get('name', '')}  {pod.get('desiredStatus', '')}"
            f"  gpus={pod.get('gpuCount', '?')}  host={host}"
        )


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ls"
    if cmd == "volume":
        ensure_volume()
    elif cmd == "up":
        up()
    elif cmd == "ls":
        ls()
    elif cmd == "ssh" and len(sys.argv) > 2:
        info = _req("GET", f"/pods/{sys.argv[2]}")
        host_id = (info.get("machine") or {}).get("podHostId", "")
        print(f"ssh -tt -i ~/.ssh/runpod_qa {host_id}@ssh.runpod.io")
    elif cmd == "stop" and len(sys.argv) > 2:
        _req("POST", f"/pods/{sys.argv[2]}/stop")
        print("stopped")
    elif cmd == "rm" and len(sys.argv) > 2:
        _req("DELETE", f"/pods/{sys.argv[2]}")
        print("terminated")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
