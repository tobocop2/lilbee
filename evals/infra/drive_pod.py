#!/usr/bin/env python3
"""Provision a RunPod GPU pod, run a script on it, bring results back, tear down.

Provisioning retries are free (RunPod bills on provision, not on attempt) and
multi-GPU capacity flaps, so create is a retry loop. Teardown is in a finally
plus a terminate-after backstop, because a pod outliving its run is the only
expensive mistake here.

Usage: drive_pod.py <script.sh> <gpu_count> [done_marker] [log_path]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tomllib
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
KEY = str(Path.home() / ".ssh" / "runpod_qa")
SSH = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
       "-o", "ConnectTimeout=20", "-o", "IdentitiesOnly=yes", "-i", KEY]
IMAGE = "runpod/pytorch:1.0.7-rc.138-cu1281-torch271-ubuntu2404"
GPU = "NVIDIA H100 80GB HBM3"
FLOOR = 5.0  # tear down below this balance rather than risk a mid-run reclaim


def _key() -> str:
    cfg = tomllib.loads(Path.home().joinpath(".runpod/config.toml").read_text())
    return cfg.get("apikey") or cfg["default"]["api_key"]


def graphql(q: str) -> dict:
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={_key()}",
        data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=30) as fh:
        return json.loads(fh.read())


def balance() -> float:
    out = subprocess.run(["runpodctl", "user"], capture_output=True, text=True)
    try:
        return float(json.loads(out.stdout)["clientBalance"])
    except Exception:
        return 99.0


def create(gpus: int, name: str) -> str | None:
    # Backstop against a pod outliving its run, so it has to exceed the run.
    # A full-corpus ingest plus merge is hours, not the 3 a validation run needs.
    hours = float(os.environ.get("HOURS", "3"))
    at = (datetime.now(UTC) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = subprocess.run(
        ["runpodctl", "pod", "create", "--name", name, "--image", IMAGE,
         # 8.8M passages at 4096 dims is ~144GB of vectors per copy, and a merge
         # holds the shards and the merged index at once, so the disk is a knob.
         "--gpu-id", GPU, "--gpu-count", str(gpus),
         "--container-disk-in-gb", os.environ.get("DISK", "150"),
         "--terminate-after", at, "--ports", "22/tcp"],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or '"error"' in out.stdout:
        return None
    try:
        return json.loads(out.stdout)["id"]
    except Exception:
        return None


def endpoint(pod: str, timeout: int = 600) -> tuple[str, int] | None:
    end = time.time() + timeout
    while time.time() < end:
        d = graphql(f'query{{pod(input:{{podId:"{pod}"}}){{runtime{{ports{{ip isIpPublic privatePort publicPort}}}}}}}}')
        # runtime is present-but-null while the pod boots, so `.get(k, {})` is
        # not enough; every level needs `or {}`.
        pod_d = (d.get("data") or {}).get("pod") or {}
        for p in (pod_d.get("runtime") or {}).get("ports") or []:
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                return p["ip"], p["publicPort"]
        time.sleep(10)
    return None


def main() -> int:
    script = Path(sys.argv[1])
    gpus = int(sys.argv[2])
    marker = sys.argv[3] if len(sys.argv) > 3 else "/root/RUN_DONE"
    logpath = sys.argv[4] if len(sys.argv) > 4 else "/root/pergpu.log"
    token = Path.home().joinpath(".cache/huggingface/token").read_text().strip()

    pod = None
    deadline = time.time() + 1800
    n = 0
    while pod is None and time.time() < deadline:
        n += 1
        pod = create(gpus, f"run-{script.stem}")
        if pod is None:
            print(f"  attempt {n}: no {gpus}xH100 capacity", flush=True)
            time.sleep(20)
    if pod is None:
        print("FAILED: never provisioned")
        return 1
    print(f"provisioned {pod} ({gpus}xH100) after {n} attempt(s)", flush=True)

    try:
        ep = endpoint(pod)
        if ep is None:
            print("FAILED: no SSH")
            return 1
        host, port = ep
        tgt = f"root@{host}"
        print(f"ssh {tgt}:{port}", flush=True)
        for _ in range(30):
            if subprocess.run(["ssh", *SSH, "-p", str(port), tgt, "true"],
                              capture_output=True).returncode == 0:
                break
            time.sleep(10)

        subprocess.run(["scp", *SSH, "-P", str(port), str(script), f"{tgt}:/root/{script.name}"],
                       check=True, capture_output=True, text=True)
        # PAYLOAD ships a locally built wheel so a fix under review can be
        # measured without pushing it anywhere.
        payload = os.environ.get("PAYLOAD")
        if payload:
            subprocess.run(["ssh", *SSH, "-p", str(port), tgt, "mkdir -p /root/payload"],
                           check=True, capture_output=True, text=True)
            for item in sorted(Path(payload).iterdir()):
                subprocess.run(["scp", *SSH, "-P", str(port), str(item), f"{tgt}:/root/payload/"],
                               check=True, capture_output=True, text=True)
            print(f"payload uploaded: {[p.name for p in sorted(Path(payload).iterdir())]}", flush=True)
        # -n plus setsid and a closed stdin: without them the remote wrapper keeps
        # the ssh channel open for the whole run, so this call blocks and the
        # progress/low-balance loop below never gets to run.
        subprocess.run(["ssh", "-n", *SSH, "-p", str(port), tgt,
                        f"chmod +x /root/{script.name} && rm -f {marker} && "
                        f"HF_TOKEN={token} {os.environ.get('RUN_ENV', '')} "
                        f"setsid bash /root/{script.name} </dev/null >/dev/null 2>&1 & echo ok"],
                       check=True, capture_output=True, text=True)
        print("launched", flush=True)

        # Watch for as long as the pod is allowed to live, or a long run gets
        # abandoned mid-flight while still billing.
        end = time.time() + float(os.environ.get("HOURS", "3")) * 3600
        while time.time() < end:
            done = subprocess.run(["ssh", *SSH, "-p", str(port), tgt, f"test -f {marker} && echo YES"],
                                  capture_output=True, text=True)
            if "YES" in done.stdout:
                print("run finished", flush=True)
                break
            bal = balance()
            if bal < FLOOR:
                print(f"LOW BALANCE ${bal:.2f} - stopping early", flush=True)
                break
            tail = subprocess.run(["ssh", *SSH, "-p", str(port), tgt,
                                   f"tr '\\r' '\\n' < {logpath} 2>/dev/null | grep -v '^$' | tail -1"],
                                  capture_output=True, text=True)
            if tail.stdout.strip():
                print(f"  [{bal:.2f}] {tail.stdout.strip()[:130]}", flush=True)
            time.sleep(45)

        print("--- RESULT ---", flush=True)
        res = subprocess.run(["ssh", *SSH, "-p", str(port), tgt,
                              f"grep -aE 'PERGPU|worker |cards=|FATAL' {logpath} | tail -20"],
                             capture_output=True, text=True)
        print(res.stdout)
        subprocess.run(["scp", *SSH, "-P", str(port), f"{tgt}:{logpath}",
                        str(script.parent / f"{script.stem}.out.log")], capture_output=True, text=True)
    finally:
        subprocess.run(["runpodctl", "pod", "delete", pod], capture_output=True, text=True)
        print(f"torn down {pod}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
