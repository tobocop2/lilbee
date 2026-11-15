"""Check that a text-embeddings-router wheel carries its binary and every licence file it needs.

Usage: check_tei_wheel.py <wheel> <backend> <licence-expression-file>

The wheel must hold an executable router, LICENSE-tei, a non-empty THIRD-PARTY/tei folder and the
licence expression that the build wrote. A CUDA backend also needs a non-empty NOTICE-NVIDIA-CUDA
and the NVIDIA licence term; no other backend may carry either. A Linux wheel also holds LICENSE-openssl.
An expression with MPL-2.0 needs a non-empty SOURCE-MPL-2.0. A router that holds ONNX Runtime is refused.
Prints one line for each problem and exits 1, or prints OK.
"""

from __future__ import annotations

import email
import stat
import sys
import zipfile
from pathlib import Path

NVIDIA_TERM = "LicenseRef-NVIDIA-CUDA-EULA"
NVIDIA_NOTICE = "NOTICE-NVIDIA-CUDA"
ROUTER = "lilbee_engine_tei/bin/text-embeddings-router"
ORT_MARK = b"onnxruntime"


def problems(wheel: Path, backend: str, expected: str) -> list[str]:
    """Each way in which *wheel* lacks what a pack of *backend* needs."""
    found: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        metadata = [name for name in infos if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            return [f"{wheel.name}: expected one METADATA file, found {len(metadata)}"]
        dist = metadata[0].removesuffix("/METADATA")
        declared = email.message_from_bytes(archive.read(metadata[0])).get("License-Expression")
        has_ort = ROUTER in infos and ORT_MARK in archive.read(ROUTER)
    if has_ort:
        found.append("the router holds ONNX Runtime, whose notices this wheel does not carry")
    router = infos.get(ROUTER)
    if router is None:
        found.append(f"no {ROUTER}")
    elif not (
        stat.S_ISREG(router.external_attr >> 16) and router.external_attr >> 16 & stat.S_IXUSR
    ):
        found.append(f"{ROUTER} is not an executable file")
    required = ["LICENSE-tei"]
    if "manylinux" in wheel.name:
        required.append("LICENSE-openssl")
    for name in required:
        info = infos.get(f"{dist}/licenses/{name}")
        if info is None or info.file_size == 0:
            found.append(f"no non-empty licenses/{name}")
    third = [
        n
        for n, i in infos.items()
        if n.startswith(f"{dist}/licenses/THIRD-PARTY/tei/") and i.file_size
    ]
    if not third:
        found.append("no non-empty file under licenses/THIRD-PARTY/tei/")
    notice = infos.get(f"{dist}/licenses/{NVIDIA_NOTICE}")
    cuda = backend.startswith("cuda")
    if cuda and (notice is None or notice.file_size == 0):
        found.append(f"no non-empty licenses/{NVIDIA_NOTICE}")
    if not cuda and notice is not None:
        found.append(f"licenses/{NVIDIA_NOTICE} in a build with no NVIDIA files")
    if declared != expected:
        found.append(f"License-Expression is {declared!r}, the build wrote {expected!r}")
    terms = (declared or "").replace(" AND ", " ").split()
    mpl = infos.get(f"{dist}/licenses/SOURCE-MPL-2.0")
    if "MPL-2.0" in terms and (mpl is None or mpl.file_size == 0):
        found.append("no non-empty licenses/SOURCE-MPL-2.0 for the MPL-2.0 crates")
    if "Apache-2.0" not in terms:
        found.append("License-Expression lacks Apache-2.0")
    if cuda != (NVIDIA_TERM in terms):
        found.append(f"License-Expression and backend {backend} disagree on {NVIDIA_TERM}")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    expected = Path(argv[3]).read_text(encoding="utf-8").strip()
    found = problems(Path(argv[1]), argv[2], expected)
    for line in found:
        print(line)
    if not found:
        print("OK")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
