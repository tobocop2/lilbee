"""Check the notices and the licence expression of an ik_llama.cpp wheel.

Usage: check_ik_wheel.py <wheel> <backend> <licence-expression> <upstream-commit> <adapter.py>

The wheel must hold each notice of the code that the server carries, with its text, one copy of the
Apache-2.0 text, and the licence expression that the build stated. A CUDA wheel also holds the
NVIDIA EULA. The expression and the commit must be the ones that the lilbee adapter pins; the
adapter file is parsed, not imported. Prints one line for each problem to stderr and exits 1.
"""

from __future__ import annotations

import ast
import sys
import zipfile
from pathlib import Path

GRANT = "Permission is hereby granted"
THIRD = "dist-info/licenses/THIRD-PARTY/ik_llama.cpp/"
NVIDIA_NOTICE = "dist-info/licenses/NOTICE-NVIDIA-CUDA"
NVIDIA_TEXTS = ["Attachment A", "libcublasLt", "libcudart", "libcublas"]
MIN_EULA_CHARS = 20000
APACHE_PHRASE = "TERMS AND CONDITIONS FOR USE, REPRODUCTION"
NOTICES = {
    "dist-info/licenses/LICENSE-ik_llama.cpp": [GRANT],
    THIRD + "examples/server/public/index.html.gz.NOTICE": ["@license React", GRANT],
    THIRD + "examples/server/public_llamacpp/index_llamacpp.html.gz.NOTICE": [
        "Mozilla Foundation",
        "Apache License, Version 2.0",
    ],
    THIRD + "vendor/nlohmann/json.hpp.NOTICE": ["Niels Lohmann", GRANT],
    THIRD + "vendor/cpp-httplib/httplib.h.NOTICE": ["Yuji Hirose", GRANT],
    THIRD + "vendor/stb/stb_image_resize2.h.NOTICE": [
        "Sean Barrett",
        "ALTERNATIVE A - MIT License",
        GRANT,
    ],
}


def notice_problems(files: list[tuple[str, bytes]], backend: str) -> list[str]:
    """Each notice of a pack of *backend* that the licence *files* of the wheel lack."""
    wanted = dict(NOTICES)
    if backend == "cuda":
        wanted[NVIDIA_NOTICE] = NVIDIA_TEXTS
    found = []
    for suffix, phrases in wanted.items():
        hits = [data for name, data in files if name.endswith(suffix)]
        if len(hits) != 1:
            found.append(f"the wheel holds {len(hits)} entries that end with {suffix}, not one")
        text = hits[0].decode("utf-8") if len(hits) == 1 else ""
        found += [f"{suffix} lacks the text: {p}" for p in phrases if p not in text]
        if suffix == NVIDIA_NOTICE and len(text) < MIN_EULA_CHARS:
            found.append("NOTICE-NVIDIA-CUDA is shorter than the EULA")
    # Apache-2.0 section 4(a): the wheel carries one copy of the licence text.
    apache = [name for name, data in files if APACHE_PHRASE.encode() in data]
    if len(apache) != 1:
        found.append(f"the Apache-2.0 text is in {len(apache)} licence files, not one: {apache}")
    return found


def metadata_problems(metadata: list[str], expression: str) -> list[str]:
    """The one METADATA file of the wheel must state the licence expression of the build."""
    if len(metadata) != 1:
        return [f"the wheel holds {len(metadata)} entries that end with .dist-info/METADATA, not one"]
    if f"License-Expression: {expression}\n" not in metadata[0]:
        return [f"METADATA lacks License-Expression: {expression}"]
    return []


def pin_problems(adapter: Path, backend: str, expression: str, commit: str) -> list[str]:
    """The licence expression and the commit must be the ones that the adapter pins."""
    values = {
        node.targets[0].id: node.value
        for node in ast.parse(adapter.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    licences = "_CUDA_LICENCES" if backend == "cuda" else "_CPU_LICENCES"
    names = ("IK_LLAMA_COMMIT", "_CPU_LICENCES", "_CUDA_LICENCES")
    missing = [name for name in names if name not in values]
    if missing:
        return [f"the adapter at the lilbee tag has no {name}" for name in missing]
    pinned = {name: ast.literal_eval(values[name]) for name in names}
    found = []
    if pinned[licences] != [expression]:
        found.append(f"the lilbee pin names {pinned[licences]}, the wheel states {expression}")
    if pinned["IK_LLAMA_COMMIT"] != commit:
        found.append("the lilbee pin names another ik_llama.cpp commit than this pack builds")
    return found


def main(argv: list[str]) -> None:
    if len(argv) != 6:
        raise SystemExit(__doc__)
    wheel, backend, expression, commit, adapter = argv[1:]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        # A list: a zip file can hold two entries with one name, and each one counts.
        files = [(name, archive.read(name)) for name in names if "dist-info/licenses/" in name]
        metadata = [
            archive.read(name).decode("utf-8")
            for name in names
            if name.endswith(".dist-info/METADATA")
        ]
    found = [
        *notice_problems(files, backend),
        *metadata_problems(metadata, expression),
        *pin_problems(Path(adapter), backend, expression, commit),
    ]
    if found:
        raise SystemExit("\n".join(found))
    print("notices, Apache-2.0 text and licence expression: ok")


if __name__ == "__main__":
    main(sys.argv)
