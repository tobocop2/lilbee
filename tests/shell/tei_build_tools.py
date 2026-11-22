"""Write stand-in wheels for the build tools of the text-embeddings-inference pack.

Usage: tei_build_tools.py <wheel-folder> <requirements-file>...

Each requirements file keeps its names. It gets the version and the digest of the stand-in, so a pip
that reads only the folder installs the names of the real file. The stand-ins need each other as the
real tools do.
"""

import hashlib
import re
import sys
import zipfile
from pathlib import Path

NEEDS: dict[str, tuple[str, ...]] = {
    "altgraph": (),
    "build": ("packaging", "pyproject_hooks"),
    "cmake": (),
    "colorama": (),
    "delocate": ("macholib", "packaging", "typing_extensions"),
    "macholib": ("altgraph",),
    "ninja": (),
    "packaging": (),
    "pyproject_hooks": (),
    "setuptools": (),
    "typing_extensions": (),
    "wheel": ("packaging",),
}
# The stand-in builds only with the tools that are installed, so an isolated build fails.
BUILD = """import os, shutil, subprocess, sys
import packaging, pyproject_hooks, setuptools
args = sys.argv[1:]
if "--no-isolation" not in args:
    sys.exit("stand-in build: an isolated build installs tools that no file pins")
out = args[args.index("--outdir") + 1]
os.makedirs(out)
if os.environ.get("STEP_LOG"):
    call = " ".join(args[: args.index("--outdir")])
    with open(os.environ["STEP_LOG"], "a") as log:
        log.write("build " + call + " | cwd=" + os.path.basename(os.getcwd()) + "\\n")
if shutil.which("tei-wheel-pack"):
    expression = os.path.join(out, "expression.txt")
    subprocess.check_call(["tei-wheel-pack", out, expression, os.environ["PLATFORM"]])
"""
WHEEL = """import sys
import packaging
assert sys.argv[1:4] == ["tags", "--remove", "--platform-tag"], sys.argv
"""
SCRIPTS = {
    "cmake": {"cmake": "#!/bin/sh\necho stand-in cmake\n"},
    "ninja": {"ninja": "#!/bin/sh\necho stand-in ninja\n"},
    "delocate": {"delocate-wheel": '#!/bin/sh\nexec stub-delocate-wheel "$@"\n'},
}
MAIN = {"build": BUILD, "wheel": WHEEL}


def write_wheel(folder: Path, name: str) -> str:
    """Write the stand-in of one tool; return its digest."""
    info = f"{name}-99.dist-info"
    needs = "".join(f"Requires-Dist: {need}\n" for need in NEEDS[name])
    files = {
        f"{name}/__init__.py": "",
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 99\n{needs}",
        f"{info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    if name in MAIN:
        files[f"{name}/__main__.py"] = MAIN[name]
    for script, text in SCRIPTS.get(name, {}).items():
        files[f"{name}-99.data/scripts/{script}"] = text
    record = f"{info}/RECORD"
    files[record] = "".join(f"{path},,\n" for path in [*files, record])
    wheel = folder / f"{name}-99-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            member = zipfile.ZipInfo(path)
            member.external_attr = (0o100755 if "/scripts/" in path else 0o100644) << 16
            archive.writestr(member, text)
    return hashlib.sha256(wheel.read_bytes()).hexdigest()


def main() -> None:
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True)
    digests = {name: write_wheel(folder, name) for name in NEEDS}
    for file in map(Path, sys.argv[2:]):
        names = re.findall(r"^([A-Za-z_]+)==", file.read_text(encoding="utf-8"), re.MULTILINE)
        lines = [f"{name}==99 --hash=sha256:{digests[name]}\n" for name in names]
        file.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
