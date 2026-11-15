"""Copy one upstream mistral.rs archive into the engine wheel tree.

Usage: repack_mistralrs.py <archive> <bin folder>

The archive holds the `mistralrs` tool, and a CUDA archive also holds its shared libraries in
`lib/`. The tool goes to the bin folder and the libraries to `bin/lib`, where the tool looks for
them. The driver library `libcuda` stays out: the host supplies it. Any other entry stops the copy.
"""

from __future__ import annotations

import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

TOOLS = frozenset({"mistralrs", "mistralrs.exe"})
LIBRARY = re.compile(r"lib[A-Za-z0-9_+-]+\.so(\.\d+)*")
DRIVER = re.compile(r"libcuda\.so(\.\d+)*")
TOOL_MODE = 0o755
FILE_MODE = 0o644


def _extract(archive: Path, scratch: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(scratch)
    else:
        with tarfile.open(archive) as t:
            t.extractall(scratch, filter="data")


def repack(archive: Path, bin_dir: Path) -> list[str]:
    """Fill *bin_dir* from *archive*; return the names of the libraries that were copied."""
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        _extract(archive, scratch)
        files = sorted(p for p in scratch.rglob("*") if p.is_file() or p.is_symlink())
        tools: list[Path] = []
        libs: list[Path] = []
        for path in files:
            rel = PurePosixPath(path.relative_to(scratch).as_posix())
            if path.is_symlink():
                raise SystemExit(f"{archive.name}: {rel} is a link")
            if len(rel.parts) == 1 and rel.name in TOOLS:
                tools.append(path)
            elif len(rel.parts) == 2 and rel.parts[0] == "lib" and DRIVER.fullmatch(rel.name):
                continue
            elif len(rel.parts) == 2 and rel.parts[0] == "lib" and LIBRARY.fullmatch(rel.name):
                libs.append(path)
            else:
                raise SystemExit(f"{archive.name}: {rel} is not the tool or a library in lib/")
        if len(tools) != 1:
            raise SystemExit(f"{archive.name}: expected one tool, found {len(tools)}")
        bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tools[0], bin_dir / tools[0].name)
        (bin_dir / tools[0].name).chmod(TOOL_MODE)
        if libs:
            (bin_dir / "lib").mkdir(exist_ok=True)
        for lib in libs:
            shutil.copyfile(lib, bin_dir / "lib" / lib.name)
            (bin_dir / "lib" / lib.name).chmod(FILE_MODE)
        return [lib.name for lib in libs]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    for name in repack(Path(argv[1]), Path(argv[2])):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
