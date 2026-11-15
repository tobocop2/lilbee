"""Copy the `mistralrs` tool of one upstream mistral.rs archive into the engine wheel tree.

Usage: repack_mistralrs.py <archive> <bin folder>

The archive holds the tool and nothing else. Any other entry stops the copy.
"""

from __future__ import annotations

import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

TOOLS = frozenset({"mistralrs", "mistralrs.exe"})
TOOL_MODE = 0o755


def _extract(archive: Path, scratch: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(scratch)
    else:
        with tarfile.open(archive) as t:
            t.extractall(scratch, filter="data")


def repack(archive: Path, bin_dir: Path) -> None:
    """Copy the tool of *archive* into *bin_dir*."""
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        _extract(archive, scratch)
        files = sorted(p for p in scratch.rglob("*") if p.is_file() or p.is_symlink())
        tools: list[Path] = []
        for path in files:
            rel = PurePosixPath(path.relative_to(scratch).as_posix())
            if path.is_symlink():
                raise SystemExit(f"{archive.name}: {rel} is a link")
            if len(rel.parts) != 1 or rel.name not in TOOLS:
                raise SystemExit(f"{archive.name}: {rel} is not the tool")
            tools.append(path)
        if len(tools) != 1:
            raise SystemExit(f"{archive.name}: expected one tool, found {len(tools)}")
        bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tools[0], bin_dir / tools[0].name)
        (bin_dir / tools[0].name).chmod(TOOL_MODE)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    repack(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
