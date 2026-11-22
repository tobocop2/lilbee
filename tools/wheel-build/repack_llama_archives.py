"""Copy llama-server and its libraries from the upstream archives into the wheel tree.

Usage: repack_llama_archives.py <bin-dir> <archive> [<runtime-archive> ...]

The first archive holds the server. The others hold libraries only. The licence of LLVM OpenMP goes
into the wheel project, two folders above the bin folder.
"""

import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

SERVER = {"llama-server", "llama-server.exe"}
LIBRARY = re.compile(r"lib.+\.so(\.[\d.]+)?|.+\.dll|lib.+(\.[\d.]+)?\.dylib")
OPENMP_LICENCE = "LICENSE-LLVM-OpenMP"


def install_name(entry: Path) -> str:
    """The file name dependents load for a dylib: its install id, else its own name."""
    if entry.suffix != ".dylib":
        return entry.name
    ident = subprocess.run(["otool", "-D", str(entry)], capture_output=True, text=True, check=True)
    lines = ident.stdout.strip().splitlines()
    return Path(lines[-1].strip()).name if len(lines) > 1 else entry.name


def repack(archive: Path, bin_dir: Path, with_server: bool) -> None:
    """Copy the server and the libraries of one archive; a runtime-only archive has no server."""
    with tempfile.TemporaryDirectory() as scratch:
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                for member in z.namelist():
                    z.extract(member, scratch)
        else:
            with tarfile.open(archive) as t:
                t.extractall(scratch, filter="data")

        def wanted(path: Path) -> bool:
            return path.name in SERVER if with_server else bool(LIBRARY.fullmatch(path.name))

        roots = {p.parent for p in Path(scratch).rglob("*") if p.is_file() and wanted(p)}
        if len(roots) != 1:
            raise SystemExit(f"{archive.name}: expected one folder to repack, found {len(roots)}")
        (root,) = roots
        # The archive carries the LLVM OpenMP licence beside libomp.dll.
        omp_licence = root / OPENMP_LICENCE
        if with_server and omp_licence.is_file():
            shutil.copy2(omp_licence, bin_dir.parents[1] / omp_licence.name)
        for entry in sorted(root.iterdir()):
            if entry.is_symlink():
                continue  # a link only repeats a file that the loop copies once
            if entry.is_file() and (entry.name in SERVER or LIBRARY.fullmatch(entry.name)):
                target = bin_dir / install_name(entry)
                if target.exists():
                    raise SystemExit(f"{archive.name}: two files install as {target.name}")
                shutil.copy2(entry, target)


def main(bin_dir: Path, archives: list[Path]) -> None:
    for archive in archives:
        repack(archive, bin_dir, with_server=archive is archives[0])
    if not any((bin_dir / name).exists() for name in SERVER):
        raise SystemExit(f"{archives[0].name}: no llama-server in the tree")
    has_openmp = (bin_dir / "libomp.dll").exists()
    if has_openmp and not (bin_dir.parents[1] / OPENMP_LICENCE).is_file():
        raise SystemExit(f"{archives[0].name}: libomp.dll without {OPENMP_LICENCE}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), [Path(argument) for argument in sys.argv[2:]])
