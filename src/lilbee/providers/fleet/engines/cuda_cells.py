"""The pick of a CUDA build by compute capability, shared by the adapters that ship one per family."""

from __future__ import annotations


def cuda_cell(cells: tuple[int, ...], compute: tuple[int, int]) -> int | None:
    """The newest of *cells* of the same major version that does not exceed *compute*.

    A cell is a compute capability as ``major * 10 + minor``. Machine code of one major
    version does not run on another, and code for a newer minor does not run on an older card.
    """
    major, minor = compute
    fits = [c for c in cells if c // 10 == major and c % 10 <= minor]
    return max(fits, default=None)
