from __future__ import annotations

from collections.abc import Callable

# Puts a tool on PATH: (name, stdout it prints, exit code). The conftest fixture builds it.
FakeTool = Callable[[str, str, int], None]
