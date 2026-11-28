"""The one reader for TOML files and text."""

import tomllib
from typing import Any, BinaryIO


def loads_toml(text: str) -> dict[str, Any]:
    """Parse *text*; a document nested too deep raises TOMLDecodeError, as a malformed one does."""
    try:
        return tomllib.loads(text)
    except RecursionError:
        raise tomllib.TOMLDecodeError("nested too deeply") from None


def load_toml(f: BinaryIO) -> dict[str, Any]:
    """Parse the UTF-8 TOML in the binary file *f*; same errors as :func:`loads_toml`."""
    return loads_toml(f.read().decode())
