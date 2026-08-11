#!/usr/bin/env python3
"""Write the server's OpenAPI schema to a file.

The release attaches the result as ``openapi.json`` so a client can pin the
contract for a version without a source checkout and a venv to introspect the
app. The docs site renders the same file through Redoc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lilbee.server.app import create_app

_DEFAULT_OUT = Path("openapi.json")


def render_schema() -> str:
    """Serialize the app's OpenAPI schema as indented JSON."""
    return json.dumps(create_app().openapi_schema.to_schema(), indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_schema(), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
