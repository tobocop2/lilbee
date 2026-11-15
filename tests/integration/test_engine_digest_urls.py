"""Every row of the shipped pack table points at an asset that a public download can fetch."""

from __future__ import annotations

import urllib.request
from typing import Any

import pytest

from lilbee.providers.fleet.engine_digests import MANIFESTS

_TIMEOUT_S = 60
_ROWS = [
    pytest.param(engine, version, key, row, id=f"{engine}-{version}-{key}")
    for engine, versions in MANIFESTS.items()
    for version, cells in versions.items()
    for key, row in cells.items()
]


def test_the_shipped_table_lists_rows() -> None:
    assert _ROWS


@pytest.mark.parametrize(("engine", "version", "key", "row"), _ROWS)
def test_the_download_url_of_each_row_answers_200(
    engine: str, version: str, key: str, row: dict[str, Any]
) -> None:
    request = urllib.request.Request(row["url"], headers={"Range": "bytes=0-0"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # noqa: S310
        assert response.status in (200, 206)
