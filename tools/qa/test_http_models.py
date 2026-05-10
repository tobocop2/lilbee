"""T2 HTTP models. List-style endpoints + the role assignment 422 negative path."""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.http
def test_models_installed_returns_200(server_url: str) -> None:
    response = httpx.get(f"{server_url}/api/models/installed", timeout=30.0)
    assert response.status_code == httpx.codes.OK
    payload = response.json()
    # Response shape varies (sometimes a list, sometimes {"models": [...]}).
    assert isinstance(payload, dict | list), payload


@pytest.mark.http
def test_models_catalog_returns_200(server_url: str) -> None:
    """Featured catalog should always be available; doesn't depend on installed models."""
    response = httpx.get(f"{server_url}/api/models/catalog", timeout=30.0)
    assert response.status_code == httpx.codes.OK
    payload = response.json()
    assert isinstance(payload, dict | list)


@pytest.mark.http
def test_catalog_includes_fit_and_size_variants(server_url: str) -> None:
    """`/api/models/catalog` rows carry server-computed `fit` and
    `size_variants` fields (added in PR #218 for the TUI fit-chip and
    size-strip widgets). A regression that drops these would silently
    break TUI rendering, so gate the response shape here.

    Doesn't pin specific fit values (those depend on the runner's RAM)
    or specific variants (those depend on the featured catalog data).
    Asserts the keys exist, the types match, and at least one row is
    present so the assertion isn't vacuous on an empty catalog.
    """
    response = httpx.get(
        f"{server_url}/api/models/catalog",
        params={"task": "chat", "limit": 5},
        timeout=30.0,
    )
    assert response.status_code == httpx.codes.OK, response.text
    payload = response.json()

    # Endpoint shape: either {"models": [...], ...} envelope OR a bare list.
    rows = payload.get("models", payload) if isinstance(payload, dict) else payload
    assert isinstance(rows, list), payload
    assert rows, f"catalog returned no chat-task rows: {payload}"

    sample = rows[0]
    assert "fit" in sample, f"catalog row missing `fit` field: {sample}"
    assert "size_variants" in sample, f"catalog row missing `size_variants` field: {sample}"

    # `fit` is FitLevel | None: string label or null.
    fit = sample["fit"]
    assert fit is None or isinstance(fit, str), f"unexpected fit type: {fit!r}"

    variants = sample["size_variants"]
    assert isinstance(variants, list), f"size_variants must be a list: {variants!r}"
    # If any variants exist, each should carry the documented fields.
    for variant in variants:
        assert isinstance(variant, dict), f"variant must be dict: {variant!r}"
        for key in ("label", "ref"):
            assert key in variant, f"variant missing `{key}`: {variant}"


@pytest.mark.http
def test_unknown_role_assignment_rejected(server_url: str) -> None:
    """PUT /api/models/<role> with an unknown role does not 5xx; the surface
    rejects the request via auth / validation / method-not-allowed / not-found.
    Any 4xx is acceptable; the contract is "doesn't crash the server"."""
    response = httpx.put(
        f"{server_url}/api/models/not-a-real-role",
        json={"model": "anything"},
        timeout=15.0,
    )
    assert httpx.codes.BAD_REQUEST <= response.status_code < httpx.codes.INTERNAL_SERVER_ERROR
