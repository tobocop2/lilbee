"""The index-mismatch refusal names the rebuild command, whether or not the dimensions match."""

from __future__ import annotations

import pytest

from lilbee.data.store import EmbeddingModelMismatchError


def _mismatch(current_dim: int) -> EmbeddingModelMismatchError:
    return EmbeddingModelMismatchError(
        persisted_model="old-embed",
        persisted_dim=8,
        current_model="new-embed",
        current_dim=current_dim,
    )


@pytest.mark.parametrize("current_dim", [8, 16], ids=["same-dim", "other-dim"])
def test_the_refusal_names_the_rebuild_command(current_dim: int) -> None:
    assert "`lilbee rebuild`" in str(_mismatch(current_dim))


@pytest.mark.parametrize("current_dim", [8, 16], ids=["same-dim", "other-dim"])
def test_the_status_record_carries_the_same_text(current_dim: int) -> None:
    error = _mismatch(current_dim)
    assert error.describe().message == str(error)
