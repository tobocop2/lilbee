"""Tests for the worker transport Protocol layer (spawner registry).

Concrete transport behaviour (PipeSpawner / PipeChannel) lives in
``test_worker_transport_pipe.py``; this file covers the backend
selection helper that the pool uses to pick a spawner.
"""

from __future__ import annotations

import pytest

from lilbee.providers.worker.transport import SPAWNERS, make_spawner
from lilbee.providers.worker.transport_pipe import PipeSpawner


def test_pipe_backend_returns_pipe_spawner():
    assert isinstance(make_spawner("pipe"), PipeSpawner)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="worker_pool_backend"):
        make_spawner("imaginary")


def test_registry_lists_pipe():
    assert "pipe" in SPAWNERS
