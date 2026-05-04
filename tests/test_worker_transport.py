"""Tests for the worker transport Protocol layer.

Concrete transport behaviour (PipeSpawner / PipeChannel) lives in
``test_worker_transport_pipe.py``; this file covers the default-spawner
helper that the pool uses to pick its IPC backend.
"""

from __future__ import annotations

from lilbee.providers.worker.transport import default_spawner
from lilbee.providers.worker.transport_pipe import PipeSpawner


def test_default_spawner_returns_pipe_spawner():
    assert isinstance(default_spawner(), PipeSpawner)
