"""The read rate of the volume that holds a model, which stretches its cold-load wait."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.providers.fleet import read_rate as read_rate_mod
from lilbee.providers.fleet.read_rate import measured_read_rate

_MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def _fresh_measurements() -> Iterator[None]:
    measured_read_rate.cache_clear()
    yield
    measured_read_rate.cache_clear()


def _clock(monkeypatch: pytest.MonkeyPatch, *readings: float) -> None:
    """Make the sample read see *readings* from the monotonic clock, in order."""
    values = iter(readings)
    monkeypatch.setattr(read_rate_mod, "time", SimpleNamespace(monotonic=lambda: next(values)))


def test_a_slow_volume_is_measured_at_its_rate(tmp_path: Path, monkeypatch) -> None:
    # Two 1 MiB chunks take the whole one-second budget: the volume reads 2 MiB/s.
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"x" * (8 * _MIB))
    _clock(monkeypatch, 0.0, 0.5, 1.0)
    assert measured_read_rate(model) == 2 * _MIB


def test_a_fast_volume_stops_at_the_sample_size(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"x" * (8 * _MIB))
    monkeypatch.setattr(read_rate_mod, "_SAMPLE_BYTES", 2 * _MIB)
    # The two chunks fill the sample at a quarter second, inside the budget.
    _clock(monkeypatch, 0.0, 0.125, 0.25)
    assert measured_read_rate(model) == 8 * _MIB


def test_a_file_that_ends_inside_the_sample_is_not_measured(tmp_path: Path) -> None:
    # The whole file read faster than the budget: it says nothing about a long load.
    model = tmp_path / "small.gguf"
    model.write_bytes(b"x" * 2048)
    assert measured_read_rate(model) == 0


def test_a_sample_with_no_clock_movement_is_not_measured(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"x" * (4 * _MIB))
    monkeypatch.setattr(read_rate_mod, "_SAMPLE_BYTES", 2 * _MIB)
    _clock(monkeypatch, 5.0, 5.0, 5.0)
    assert measured_read_rate(model) == 0


def test_a_path_that_does_not_open_is_not_measured(tmp_path: Path) -> None:
    assert measured_read_rate(tmp_path / "missing.gguf") == 0


def test_a_model_folder_is_measured_on_its_largest_file(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "model"
    (folder / "nested").mkdir(parents=True)
    (folder / "config.json").write_bytes(b"{}")
    (folder / "nested" / "model.safetensors").write_bytes(b"x" * (8 * _MIB))
    _clock(monkeypatch, 0.0, 0.5, 1.0)
    assert measured_read_rate(folder) == 2 * _MIB


def test_an_empty_model_folder_is_not_measured(tmp_path: Path) -> None:
    folder = tmp_path / "model"
    folder.mkdir()
    assert measured_read_rate(folder) == 0


def test_a_volume_is_read_once_per_file(tmp_path: Path, monkeypatch) -> None:
    # Every plan reads the rate; a replan must not pay the sample again.
    model = tmp_path / "chat.gguf"
    model.write_bytes(b"x" * (8 * _MIB))
    _clock(monkeypatch, 0.0, 0.5, 1.0)
    first = measured_read_rate(model)
    assert measured_read_rate(model) == first
