"""Tests for the per-task source chip [local | cloud | both]."""

from __future__ import annotations

from lilbee.cli.tui.screens.catalog_utils import SourceMode, next_source_mode


def test_source_mode_cycle_loops() -> None:
    assert next_source_mode(SourceMode.LOCAL) is SourceMode.CLOUD
    assert next_source_mode(SourceMode.CLOUD) is SourceMode.BOTH
    assert next_source_mode(SourceMode.BOTH) is SourceMode.LOCAL


def test_source_mode_values_are_lowercase_strings() -> None:
    assert SourceMode.LOCAL.value == "local"
    assert SourceMode.CLOUD.value == "cloud"
    assert SourceMode.BOTH.value == "both"
