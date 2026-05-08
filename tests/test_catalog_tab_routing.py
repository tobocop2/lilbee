"""Tests for the task-to-tab routing primitives.

The 6-tab catalog shell needs a deterministic mapping from a row's
ModelTask to the tab id that should render it. This is a leaf
primitive used during the per-tab refactor.
"""

from __future__ import annotations

import pytest

from lilbee.cli.tui.screens.catalog_utils import (
    ALL_TAB_IDS,
    TAB_CHAT,
    TAB_DISCOVER,
    TAB_EMBED,
    TAB_ID_TO_TASK,
    TAB_LIBRARY,
    TAB_RERANK,
    TAB_VISION,
    TASK_TAB_IDS,
    TASK_TO_TAB_ID,
    task_to_tab_id,
)
from lilbee.modelhub.models import ModelTask


def test_all_tab_ids_in_keyboard_order() -> None:
    assert ALL_TAB_IDS == (
        TAB_DISCOVER,
        TAB_CHAT,
        TAB_EMBED,
        TAB_VISION,
        TAB_RERANK,
        TAB_LIBRARY,
    )


def test_task_tab_ids_excludes_discover_and_library() -> None:
    assert TAB_DISCOVER not in TASK_TAB_IDS
    assert TAB_LIBRARY not in TASK_TAB_IDS
    assert set(TASK_TAB_IDS) == {TAB_CHAT, TAB_EMBED, TAB_VISION, TAB_RERANK}


def test_task_to_tab_id_routes_each_task() -> None:
    assert task_to_tab_id(ModelTask.CHAT) == TAB_CHAT
    assert task_to_tab_id(ModelTask.EMBEDDING) == TAB_EMBED
    assert task_to_tab_id(ModelTask.VISION) == TAB_VISION
    assert task_to_tab_id(ModelTask.RERANK) == TAB_RERANK


def test_task_to_tab_id_accepts_string_form() -> None:
    assert task_to_tab_id("chat") == TAB_CHAT
    assert task_to_tab_id("embedding") == TAB_EMBED
    assert task_to_tab_id("vision") == TAB_VISION
    assert task_to_tab_id("rerank") == TAB_RERANK


def test_task_to_tab_id_rejects_unknown_task() -> None:
    with pytest.raises(KeyError, match="unknown task"):
        task_to_tab_id("frontier")
    with pytest.raises(KeyError, match="unknown task"):
        task_to_tab_id("")


def test_every_modeltask_has_a_tab() -> None:
    for task in ModelTask:
        assert task_to_tab_id(task) in TASK_TAB_IDS


def test_tab_id_to_task_is_inverse_of_task_to_tab_id() -> None:
    for task, tab_id in TASK_TO_TAB_ID.items():
        assert TAB_ID_TO_TASK[tab_id] is task


def test_tab_id_to_task_covers_all_task_tabs() -> None:
    assert set(TAB_ID_TO_TASK.keys()) == set(TASK_TAB_IDS)
