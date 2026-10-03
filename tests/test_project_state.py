"""The analyze state in state.toml."""

import tomllib
from datetime import datetime

from lilbee.core.project_state import (
    STATE_FILE_NAME,
    ProjectState,
    dismiss_tip,
    mark_analyzed,
    read_state,
)


def test_a_project_with_no_state_file_reads_as_never_analyzed(tmp_path):
    assert read_state(tmp_path) == ProjectState(analyzed_at=None, tip_dismissed=False)


def test_mark_analyzed_and_dismiss_tip_round_trip_and_keep_each_other(tmp_path):
    mark_analyzed(tmp_path)
    dismiss_tip(tmp_path)
    state = read_state(tmp_path)
    assert state.tip_dismissed is True
    assert state.analyzed_at is not None
    assert datetime.fromisoformat(state.analyzed_at).utcoffset().total_seconds() == 0
    stored = tomllib.loads((tmp_path / STATE_FILE_NAME).read_text(encoding="utf-8"))
    assert set(stored) == {"analyzed_at", "tip_dismissed"}


def test_a_state_write_creates_the_data_root(tmp_path):
    root = tmp_path / "new" / "root"
    dismiss_tip(root)
    assert read_state(root).tip_dismissed is True


def test_a_corrupt_state_file_reads_as_empty_and_logs(tmp_path, caplog):
    (tmp_path / STATE_FILE_NAME).write_text("not = [valid", encoding="utf-8")
    assert read_state(tmp_path) == ProjectState()
    assert "Ignoring unreadable" in caplog.text


def test_a_corrupt_state_file_is_replaced_by_the_next_write(tmp_path):
    (tmp_path / STATE_FILE_NAME).write_text("not = [valid", encoding="utf-8")
    dismiss_tip(tmp_path)
    assert read_state(tmp_path) == ProjectState(analyzed_at=None, tip_dismissed=True)


def test_values_of_the_wrong_type_read_as_unset(tmp_path):
    (tmp_path / STATE_FILE_NAME).write_text(
        'analyzed_at = 5\ntip_dismissed = "yes"\n', encoding="utf-8"
    )
    assert read_state(tmp_path) == ProjectState(analyzed_at=None, tip_dismissed=False)


def test_an_unreadable_state_path_reads_as_empty(tmp_path, caplog):
    (tmp_path / STATE_FILE_NAME).mkdir()
    assert read_state(tmp_path) == ProjectState()
    assert "Ignoring unreadable" in caplog.text
