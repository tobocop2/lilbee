"""The integration rerun covers the model-download tests and nothing else."""

from __future__ import annotations

import pytest

from tests.integration import conftest as integration_conftest
from tests.integration.conftest import _MODEL_DOWNLOAD_FIXTURES, pytest_collection_modifyitems

# Literals, not the conftest constants: reading those back would assert nothing.
EXPECTED_RERUNS = {"reruns": 2, "reruns_delay": 10}


class _CollectedItem:
    """A collected test item, reduced to what the collection hook reads."""

    def __init__(self, *fixturenames: str) -> None:
        self.fixturenames = list(fixturenames)
        self.markers: list[pytest.MarkDecorator] = []

    def get_closest_marker(self, name: str) -> pytest.Mark | None:
        return None

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.markers.append(marker)

    def marker_names(self) -> list[str]:
        return [marker.name for marker in self.markers]

    def marker(self, name: str) -> pytest.MarkDecorator:
        return next(marker for marker in self.markers if marker.name == name)


@pytest.mark.parametrize("fixture_name", sorted(_MODEL_DOWNLOAD_FIXTURES))
def test_a_model_download_test_is_rerun(fixture_name: str) -> None:
    item = _CollectedItem("tmp_path", fixture_name)

    pytest_collection_modifyitems([item])

    assert "flaky" in item.marker_names()
    assert item.marker("flaky").kwargs == EXPECTED_RERUNS


def test_a_test_that_downloads_nothing_is_not_rerun() -> None:
    """A genuine failure is reported on its first attempt, not retried."""
    item = _CollectedItem("tmp_path", "run_async")

    pytest_collection_modifyitems([item])

    assert "flaky" not in item.marker_names()


def test_every_test_still_gets_the_integration_timeout() -> None:
    """The rerun marker is added beside the timeout default, not in place of it."""
    items = [_CollectedItem("rag_pipeline"), _CollectedItem("tmp_path")]

    pytest_collection_modifyitems(items)

    assert [item.marker("timeout").args for item in items] == [(180,), (180,)]


def test_the_scoped_fixture_names_still_exist() -> None:
    """A renamed fixture would silently drop the rerun, so pin the names."""
    missing = [name for name in _MODEL_DOWNLOAD_FIXTURES if not hasattr(integration_conftest, name)]

    assert missing == []
