"""The litellm package installed in the integration environment."""

from __future__ import annotations

from importlib.metadata import packages_distributions
from importlib.util import find_spec

import pytest


@pytest.mark.skipif(find_spec("litellm") is None, reason="the litellm extra is not installed")
def test_exactly_one_installed_distribution_provides_litellm() -> None:
    # crawl4ai's unclecode-litellm fork writes the same litellm/ files, and the
    # directory then mixes two versions depending on install order.
    assert set(packages_distributions()["litellm"]) == {"litellm"}
