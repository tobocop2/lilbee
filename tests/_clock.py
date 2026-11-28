"""A fixed clock and a process time zone for the date range tests."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

needs_tzset = pytest.mark.skipif(not hasattr(time, "tzset"), reason="no time.tzset on this OS")


class FrozenClock(datetime):
    """A ``datetime`` whose ``now`` is a fixed instant."""

    frozen = datetime(2026, 3, 22, 15, 30, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(cls.frozen.timestamp(), tz)
