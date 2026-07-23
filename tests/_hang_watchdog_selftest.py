"""Manual self-test for the hang watchdog. Not collected in the normal suite.

Run explicitly:
    LILBEE_TEST_HANG_DUMP_S=3 LILBEE_TEST_HANG_DUMP_DIR=/tmp/hd \
      uv run pytest tests/_hang_watchdog_selftest.py -p no:xdist -p no:cacheprovider \
      --no-cov -o addopts='' -s
Expect: each test wedges, the watchdog dumps its stack after 3s and _exits the
worker; /tmp/hd/hang-master.txt holds the traceback naming the wedged frame.
"""

import os
import time

import pytest

# These tests wedge on purpose to exercise the watchdog, so they must never run
# in a normal or CI suite (where they would trip it and fail). Gated behind a
# dedicated var, separate from the LILBEE_TEST_HANG_DUMP_S that arms the
# watchdog, so CI can set the latter without collecting these.
pytestmark = pytest.mark.skipif(
    not os.environ.get("LILBEE_HANG_SELFTEST"),
    reason="watchdog self-test; set LILBEE_HANG_SELFTEST=1 to run",
)


def test_hang_via_sleep():
    # sleep releases the GIL; the easy case.
    time.sleep(120)


def test_hang_via_busy_loop():
    # A tight loop holds the GIL at the eval level; the hard case that proves
    # faulthandler's C watchdog does not need the GIL to dump.
    while True:
        pass
