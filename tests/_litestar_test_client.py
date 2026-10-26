"""``create_test_client`` without Litestar's default logging setup."""

from functools import partial

from litestar.testing import create_test_client as _create_test_client

create_test_client = partial(_create_test_client, logging_config=None)
