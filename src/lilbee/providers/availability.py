"""Install check for the client library behind the provider backends."""

from __future__ import annotations

import functools
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

PROVIDER_SDK_MISSING_MESSAGE = (
    "Remote and API models need the lilbee[providers] extra. If you installed lilbee "
    "with uv tool install, run your install command again with providers added to the "
    "extras. If you installed lilbee with pip, run: pip install 'lilbee[providers]'."
)


@functools.cache
def provider_sdk_available() -> bool:
    """Return True if the provider client distribution is installed.

    Reads package metadata instead of importing the native module, so the
    Settings screen's compose does not pay for the import.
    """
    # circular: the factory imports this module
    from lilbee.providers import factory

    try:
        _dist_version(factory.SDK_DIST)
    except PackageNotFoundError:
        return False
    return True
