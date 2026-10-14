"""Point the test run's default library at a temporary folder before lilbee builds ``cfg``.

Every test restores ``cfg`` to the root built here, and a thread that outlives
its test writes there. A module of its own runs once, though ``conftest.py`` is
imported under two names.
"""

import os
import sys
import tempfile

if "lilbee.core.config" in sys.modules:
    raise ImportError("import tests._suite_data_root before lilbee.core.config builds cfg")

SUITE_DATA_ROOT = tempfile.TemporaryDirectory(prefix="lilbee-tests-", ignore_cleanup_errors=True)
os.environ["LILBEE_DATA"] = SUITE_DATA_ROOT.name
os.environ.pop("LILBEE_DATA_ROOT", None)
