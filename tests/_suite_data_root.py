"""Point the test run's library, engine slot and cache at a temporary folder.

Every test restores ``cfg`` to the root built here, and a thread that outlives
its test writes there. A module of its own runs once, though ``conftest.py`` is
imported under two names. It must run before lilbee builds ``cfg`` and before
huggingface_hub reads its constants, which both happen at import.
"""

import os
import sys
import tempfile
from pathlib import Path

if {"lilbee.core.config", "huggingface_hub"} & sys.modules.keys():
    raise ImportError("import tests._suite_data_root before lilbee and huggingface_hub")

SUITE_DATA_ROOT = tempfile.TemporaryDirectory(prefix="lilbee-tests-", ignore_cleanup_errors=True)
os.environ["LILBEE_DATA"] = SUITE_DATA_ROOT.name
os.environ.pop("LILBEE_DATA_ROOT", None)
os.environ["LILBEE_ENGINE_DIR"] = os.path.join(SUITE_DATA_ROOT.name, "engine-slot")
os.environ["HF_HOME"] = os.path.join(SUITE_DATA_ROOT.name, "huggingface")

# Imported after the environment above: lilbee's package init imports
# huggingface_hub's constants, which read it.
import lilbee.core.system  # noqa: E402

platform_default_cache_dir = lilbee.core.system.default_cache_dir
_RUN_CACHE_DIR = Path(SUITE_DATA_ROOT.name) / "cache"


def _run_cache_dir() -> Path:
    """The run's stand-in for the platform cache directory."""
    return _RUN_CACHE_DIR


# lilbee has no environment variable for its cache location, so the run
# patches the one function that gives it; the platform original above stays
# importable for the tests that check its formula.
lilbee.core.system.default_cache_dir = _run_cache_dir
