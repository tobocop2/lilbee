"""Point the test run's library, engine slot and Hugging Face home at a temporary folder.

Every test restores ``cfg`` to the root built here, and a thread that outlives
its test writes there. A module of its own runs once, though ``conftest.py`` is
imported under two names. It must run before lilbee builds ``cfg`` and before
huggingface_hub reads ``HF_HOME``, which both happen at import.
"""

import os
import sys
import tempfile

if {"lilbee.core.config", "huggingface_hub"} & sys.modules.keys():
    raise ImportError("import tests._suite_data_root before lilbee and huggingface_hub")

SUITE_DATA_ROOT = tempfile.TemporaryDirectory(prefix="lilbee-tests-", ignore_cleanup_errors=True)
os.environ["LILBEE_DATA"] = SUITE_DATA_ROOT.name
os.environ.pop("LILBEE_DATA_ROOT", None)
os.environ["LILBEE_ENGINE_DIR"] = os.path.join(SUITE_DATA_ROOT.name, "engine-slot")
os.environ["HF_HOME"] = os.path.join(SUITE_DATA_ROOT.name, "huggingface")
