"""QA-matrix-only Python startup hook: allow loopback crawls.

The QA harness's e2e crawler test serves a static HTML fixture from a
local ``http.server`` on 127.0.0.1, then invokes ``lilbee add --crawl
http://127.0.0.1:<port>`` against it. lilbee's URL filter rightly blocks
private and reserved IPs in production but does not yet expose an opt-in
env var for development / testing (tracked as bb-235r).

Until that env-var lands in lilbee, the qa-matrix workflow copies this
file into the runner's ``sitecustomize.py`` slot. The patch is gated on
``LILBEE_QA_LANE`` so nothing happens outside the QA harness even if a
copy of this module ends up on a machine's import path by accident.

The lane-env-var name and the lane values are duplicated as string
literals here on purpose: this script runs at Python startup and cannot
import ``conftest.LaneName`` (conftest isn't on the import path of an
arbitrary Python invocation). Keep the values aligned with
``LaneName`` / ``LANE_ENV_VAR`` in ``tools/qa/conftest.py``.
"""

from __future__ import annotations

import os

if os.environ.get("LILBEE_QA_LANE") in {"l1-pypi", "l2-binary"}:
    try:
        from lilbee.crawler import url_filter
    except ImportError:
        # lilbee is not installed in this interpreter yet (e.g. the
        # workflow ran the cp before `pip install lilbee` completed).
        # The next Python invocation re-evaluates this hook; nothing to
        # patch in the meantime.
        pass
    else:
        # Reassign the public `get_blocked_networks` reader; lilbee's URL
        # filter calls that function rather than reading
        # `_BLOCKED_NETWORKS` directly. Reassigning a public name is
        # still a runtime monkey-patch; it stays here only until lilbee
        # exposes a real opt-in env (bb-235r).
        url_filter.get_blocked_networks = lambda: ()
