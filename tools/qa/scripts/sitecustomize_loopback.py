"""QA-matrix-only Python startup hook: allow loopback crawls.

The QA harness's e2e crawler test serves a static HTML fixture from a
local ``http.server`` on 127.0.0.1, then invokes ``lilbee add --crawl
http://127.0.0.1:<port>`` against it. lilbee's URL filter rightly blocks
private and reserved IPs in production but does not yet expose an opt-in
env var for development / testing (tracked as bb-235r).

Until that env-var lands in lilbee, this module gets copied into the
Python install's ``sitecustomize.py`` slot by the qa-matrix workflow on
the lanes that need it. The patch is gated on ``LILBEE_QA_LANE`` so
nothing happens outside the QA harness even if a copy of this file ends
up on a machine's import path by accident.
"""

from __future__ import annotations

import os

if os.environ.get("LILBEE_QA_LANE") in {"l1-pypi", "l2-binary"}:
    try:
        from lilbee.crawler import url_filter

        url_filter._BLOCKED_NETWORKS = ()
        url_filter.get_blocked_networks = lambda: ()
    except ImportError:
        pass
