"""Tiny stdlib client for a running ``lilbee serve``.

Hits the real product HTTP surface, the same routes the TUI and the MCP server
use: ``GET /api/search`` for retrieval and ``PATCH /api/config`` for the
retrieval knobs (which lands in ``apply_settings_update`` and invalidates the
in-process caches). Stdlib only (urllib) so it runs under whatever python the pod
has, with no dependency on the lilbee venv.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

_TIMEOUT_S = 30.0


@dataclass
class LilbeeClient:
    base_url: str
    token: str

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def health(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/health", headers=self._headers())
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return False

    def search(self, query: str, top_k: int, scope: str = "raw") -> list[dict[str, Any]]:
        """Return the raw ``/api/search`` document list for *query*.

        ``scope="raw"`` restricts to ingested docs/code (the demo corpus), the
        same scope the SKILL.md tells agents to use for code lookups.
        """
        params = urllib.parse.urlencode({"q": query, "top_k": top_k, "chunk_type": scope})
        req = urllib.request.Request(f"{self.base_url}/api/search?{params}", headers=self._headers())
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        # The route returns either a bare list or a {"results": [...]} envelope
        # depending on version; accept both so the harness is not version-brittle.
        if isinstance(payload, dict):
            return payload.get("results") or payload.get("documents") or []
        return payload

    def patch_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Apply retrieval knobs via the real settings boundary; return the response."""
        body = json.dumps(updates).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/config", data=body, headers=self._headers(), method="PATCH"
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))

    def get_config(self) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}/api/config", headers=self._headers())
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
