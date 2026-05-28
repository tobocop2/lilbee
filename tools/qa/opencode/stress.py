"""Production-readiness stress + soak harness for the lilbee fleet.

Drives the real production endpoints -- ``/v1/chat/completions`` (the tool-calling
path opencode uses) and ``/api/search`` (the retrieval / shared-embedder path the
MCP server uses) -- to answer the questions that decide production confidence:

* Concurrency: can the fleet serve a realistic multi-agent dev workload (many
  concurrent tool-calling + search turns) without failing or wedging?
* Long conversations: does a long multi-turn session stay up?
* Token-limit exhaustion: when a prompt/conversation exceeds the model's context,
  does it resolve gracefully (clean error, server survives) or crash?

Run on a host with a cached chat model, pointing LILBEE_DATA at a workspace whose
config.toml pins that model and whose data dir holds an indexed corpus::

    LILBEE_DATA=.../workspace/qwen3/.lilbee LILBEE_MODELS_DIR=/root/models \
      uv run python tools/qa/opencode/stress.py --agents 8 --rounds 3 --turns 12

Exit code is 0 only if every sub-test passes (no crash; concurrency clean;
token-exhaustion survived). Token-exhaustion also reports whether resolution was
*graceful* (a clean context-length error) vs a generic error -- both survive, but
graceful is the production bar.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

from lilbee.cli.launchers.server import ensure_server_running
from lilbee.core.config import cfg

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "lilbee_search",
        "description": "Search the indexed Godot 4 reference for relevant chunks.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    },
}
_QUERIES = [
    "AStarGrid2D diagonal_mode",
    "TileMapLayer set_cell",
    "Node _process delta",
    "Object connect signal flags",
    "NavigationServer2D map_get_path",
    "Sprite2D texture region",
]


def _server() -> tuple[str, dict[str, str], str]:
    (token, port), _ = ensure_server_running()
    return f"http://127.0.0.1:{port}", {"Authorization": f"Bearer {token}"}, cfg.chat_model


async def _chat(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    model: str,
    messages: list[dict],
    *,
    tools: list | None = None,
    max_tokens: int = 200,
) -> tuple[int, float, httpx.Response | Exception]:
    body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    if tools:
        body["tools"] = tools
    t0 = time.monotonic()
    try:
        r = await client.post(f"{base}/v1/chat/completions", headers=headers, json=body, timeout=240)
        return r.status_code, time.monotonic() - t0, r
    except Exception as exc:  # network / timeout / reset = the fleet wedged
        return -1, time.monotonic() - t0, exc


async def _search(client: httpx.AsyncClient, base: str, headers: dict[str, str], q: str) -> int:
    try:
        r = await client.get(f"{base}/api/search", headers=headers, params={"q": q, "top_k": 5}, timeout=90)
        return r.status_code
    except Exception:
        return -1


async def _agent(idx, client, base, headers, model, rounds, rec) -> None:
    for r in range(rounds):
        q = _QUERIES[(idx + r) % len(_QUERIES)]
        code, dt, _ = await _chat(
            client, base, headers, model,
            [{"role": "user", "content": f"Look up {q} in the Godot 4 reference and explain it briefly."}],
            tools=[_SEARCH_TOOL], max_tokens=160,
        )
        rec.append(("chat", code, dt))
        rec.append(("search", await _search(client, base, headers, q), 0.0))


async def test_concurrency(base, headers, model, agents, rounds) -> bool:
    rec: list[tuple[str, int, float]] = []
    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[_agent(i, client, base, headers, model, rounds, rec) for i in range(agents)])
    wall = time.monotonic() - t0
    chat_lat = [d for k, c, d in rec if k == "chat"]
    chat_codes = [c for k, c, d in rec if k == "chat"]
    search_codes = [c for k, c, d in rec if k == "search"]
    ok = sum(1 for c in chat_codes if c == 200)
    busy = sum(1 for c in chat_codes if c == 429)
    fail = sum(1 for c in chat_codes if c not in (200, 429))
    s_ok = sum(1 for c in search_codes if c == 200)
    p95 = sorted(chat_lat)[max(0, int(len(chat_lat) * 0.95) - 1)] if chat_lat else 0.0
    print(f"[concurrency] {agents} agents x {rounds} rounds in {wall:.1f}s")
    print(f"  chat:   {ok}/{len(chat_codes)} ok, {busy} busy(429), {fail} hard-fail; "
          f"p50={statistics.median(chat_lat):.1f}s p95={p95:.1f}s")
    print(f"  search: {s_ok}/{len(search_codes)} ok")
    # 429 is acceptable backpressure; a hard failure (5xx / -1) is not.
    return fail == 0 and ok > 0


async def test_long_conversation(base, headers, model, turns) -> bool:
    msgs: list[dict] = []
    async with httpx.AsyncClient() as client:
        for t in range(turns):
            msgs.append({"role": "user",
                         "content": f"Turn {t}: name one more Godot 4 class useful for 2D games and why (2 sentences)."})
            code, dt, r = await _chat(client, base, headers, model, msgs, max_tokens=120)
            if code != 200:
                body = r.text[:200] if isinstance(r, httpx.Response) else repr(r)
                print(f"[long-conv] stopped at turn {t} (status {code}): {body}")
                alive = await _search(client, base, headers, "Node")
                print(f"[long-conv] serve responsive after: {alive == 200}")
                return alive == 200  # surviving an overflow late in a long convo is acceptable
            content = r.json()["choices"][0]["message"].get("content", "") if isinstance(r, httpx.Response) else ""
            msgs.append({"role": "assistant", "content": content})
    chars = sum(len(m["content"]) for m in msgs)
    print(f"[long-conv] completed {turns} turns cleanly; conversation grew to ~{chars} chars")
    return True


async def test_token_exhaustion(base, headers, model) -> bool:
    filler = "Godot 4 AStarGrid2D pathfinding tile map cell navigation server query. " * 60000
    async with httpx.AsyncClient() as client:
        code, dt, r = await _chat(client, base, headers, model,
                                  [{"role": "user", "content": filler + "\nSummarize the above."}], max_tokens=80)
        body = r.text[:400] if isinstance(r, httpx.Response) else repr(r)
        try:
            alive = (await client.get(f"{base}/api/health", headers=headers, timeout=10)).status_code
        except Exception:
            alive = 0
    graceful = code in (400, 413) and "context" in body.lower()
    survived = alive == 200
    print(f"[token-exhaustion] status={code} ({dt:.1f}s); serve-alive-after={survived}")
    print(f"[token-exhaustion] body: {body}")
    print(f"[token-exhaustion] graceful(clean context error)={graceful}  survived(no crash)={survived}")
    if survived and not graceful:
        print("[token-exhaustion] NOTE: survives but does not return a clean context_length_exceeded -- "
              "production UX gap (generic error reaches the client).")
    return survived


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--skip", default="", help="comma list: concurrency,long,token")
    args = ap.parse_args()
    skip = set(args.skip.split(",")) if args.skip else set()

    base, headers, model = _server()
    print(f"serve at {base}, model={model}\n")

    results: dict[str, bool] = {}
    if "concurrency" not in skip:
        results["concurrency"] = asyncio.run(test_concurrency(base, headers, model, args.agents, args.rounds))
    if "long" not in skip:
        results["long_conversation"] = asyncio.run(test_long_conversation(base, headers, model, args.turns))
    if "token" not in skip:
        results["token_exhaustion"] = asyncio.run(test_token_exhaustion(base, headers, model))

    print("\n=== STRESS SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
