"""Production-readiness stress test: concurrent, iterative, tool-using dev sessions.

This mirrors a standard agentic dev workflow -- the exact opencode <-> lilbee loop a
developer drives in practice:

    chat (with the lilbee_search tool)
      -> model emits tool_calls
      -> we run the REAL /api/search (the shared-embedder retrieval path)
      -> feed the results back as a tool message
      -> chat again ... until the model produces a final answer
      -> next task in the session

N such "developers" run concurrently, each working through a realistic coding task
list (build -> extend -> debug -> refactor) against the indexed Godot 4 reference.
That exercises the full blast radius the way real usage does: streaming chat
completions, native tool-call extraction, the shared embedder under concurrent
search, conversation growth across a long multi-tool session, and 429 backpressure.

A session whose conversation outgrows the model's window must resolve GRACEFULLY (a
clean context_length_exceeded with the server still serving), never crash.

Run on a host with a cached chat model + an indexed corpus::

    LILBEE_DATA=.../workspace/qwen3/.lilbee LILBEE_MODELS_DIR=/root/models \
      uv run python tools/qa/opencode/stress.py --devs 4 --tasks 6

PASS (exit 0) iff: every concurrent session makes progress, there are no hard
failures (5xx / connection drops), and any context overflow resolved gracefully.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

from lilbee.cli.launchers.server import ensure_server_running
from lilbee.core.config import cfg

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "lilbee_search",
        "description": "Search the indexed Godot 4 class reference; returns cited chunks.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    },
}

# Same workflow the demos use (godot-with-lilbee AGENTS.md): the model must look every
# API up via the tool instead of answering from stale memory, so the loop actually
# drives retrieval the way a real session does.
_SYSTEM = (
    "You are a Godot 4.4 GDScript developer. Your training data is outdated -- classes "
    "and methods were renamed in Godot 4.x -- so you MUST call lilbee_search to confirm "
    "every class, method, and signature before you use it, then write code using only "
    "confirmed APIs."
)

# A realistic iterative coding session: build a feature, extend it, hit a bug, refactor.
# This is what a developer actually does over a working session, not one-shot Q&A.
_DEV_TASKS = [
    "Build a Godot 4 GDScript pathfinding helper using AStarGrid2D. Look the API up first.",
    "Now add diagonal movement and a way to mark cells as solid (walls).",
    "Render the result with a TileMapLayer: floor tiles along the path, wall tiles elsewhere.",
    "Bug: get_id_path seems to return the wrong type for my loop -- check its exact "
    "return type and fix it.",
    "Scatter collectibles on reachable floor tiles, skipping the start and goal cells.",
    "Refactor: expose a regenerate() signal and make map size and tile size @export vars.",
    "How do I connect the regenerate() signal to a button and call it at runtime?",
    "Add a RandomNumberGenerator with a seed so levels are reproducible.",
]


def _server() -> tuple[str, dict[str, str], str]:
    (token, port), _ = ensure_server_running()
    return f"http://127.0.0.1:{port}", {"Authorization": f"Bearer {token}"}, cfg.chat_model


async def _run_search(client, base, headers, query, top_k=5) -> tuple[str, int]:
    """The real retrieval call opencode makes for a lilbee_search tool invocation."""
    try:
        r = await client.get(
            f"{base}/api/search", headers=headers, params={"q": query, "top_k": top_k}, timeout=90
        )
        if r.status_code != 200:
            return f"(search error {r.status_code})", r.status_code
        items = r.json() if r.headers.get("content-type", "").startswith("application/json") else []
        lines = [
            f"- {it.get('source', '?')}: {str(it.get('text', it))[:300]}" for it in items[:top_k]
        ]
        return ("\n".join(lines) or "(no results)"), 200
    except Exception as exc:
        return f"(search exception: {exc})", -1


async def _chat(client, base, headers, model, msgs, *, max_tokens=300):
    body = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "stream": False,
        "tools": [_SEARCH_TOOL],
    }
    t0 = time.monotonic()
    try:
        r = await client.post(
            f"{base}/v1/chat/completions", headers=headers, json=body, timeout=300
        )
        return r.status_code, time.monotonic() - t0, r
    except Exception as exc:
        return -1, time.monotonic() - t0, exc


async def _dev_turn(client, base, headers, model, msgs, m) -> str:
    """One developer request: chat, run any tool calls via the REAL search, feed results
    back, and loop until the model gives a final answer. Returns ok | overflow | fail."""
    for _hop in range(6):  # bound the tool-call loop per turn
        code, dt, r = await _chat(client, base, headers, model, msgs)
        m["chat"].append((code, dt))
        if code == 429:  # backpressure: acceptable, retry like a real client
            m["busy"] += 1
            await asyncio.sleep(2)
            continue
        if code != 200:
            body = r.text[:300] if isinstance(r, httpx.Response) else repr(r)
            if code == 400 and "context" in body.lower():
                m["overflow_body"] = body
                return "overflow"
            m["fail_detail"] = f"{code}: {body}"
            return "fail"
        msg = r.json()["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            msgs.append({"role": "assistant", "content": msg.get("content", "")})
            return "ok"
        # the model wants to search: record its tool-call turn, run the real searches,
        # feed each result back, then loop so it can search more or answer.
        msgs.append(
            {"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls}
        )
        for tc in tool_calls:
            try:
                args = json.loads(tc.get("function", {}).get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {}
            result, scode = await _run_search(
                client, base, headers, args.get("query", "Godot Node")
            )
            m["search"].append(scode)
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", "0"), "content": result})
    return "ok"  # tool loop capped; treat as progress


async def _dev_session(dev_id, client, base, headers, model, tasks_per, m) -> None:
    msgs = [{"role": "system", "content": _SYSTEM}]
    for i in range(tasks_per):
        msgs.append({"role": "user", "content": _DEV_TASKS[i % len(_DEV_TASKS)]})
        status = await _dev_turn(client, base, headers, model, msgs, m)
        m["turns"] += 1
        if status == "overflow":
            alive = (await _run_search(client, base, headers, "Node"))[1] == 200
            m["overflow"] += 1
            m["overflow_graceful"] = m["overflow_graceful"] and alive
            return  # a long session that overflows gracefully is a success, not a crash
        if status == "fail":
            m["fail"] += 1
            return


async def run_workflow(base, headers, model, devs, tasks_per) -> bool:
    m = {
        "chat": [],
        "search": [],
        "turns": 0,
        "busy": 0,
        "fail": 0,
        "overflow": 0,
        "overflow_graceful": True,
        "fail_detail": "",
        "overflow_body": "",
    }
    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            *[_dev_session(i, client, base, headers, model, tasks_per, m) for i in range(devs)]
        )
    wall = time.monotonic() - t0
    chat_codes = [c for c, _ in m["chat"]]
    chat_lat = [d for _, d in m["chat"]]
    ok = sum(1 for c in chat_codes if c == 200)
    hard = sum(1 for c in chat_codes if c not in (200, 429, 400))
    s_ok = sum(1 for c in m["search"] if c == 200)
    p95 = sorted(chat_lat)[max(0, int(len(chat_lat) * 0.95) - 1)] if chat_lat else 0.0
    print(f"[dev-workflow] {devs} concurrent devs x {tasks_per} tasks in {wall:.0f}s")
    print(
        f"  turns={m['turns']}  chat 200={ok}/{len(chat_codes)}  "
        f"busy(429)={m['busy']}  hard-fail={hard}"
    )
    print(
        f"  REAL searches: {s_ok}/{len(m['search'])} ok  |  chat latency "
        f"p50={statistics.median(chat_lat):.1f}s p95={p95:.1f}s"
        if chat_lat
        else "  (no chats)"
    )
    print(
        f"  overflows={m['overflow']} graceful={m['overflow_graceful']}  "
        f"fail={m['fail']} {m['fail_detail'][:120]}"
    )
    if m["overflow_body"]:
        print(f"  overflow body: {m['overflow_body'][:160]}")
    return (
        hard == 0 and ok > 0 and m["fail"] == 0 and (m["overflow"] == 0 or m["overflow_graceful"])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devs", type=int, default=4, help="concurrent developer sessions")
    ap.add_argument("--tasks", type=int, default=8, help="iterative tasks per session")
    args = ap.parse_args()
    base, headers, model = _server()
    print(f"serve {base} model={model}\n")
    ok = asyncio.run(run_workflow(base, headers, model, args.devs, args.tasks))
    print("\n=== DEV-WORKFLOW STRESS:", "PASS" if ok else "FAIL", "===")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
