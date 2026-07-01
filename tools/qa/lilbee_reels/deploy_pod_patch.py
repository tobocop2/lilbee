#!/usr/bin/env python3
"""Idempotent, self-verifying patch for the bigmodel demo pod.
Applies (1) the chat-stream over-buffer fix (bb-4m1) and (2) --reasoning-format none
on the CHAT adapter (bb-rbb), regardless of starting branch state. Run on the pod:
  python3 deploy_pod_patch.py /workspace/lilbee
Exit 0 only when BOTH are verified present.
"""
import sys, pathlib
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/lilbee")
client = root / "src/lilbee/providers/fleet/client.py"
adapters = root / "src/lilbee/providers/fleet/adapters.py"

CLIENT_OLD = '''    buffer = ""  # leading text held back as a potential bare call until resolved
    saw_native = False
    try:
        for item in items:
            if isinstance(item, ToolCallDelta):
                yield from _flush_plain(buffer)
                buffer, saw_native = "", True
                yield item
            elif isinstance(item, TokenUsage | StreamFinish):
                yield from _recover_buffer(buffer)
                buffer = ""
                yield item
            elif saw_native or _passthrough_text(buffer, item):
                yield from _flush_plain(buffer)
                buffer = ""
                yield item
            else:
                buffer += item
        yield from _recover_buffer(buffer)'''
CLIENT_NEW = '''    buffer = ""  # leading text held back as a potential bare call until resolved
    committed = False  # leading text already streamed as plain (or a native call seen): never buffer again
    try:
        for item in items:
            if isinstance(item, ToolCallDelta):
                yield from _flush_plain(buffer)
                buffer, committed = "", True
                yield item
            elif isinstance(item, TokenUsage | StreamFinish):
                yield from _recover_buffer(buffer)
                buffer = ""
                yield item
            elif committed or _passthrough_text(buffer, item):
                yield from _flush_plain(buffer)
                buffer = ""
                committed = True
                yield item
            else:
                buffer += item
        yield from _recover_buffer(buffer)'''

def patch(path, old, new, marker, label):
    txt = path.read_text()
    if marker in txt:
        print(f"  [skip] {label}: already present"); return True
    if old not in txt:
        print(f"  [FAIL] {label}: expected block not found (file drifted) -> {path}"); return False
    path.write_text(txt.replace(old, new, 1))
    ok = marker in path.read_text()
    print(f"  [{'done' if ok else 'FAIL'}] {label}"); return ok

ok1 = patch(client, CLIENT_OLD, CLIENT_NEW, "committed or _passthrough_text", "bb-4m1 streaming fix")
# adapters: CHAT extra_args, tolerate either compact or already-patched form
a_txt = adapters.read_text()
if '"--jinja", "--reasoning-format", "none"' in a_txt:
    print("  [skip] bb-rbb reasoning-format: already present"); ok2 = True
elif 'extra_args=("--jinja",)' in a_txt:
    # replace the FIRST occurrence (CHAT spec); LLM_RERANK also uses ("--jinja",) but
    # the CHAT spec is the first ROLE_SPECS entry, so first-replace targets CHAT.
    adapters.write_text(a_txt.replace('extra_args=("--jinja",)', 'extra_args=("--jinja", "--reasoning-format", "none")', 1))
    ok2 = '"--jinja", "--reasoning-format", "none"' in adapters.read_text()
    print(f"  [{'done' if ok2 else 'FAIL'}] bb-rbb reasoning-format on CHAT")
else:
    print("  [FAIL] bb-rbb: CHAT extra_args form not found"); ok2 = False

print("RESULT:", "ALL PATCHES VERIFIED" if (ok1 and ok2) else "PATCH FAILED")
sys.exit(0 if (ok1 and ok2) else 1)

# NOTE: after patching, RESTART the lilbee server so Python re-imports the changed
# modules (the running process holds the old bytecode). On the bigmodel pod:
#   ssh -f lilbee-bigmodel 'bash /workspace/start_server.sh > /workspace/server.log 2>&1'
