"""The chat window an agent client needs, shared by the launchers and `lilbee agent-config`."""

from __future__ import annotations

# Floor on the served chat window for an agent client. Agents open with a large
# baseline prompt (system prompt, tool schemas, agent instructions): opencode's
# first turn measures ~32k tokens and Claude Code's ~47k, plus reserved output.
# The RAM-derived chat_n_ctx_target default is smaller on most hosts, so a
# window below this floor rejects the first message. hermes also refuses a
# window under 64000. The dynamic picker still clamps to the model's trained
# context and device memory, so asking for the floor never over-allocates.
AGENT_CHAT_CTX_FLOOR = 65536


def agent_chat_ctx_target(configured: int) -> int:
    """Lift a configured chat-context target to the agent floor, never lowering a larger one."""
    return max(configured, AGENT_CHAT_CTX_FLOOR)
