// QA event tap: appends one JSON line per opencode event to
// .lilbee/qa-events.jsonl in the cell workspace, so the harness reads real
// signals (tool dispatched, session idle, session error) instead of scraping
// the tmux pane. Deployed per cell by workspace.py into .opencode/plugins/.
import { appendFileSync } from "node:fs"

export const QaEvents = async ({ directory }) => {
  const sink = `${directory}/.lilbee/qa-events.jsonl`
  const log = (record) => {
    try {
      appendFileSync(sink, JSON.stringify({ ts: Date.now(), ...record }) + "\n")
    } catch {
      // The tap must never break the session it observes.
    }
  }
  log({ type: "qa.plugin.loaded" })
  return {
    event: async ({ event }) => {
      log({ type: event?.type ?? "unknown" })
    },
    "tool.execute.before": async (input) => {
      log({ type: "qa.tool.before", tool: input?.tool ?? "" })
    },
    "tool.execute.after": async (input) => {
      log({ type: "qa.tool.after", tool: input?.tool ?? "" })
    },
  }
}
