import { readdirSync, readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { MCP_COMMAND, MCP_NAME } from "./command.js"

export const SKILLS_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "skills")

const FRONTMATTER = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/

// A top-level key's value: inline, or the indented lines below it. A block scalar
// (> folds the lines into spaces, | keeps the newlines) and a plain value that
// continues on indented lines both read as the YAML text they stand for.
function field(block, key) {
  const lines = block.split(/\r?\n/)
  const at = lines.findIndex((line) => line.startsWith(`${key}:`))
  if (at < 0) return undefined
  const inline = lines[at].slice(key.length + 1).trim()
  const rest = []
  for (const line of lines.slice(at + 1)) {
    if (line.trim() !== "" && !/^\s/.test(line)) break
    rest.push(line.trim())
  }
  while (rest.length > 0 && rest[rest.length - 1] === "") rest.pop()
  const blockScalar = inline.match(/^([>|])[+-]?$/)
  if (blockScalar) return blockScalar[1] === ">" ? rest.filter(Boolean).join(" ") : rest.join("\n")
  const joined = [inline, ...rest].filter((part) => part !== "").join(" ")
  return joined.replace(/^(["'])(.*)\1$/, "$2")
}

export function readSkills(dir = SKILLS_DIR) {
  return readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const file = path.join(dir, entry.name, "SKILL.md")
      const parsed = readFileSync(file, "utf8").replace(/^\uFEFF/, "").match(FRONTMATTER)
      if (!parsed) throw new Error(`${file} has no frontmatter`)
      return {
        id: entry.name,
        name: field(parsed[1], "name") ?? entry.name,
        description: field(parsed[1], "description"),
        path: file,
        content: parsed[2],
      }
    })
}

// opencode 1.x: edit the resolved config.
export async function server() {
  return {
    config: async (config) => {
      config.mcp = config.mcp ?? {}
      config.mcp[MCP_NAME] = config.mcp[MCP_NAME] ?? { type: "local", command: [...MCP_COMMAND] }
      config.skills = config.skills ?? {}
      config.skills.paths = [...(config.skills.paths ?? []), SKILLS_DIR]
    },
  }
}

// opencode 2.x: register through the plugin context.
export async function setup(ctx) {
  const skills = readSkills()
  await ctx.mcp.transform((editor) => {
    if (!editor.get(MCP_NAME)) editor.set(MCP_NAME, { type: "local", command: [...MCP_COMMAND] })
  })
  await ctx.skill.transform((editor) => {
    for (const skill of skills) if (!editor.get(skill.id)) editor.add(skill)
  })
}

export default { id: "lilbee", server, setup }
