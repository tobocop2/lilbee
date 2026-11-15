import assert from "node:assert/strict"
import { execFileSync } from "node:child_process"
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import path from "node:path"
import { test } from "node:test"
import { fileURLToPath } from "node:url"

import lilbeeExtension from "../plugin/extensions/lilbee.js"
import plugin, { readSkills, server, setup, SKILLS_DIR } from "../plugin/server.js"

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..")
const MCP_ENTRY = { type: "local", command: ["lilbee", "mcp"] }

function stubSkillEditor(existing = []) {
  const skills = new Map(existing.map((skill) => [skill.id, skill]))
  return { skills, get: (id) => skills.get(id), add: (skill) => skills.set(skill.id, skill) }
}

function stubMcpEditor(existing = {}) {
  const servers = new Map(Object.entries(existing))
  return { servers, get: (name) => servers.get(name), set: (name, config) => servers.set(name, config) }
}

function stubContext(mcpEditor, skillEditor) {
  return {
    mcp: { transform: async (callback) => callback(mcpEditor) },
    skill: { transform: async (callback) => callback(skillEditor) },
  }
}

test("readSkills returns both skills with name, description and body", () => {
  const skills = readSkills()
  assert.deepEqual(skills.map((skill) => skill.id).sort(), ["lilbee-mcp", "lilbee-mcp-wiki"])
  for (const skill of skills) {
    assert.equal(skill.name, skill.id)
    assert.ok(skill.description.length > 0)
    assert.ok(path.isAbsolute(skill.path))
    assert.ok(skill.content.startsWith("\n# ") || skill.content.startsWith("# "))
    assert.ok(!skill.content.includes("description:"))
  }
})

test("the default export is one definition that serves opencode 1.x and 2.x", () => {
  assert.equal(plugin.id, "lilbee")
  assert.equal(plugin.server, server)
  assert.equal(plugin.setup, setup)
})

test("opencode 1.x config hook adds the server and the skills directory", async () => {
  const config = {}
  await (await server()).config(config)
  assert.deepEqual(config.mcp.lilbee, MCP_ENTRY)
  assert.deepEqual(config.skills.paths, [SKILLS_DIR])
})

test("opencode 1.x config hook keeps the user's own entries", async () => {
  const own = { type: "remote", url: "https://example.test/mcp" }
  const config = { mcp: { lilbee: own, other: MCP_ENTRY }, skills: { paths: ["/mine"] } }
  await (await server()).config(config)
  assert.deepEqual(config.mcp.lilbee, own)
  assert.deepEqual(config.mcp.other, MCP_ENTRY)
  assert.deepEqual(config.skills.paths, ["/mine", SKILLS_DIR])
})

test("opencode 2.x setup registers the server and both skills", async () => {
  const mcp = stubMcpEditor()
  const skills = stubSkillEditor()
  await setup(stubContext(mcp, skills))
  assert.deepEqual(mcp.servers.get("lilbee"), MCP_ENTRY)
  assert.deepEqual([...skills.skills.keys()].sort(), ["lilbee-mcp", "lilbee-mcp-wiki"])
  assert.equal(skills.skills.get("lilbee-mcp").path, path.join(SKILLS_DIR, "lilbee-mcp", "SKILL.md"))
})

test("opencode 2.x setup keeps a server and a skill the user already defined", async () => {
  const own = { type: "remote", url: "https://example.test/mcp" }
  const mine = { id: "lilbee-mcp", name: "mine", path: "/mine/SKILL.md", content: "" }
  const mcp = stubMcpEditor({ lilbee: own })
  const skills = stubSkillEditor([mine])
  await setup(stubContext(mcp, skills))
  assert.deepEqual(mcp.servers.get("lilbee"), own)
  assert.equal(skills.skills.get("lilbee-mcp"), mine)
  assert.ok(skills.skills.has("lilbee-mcp-wiki"))
})

test("the pi extension registers the lilbee server", () => {
  const calls = []
  lilbeeExtension({ registerMcpServer: (...args) => calls.push(args) })
  assert.deepEqual(calls, [["lilbee", { command: "lilbee", args: ["mcp"] }]])
})

test("package.json points opencode and pi at existing plugin paths", () => {
  const manifest = JSON.parse(readFileSync(path.join(ROOT, "package.json"), "utf8"))
  assert.equal(manifest.name, "lilbee")
  assert.ok(manifest.keywords.includes("pi-package"))
  assert.ok(manifest.keywords.includes("opencode-plugin"))
  assert.equal(manifest.exports["./server"], "./plugin/server.js")
  assert.ok(existsSync(path.join(ROOT, manifest.exports["./server"])))
  const paths = [...manifest.pi.extensions, ...manifest.pi.skills]
  assert.ok(paths.length > 0)
  for (const entry of paths) assert.ok(existsSync(path.join(ROOT, entry)), entry)
})

test("the packed tarball holds every file the entry points import", () => {
  const out = execFileSync("npm", ["pack", "--dry-run", "--json"], { cwd: ROOT, encoding: "utf8" })
  const files = JSON.parse(out)[0].files.map((file) => file.path)
  for (const required of [
    "package.json",
    "plugin/server.js",
    "plugin/command.js",
    "plugin/extensions/lilbee.js",
    "skills/lilbee-mcp/SKILL.md",
    "skills/lilbee-mcp-wiki/SKILL.md",
  ]) {
    assert.ok(files.includes(required), `${required} missing from ${files.join(", ")}`)
  }
  assert.deepEqual(
    files.filter((file) => file.startsWith("test/") || file.startsWith(".")),
    [],
  )
})

function skillDirWith(text) {
  const dir = mkdtempSync(path.join(tmpdir(), "lilbee-skill-"))
  mkdirSync(path.join(dir, "one"))
  writeFileSync(path.join(dir, "one", "SKILL.md"), text)
  return dir
}

test("a folded description reads as one line of text", () => {
  const [skill] = readSkills(skillDirWith("---\nname: one\ndescription: >\n  First part\n  second part.\nother: x\n---\nBody\n"))
  assert.equal(skill.description, "First part second part.")
  assert.equal(skill.content, "Body\n")
})

test("a blank line inside a folded block does not end the description", () => {
  const [skill] = readSkills(skillDirWith("---\nname: one\ndescription: >\n  First part\n\n  second part.\nother: x\n---\nBody\n"))
  assert.equal(skill.description, "First part second part.")
})

test("a folded block ends at the next unindented key after a blank line", () => {
  const [skill] = readSkills(skillDirWith("---\nname: one\ndescription: >\n  First part\n\nother: x\n---\nBody\n"))
  assert.equal(skill.description, "First part")
})

test("a literal description keeps its line breaks", () => {
  const [skill] = readSkills(skillDirWith("---\nname: one\ndescription: |-\n  First\n  second\n---\nBody\n"))
  assert.equal(skill.description, "First\nsecond")
})

test("a plain description continues on the next indented line", () => {
  const [skill] = readSkills(skillDirWith("---\nname: one\ndescription: First\n  second\n---\nBody\n"))
  assert.equal(skill.description, "First second")
})

test("a quoted description loses its quotes", () => {
  const [skill] = readSkills(skillDirWith("---\nname: 'one'\ndescription: \"Quoted text\"\n---\nBody\n"))
  assert.equal(skill.name, "one")
  assert.equal(skill.description, "Quoted text")
})

test("a skill file with a byte order mark and CRLF lines parses", () => {
  const [skill] = readSkills(skillDirWith("﻿---\r\nname: one\r\ndescription: >\r\n  Windows\r\n  text\r\n---\r\nBody\r\n"))
  assert.equal(skill.name, "one")
  assert.equal(skill.description, "Windows text")
})

test("a skill file without frontmatter is refused with its path", () => {
  assert.throws(() => readSkills(skillDirWith("no frontmatter")), /one\/SKILL\.md has no frontmatter/)
})
