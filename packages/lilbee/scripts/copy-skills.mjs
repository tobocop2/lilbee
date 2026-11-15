import { cpSync, rmSync } from "node:fs"

const from = new URL("../../../plugins/lilbee/skills", import.meta.url)
const to = new URL("../skills", import.meta.url)

rmSync(to, { recursive: true, force: true })
cpSync(from, to, { recursive: true })
