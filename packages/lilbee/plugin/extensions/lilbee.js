import { MCP_COMMAND, MCP_NAME } from "../command.js"

// `lilbee launch pi` sets these for the `lilbee serve` session it starts or reuses.
const OPENAI_URL_ENV = "LILBEE_OPENAI_URL"
const MCP_URL_ENV = "LILBEE_MCP_URL"
const MODELS_ENV = "LILBEE_PI_MODELS"

// The running server over HTTP under the launcher, else a stdio `lilbee mcp`;
// none when the launcher was asked for no MCP.
function mcpServer(env) {
  if (env[MCP_URL_ENV]) {
    return { url: env[MCP_URL_ENV], headers: { Authorization: "Bearer ${LILBEE_TOKEN}" }, exposure: "direct" }
  }
  if (env[OPENAI_URL_ENV]) return undefined
  const [command, ...args] = MCP_COMMAND
  return { command, args }
}

export default function lilbee(pi, env = process.env) {
  if (env[OPENAI_URL_ENV]) {
    pi.registerProvider(MCP_NAME, {
      baseUrl: env[OPENAI_URL_ENV],
      api: "openai-completions",
      apiKey: "$LILBEE_TOKEN",
      models: JSON.parse(env[MODELS_ENV]),
    })
  }
  const server = mcpServer(env)
  // A second copy of this extension (an installed package plus the launcher's -e)
  // leaves the first copy's server in place: pi throws on a name another extension owns.
  if (server && !pi.getMcpServers().some((registered) => registered.name === MCP_NAME)) {
    pi.registerMcpServer(MCP_NAME, server)
  }
}
