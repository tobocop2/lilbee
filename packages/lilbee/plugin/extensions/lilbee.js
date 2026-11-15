import { MCP_COMMAND, MCP_NAME } from "../command.js"

export default function lilbee(pi) {
  const [command, ...args] = MCP_COMMAND
  pi.registerMcpServer(MCP_NAME, { command, args })
}
